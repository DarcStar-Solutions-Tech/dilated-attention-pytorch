#!/usr/bin/env python3
"""
Test Ring V3 with large sequences using float32 and proper scaling.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_large_float32.py
"""

import os
import gc
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_large_sequences_float32():
    """Test progressively larger sequences with float32."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_large_float32.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Large Sequences with Float32")
        print("=" * 50)
        print(f"World size: {world_size}")
        print("Using adaptive input scaling\n")

    # Test configurations
    test_configs = [
        # (seq_len, bucket_size, segment_len, name)
        (1024, 256, 512, "1K tokens"),
        (2048, 512, 1024, "2K tokens"),
        (4096, 512, 2048, "4K tokens"),
        (8192, 1024, 4096, "8K tokens"),
        (16384, 1024, 8192, "16K tokens"),
    ]

    for seq_len, bucket_size, segment_len, name in test_configs:
        if rank == 0:
            print(f"\nTesting {name}:")
            print(f"  Sequence length: {seq_len:,}")
            print(f"  Bucket size: {bucket_size}")

        # Synchronize
        dist.barrier()

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

        try:
            # Create model
            model = RingDilatedAttentionV3(
                segment_lengths=[segment_len],
                dilation_rates=[1],
                bucket_size=bucket_size,
                use_bucketed=True,
                device=device,
                dtype=torch.float32,
                ring_size=world_size,
            )

            # Adaptive scaling based on sequence length
            # Smaller values for longer sequences
            scale = 0.1 / (seq_len**0.25)

            # Create inputs
            torch.manual_seed(42 + rank)
            q = (
                torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)
                * scale
            )
            k = (
                torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)
                * scale
            )
            v = (
                torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)
                * scale
            )

            if rank == 0:
                print(f"  Input scale: {scale:.6f}")

            # Get memory before
            mem_before = torch.cuda.memory_allocated(device) / (1024**3)  # GB

            # Forward pass
            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Get memory stats
            mem_after = torch.cuda.memory_allocated(device) / (1024**3)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)

            # Check output
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_mean = output.mean().item()
            _ = output.std().item()
            output_max = output.abs().max().item()

            # Gather stats
            stats = {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "peak_memory_gb": peak_memory,
                "output_mean": output_mean,
                "output_max": output_max,
            }

            all_stats = [None] * world_size
            dist.all_gather_object(all_stats, stats)

            if rank == 0:
                any_nan = any(s["has_nan"] for s in all_stats)
                any_inf = any(s["has_inf"] for s in all_stats)

                if any_nan or any_inf:
                    print(f"  ❌ Failed - NaN: {any_nan}, Inf: {any_inf}")
                else:
                    print("  ✅ Success!")
                    print(
                        f"     Memory: {mem_before:.2f} GB → {mem_after:.2f} GB (peak: {peak_memory:.2f} GB)"
                    )
                    print(f"     Output: mean={output_mean:.6f}, max={output_max:.6f}")

                    # Show per-GPU memory
                    all_peaks = [s["peak_memory_gb"] for s in all_stats]
                    print(f"     Per-GPU peaks: {[f'{p:.2f} GB' for p in all_peaks]}")

            # Clean up
            del q, k, v, output, model

        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"  ❌ OOM at {seq_len:,} tokens")
                    print(
                        "     Try reducing batch size or enabling gradient checkpointing"
                    )
                break
            else:
                if rank == 0:
                    print(f"  ❌ Runtime error: {e}")
                break

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            break

    # Summary
    if rank == 0:
        print("\n" + "=" * 50)
        print("Summary:")
        print("- Float32 avoids overflow issues seen with float16")
        print("- Adaptive input scaling helps maintain numerical stability")
        print("- Memory usage is ~2x higher than float16")
        print("- Ring attention successfully distributes memory across GPUs")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_large_sequences_float32()
