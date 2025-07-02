#!/usr/bin/env python3
"""
Test Ring V3 with float32 precision for larger sequences.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_float32.py
"""

import os
import gc
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_float32_large_sequences():
    """Test large sequences with float32 to avoid overflow."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_float32.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Ring V3 with Float32 Precision")
        print("=" * 50)
        print(f"World size: {world_size}")
        print("Using float32 to avoid overflow issues\n")

    # Test configurations - same as before but with float32
    test_configs = [
        # (seq_len, bucket_size, segment_len, name)
        (1024, 256, 512, "1K tokens"),
        (2048, 256, 1024, "2K tokens"),
        (4096, 512, 2048, "4K tokens"),
        (8192, 512, 4096, "8K tokens"),
    ]

    for seq_len, bucket_size, segment_len, name in test_configs:
        if rank == 0:
            print(f"\nTesting {name}:")
            print(f"  Sequence length: {seq_len:,}")
            print(f"  Per GPU: ~{seq_len // world_size:,} tokens for K,V")

        # Synchronize before test
        dist.barrier()

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create model with float32
            model = RingDilatedAttentionV3(
                segment_lengths=[segment_len],
                dilation_rates=[1],
                bucket_size=bucket_size,
                use_bucketed=True,
                device=device,
                dtype=torch.float32,  # Use float32
                ring_size=world_size,
            )

            # Create inputs with controlled values to avoid overflow
            torch.manual_seed(42 + rank)
            batch_size = 1
            num_heads = 8
            head_dim = 64

            # Scale inputs to prevent overflow
            scale = 0.1 / (seq_len**0.25)  # Adaptive scaling

            q = (
                torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                * scale
            )
            k = (
                torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                * scale
            )
            v = (
                torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                * scale
            )

            # Memory before forward
            _ = torch.cuda.memory_allocated(device) / (1024**2)

            # Forward pass
            output = model(q, k, v, is_causal=False)

            # Ensure computation completes
            torch.cuda.synchronize()

            # Get memory stats
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            _ = torch.cuda.memory_allocated(device) / (1024**2)

            # Check output validity
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_mean = output.mean().item()
            output_std = output.std().item()
            output_max = output.abs().max().item()

            # Gather stats from all ranks
            stats = {
                "peak_memory": peak_memory,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "output_mean": output_mean,
                "output_std": output_std,
                "output_max": output_max,
            }

            all_stats = [None] * world_size
            dist.all_gather_object(all_stats, stats)

            # Report results
            if rank == 0:
                any_nan = any(s["has_nan"] for s in all_stats)
                any_inf = any(s["has_inf"] for s in all_stats)

                if any_nan or any_inf:
                    print(f"  ❌ Failed - NaN: {any_nan}, Inf: {any_inf}")
                else:
                    print(f"  ✅ Success!")
                    print(f"     Memory usage: {peak_memory:.1f} MB (peak)")
                    print(
                        f"     Output stats: mean={output_mean:.6f}, std={output_std:.6f}, max={output_max:.6f}"
                    )

                    # Check all GPUs
                    all_peaks = [s["peak_memory"] for s in all_stats]
                    all_maxs = [s["output_max"] for s in all_stats]
                    print(f"     All GPU peaks: {[f'{p:.1f}' for p in all_peaks]} MB")
                    print(f"     All output maxs: {[f'{m:.6f}' for m in all_maxs]}")

            # Clean up
            del q, k, v, output, model

        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"  ❌ OOM at {seq_len:,} tokens (even with float32)")
                break
            else:
                if rank == 0:
                    print(f"  ❌ Runtime error: {e}")
                import traceback

                traceback.print_exc()
                break

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            break

    # Test comparison: float16 vs float32 on same sequence
    if rank == 0:
        print("\n\nComparing Float16 vs Float32:")
        print("=" * 50)

    dist.barrier()

    seq_len = 1024
    for dtype, dtype_name in [(torch.float16, "float16"), (torch.float32, "float32")]:
        if rank == 0:
            print(f"\nTesting {dtype_name} with {seq_len} tokens:")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            model = RingDilatedAttentionV3(
                segment_lengths=[512],
                dilation_rates=[1],
                bucket_size=256,
                use_bucketed=True,
                device=device,
                dtype=dtype,
                ring_size=world_size,
            )

            # Use same scaling for both
            scale = 0.1
            q = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype) * scale
            k = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype) * scale
            v = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype) * scale

            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

            stats = {"has_nan": has_nan, "has_inf": has_inf, "peak_mem": peak_mem}
            all_stats = [None] * world_size
            dist.all_gather_object(all_stats, stats)

            if rank == 0:
                any_nan = any(s["has_nan"] for s in all_stats)
                any_inf = any(s["has_inf"] for s in all_stats)
                avg_peak = sum(s["peak_mem"] for s in all_stats) / len(all_stats)

                if any_nan or any_inf:
                    print(f"  ❌ {dtype_name}: NaN={any_nan}, Inf={any_inf}")
                else:
                    print(
                        f"  ✅ {dtype_name}: Success! Avg peak memory: {avg_peak:.1f} MB"
                    )

        except Exception as e:
            if rank == 0:
                print(f"  ❌ {dtype_name}: {e}")

    # Clean up
    dist.destroy_process_group()

    if rank == 0:
        print("\n✅ Float32 testing completed")
        print("\nRecommendations:")
        print("- Use float32 for sequences > 1K tokens to avoid overflow")
        print("- Consider adaptive input scaling based on sequence length")
        print("- Monitor memory usage as float32 uses 2x memory of float16")


if __name__ == "__main__":
    test_float32_large_sequences()
