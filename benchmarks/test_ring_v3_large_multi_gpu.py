#!/usr/bin/env python3
"""
Test Ring V3 with large sequences on multiple GPUs.
Run with: torchrun --nproc_per_node=2 test_ring_v3_large_multi_gpu.py
"""

import os
import gc
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_large_sequences_multi_gpu():
    """Test increasingly large sequences on multiple GPUs."""

    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 test_ring_v3_large_multi_gpu.py")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Large Sequences on Multiple GPUs")
        print("=" * 50)
        print(f"World size: {world_size}")
        print(f"Each GPU should use ~O(n/{world_size}) memory\n")

    # Test configurations - progressively larger
    test_configs = [
        # (seq_len, bucket_size, segment_len, name)
        (1024, 256, 512, "1K tokens"),
        (2048, 256, 1024, "2K tokens"),
        (4096, 512, 2048, "4K tokens"),
        (8192, 512, 4096, "8K tokens"),
        (16384, 1024, 8192, "16K tokens"),
        (32768, 1024, 16384, "32K tokens"),
    ]

    for seq_len, bucket_size, segment_len, name in test_configs:
        if rank == 0:
            print(f"\nTesting {name}:")
            print(f"  Sequence length: {seq_len:,}")
            print(f"  Per GPU: ~{seq_len // world_size:,} tokens for K,V")
            print(f"  Bucket size: {bucket_size}")

        # Synchronize before test
        dist.barrier()

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create model with bucketing and half precision
            model = RingDilatedAttentionV3(
                segment_lengths=[segment_len],
                dilation_rates=[1],
                bucket_size=bucket_size,
                use_bucketed=True,
                grad_checkpoint_buckets=False,  # Can enable for more memory savings
                device=device,
                dtype=torch.float16,  # Use half precision
                ring_size=world_size,
            )

            # Create minimal test inputs
            batch_size = 1
            num_heads = 8
            head_dim = 64

            # Create inputs - same seed for consistency
            torch.manual_seed(42 + rank)
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            k = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            v = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )

            # Memory before forward
            mem_before = torch.cuda.memory_allocated(device) / (1024**2)

            # Forward pass
            output = model(q, k, v, is_causal=False)

            # Ensure computation completes
            torch.cuda.synchronize()

            # Get memory stats
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            mem_after = torch.cuda.memory_allocated(device) / (1024**2)

            # Check output validity
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_mean = output.float().mean().item()
            output_std = output.float().std().item()

            # Gather stats from all ranks
            all_stats = None
            if world_size > 1:
                stats = {
                    "peak_memory": peak_memory,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "output_mean": output_mean,
                    "output_std": output_std,
                }
                all_stats = [None] * world_size
                dist.all_gather_object(all_stats, stats)

            # Report results
            if rank == 0:
                print(f"  ✅ Success!")
                print(f"     Memory before: {mem_before:.1f} MB")
                print(f"     Memory after: {mem_after:.1f} MB")
                print(f"     Peak memory: {peak_memory:.1f} MB")

                if all_stats:
                    all_peaks = [s["peak_memory"] for s in all_stats]
                    all_means = [s["output_mean"] for s in all_stats]
                    any_nan = any(s["has_nan"] for s in all_stats)
                    any_inf = any(s["has_inf"] for s in all_stats)

                    print(f"     All GPU peaks: {[f'{p:.1f}' for p in all_peaks]} MB")
                    print(f"     Output means: {[f'{m:.6f}' for m in all_means]}")

                    if any_nan:
                        print(f"     ⚠️  Warning: NaN detected!")
                    if any_inf:
                        print(f"     ⚠️  Warning: Inf detected!")
                else:
                    print(f"     Output mean: {output_mean:.6f}, std: {output_std:.6f}")

            # Clean up
            del q, k, v, output, model

        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_before_oom = torch.cuda.max_memory_allocated(device) / (1024**2)
                if rank == 0:
                    print(f"  ❌ OOM at {seq_len:,} tokens")
                    print(f"     Peak before OOM: {peak_before_oom:.1f} MB")
                break
            else:
                if rank == 0:
                    print(f"  ❌ Runtime error: {e}")
                break

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            break

    # Test with gradient checkpointing for even larger sequences
    if rank == 0:
        print("\n\nTesting with Gradient Checkpointing:")
        print("=" * 50)

    # Synchronize
    dist.barrier()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        seq_len = 8192  # Try 8K with checkpointing
        model = RingDilatedAttentionV3(
            segment_lengths=[4096],
            dilation_rates=[1],
            bucket_size=512,
            use_bucketed=True,
            grad_checkpoint_buckets=True,  # Enable gradient checkpointing
            device=device,
            dtype=torch.float16,
            ring_size=world_size,
        )

        model.train()  # Required for gradient checkpointing

        # Create inputs
        q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)
        k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)
        v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)

        # Forward pass
        _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)

        if rank == 0:
            print(f"8K tokens with gradient checkpointing:")
            print(f"  ✅ Success! Peak memory: {peak_memory:.1f} MB")

    except Exception as e:
        if rank == 0:
            print(f"  ❌ Failed with gradient checkpointing: {e}")

    # Clean up
    dist.destroy_process_group()

    if rank == 0:
        print("\n✅ Testing completed")


if __name__ == "__main__":
    test_large_sequences_multi_gpu()
