#!/usr/bin/env python3
"""
Detailed test of Ring V3 bucketed processing on multiple GPUs.
Run with: torchrun --nproc_per_node=2 test_ring_v3_multi_gpu_detailed.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_multi_gpu_bucketed():
    """Test bucketed processing with multiple GPUs with detailed checks."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 test_ring_v3_multi_gpu_detailed.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Multi-GPU Bucketed Processing")
        print("=" * 50)
        print(f"World size: {world_size}")

    # Synchronize
    dist.barrier()

    # Test configurations
    test_configs = [
        # (seq_len, bucket_size, use_bucketed, test_name)
        (512, 256, False, "Small seq, no buckets"),
        (512, 256, True, "Small seq, with buckets"),
        (1024, 256, False, "Medium seq, no buckets"),
        (1024, 256, True, "Medium seq, with buckets"),
        (2048, 512, True, "Large seq, with buckets"),
    ]

    for seq_len, bucket_size, use_bucketed, test_name in test_configs:
        if rank == 0:
            print(f"\n{test_name}:")
            print(
                f"  seq_len={seq_len}, bucket_size={bucket_size}, use_bucketed={use_bucketed}"
            )

        # Clear memory
        torch.cuda.empty_cache()

        try:
            # Create model
            model = RingDilatedAttentionV3(
                segment_lengths=[512],
                dilation_rates=[1],
                bucket_size=bucket_size,
                use_bucketed=use_bucketed,
                device=device,
                dtype=torch.float32,  # Use float32 for better numerical stability
                ring_size=world_size,
            )

            # Create identical inputs on all ranks for consistency
            torch.manual_seed(42)
            q = torch.randn(1, seq_len, 4, 32, device=device)
            k = torch.randn(1, seq_len, 4, 32, device=device)
            v = torch.randn(1, seq_len, 4, 32, device=device)

            # Forward pass
            output = model(q, k, v, is_causal=False)

            # Check for NaN/Inf
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            # Gather statistics from all ranks
            output_mean = output.mean().item()
            output_std = output.std().item()
            output_min = output.min().item()
            output_max = output.max().item()

            # Synchronize results
            if world_size > 1:
                stats = [
                    output_mean,
                    output_std,
                    output_min,
                    output_max,
                    float(has_nan),
                    float(has_inf),
                ]
                all_stats = [None] * world_size
                dist.all_gather_object(all_stats, stats)

                if rank == 0:
                    # Check consistency across ranks
                    all_means = [s[0] for s in all_stats]
                    all_stds = [s[1] for s in all_stats]
                    all_nans = [bool(s[4]) for s in all_stats]
                    all_infs = [bool(s[5]) for s in all_stats]

                    print("  ✅ Forward pass completed")
                    print(f"     Output shape: {output.shape}")
                    print(f"     Means by rank: {[f'{m:.6f}' for m in all_means]}")
                    print(f"     Stds by rank: {[f'{s:.6f}' for s in all_stds]}")

                    if any(all_nans):
                        print(
                            f"     ⚠️  NaN detected on ranks: {[i for i, x in enumerate(all_nans) if x]}"
                        )
                    if any(all_infs):
                        print(
                            f"     ⚠️  Inf detected on ranks: {[i for i, x in enumerate(all_infs) if x]}"
                        )

                    # Check if outputs are consistent
                    mean_diff = max(all_means) - min(all_means)
                    if mean_diff < 1e-5:
                        print("     ✅ Outputs consistent across ranks")
                    else:
                        print(
                            f"     ❌ Outputs differ across ranks (diff={mean_diff:.6f})"
                        )
            else:
                print("  ✅ Forward pass completed")
                print(
                    f"     Output stats: mean={output_mean:.6f}, std={output_std:.6f}"
                )
                if has_nan:
                    print("     ⚠️  Contains NaN!")
                if has_inf:
                    print("     ⚠️  Contains Inf!")

        except Exception as e:
            print(f"[Rank {rank}] ❌ Error: {e}")
            import traceback

            traceback.print_exc()

        # Synchronize before next test
        dist.barrier()

    # Test causal mode
    if rank == 0:
        print("\n\nTesting Causal Mode:")

    model = RingDilatedAttentionV3(
        segment_lengths=[256],
        dilation_rates=[1],
        bucket_size=128,
        use_bucketed=True,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    # Small test for causal
    seq_len = 512
    q = torch.randn(1, seq_len, 4, 32, device=device)
    k = torch.randn(1, seq_len, 4, 32, device=device)
    v = torch.randn(1, seq_len, 4, 32, device=device)

    try:
        output_causal = model(q, k, v, is_causal=True)

        has_nan = torch.isnan(output_causal).any().item()
        output_mean = output_causal.mean().item()

        if rank == 0:
            if has_nan:
                print("  ❌ Causal mode produces NaN!")
            else:
                print(f"  ✅ Causal mode works! Mean={output_mean:.6f}")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Causal mode error: {e}")

    # Clean up
    dist.destroy_process_group()
    if rank == 0:
        print("\n✅ All tests completed")


if __name__ == "__main__":
    test_multi_gpu_bucketed()
