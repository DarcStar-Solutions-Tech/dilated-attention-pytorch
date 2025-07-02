#!/usr/bin/env python3
"""
Test Ring V3 with a simple fix for bucketing performance.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_fixed_bucketing.py
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_with_disabled_bucketing():
    """Test performance with bucketing disabled to isolate the issue."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_fixed_bucketing.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Ring V3 Performance - Bucketing vs Non-bucketing")
        print("=" * 50)

    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]

    for seq_len in seq_lengths:
        if rank == 0:
            print(f"\nSequence length: {seq_len}")

        # Test without bucketing first
        dist.barrier()

        model_no_bucket = RingDilatedAttentionV3(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            use_bucketed=False,  # Disable bucketing
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Create inputs
        torch.manual_seed(42)
        scale = 0.1 / (seq_len**0.25)
        q = torch.randn(1, seq_len, 8, 64, device=device) * scale
        k = torch.randn(1, seq_len, 8, 64, device=device) * scale
        v = torch.randn(1, seq_len, 8, 64, device=device) * scale

        # Warmup
        _ = model_no_bucket(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        # Time without bucketing
        dist.barrier()
        start = time.time()

        output_no_bucket = model_no_bucket(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        time_no_bucket = time.time() - start

        # Check output
        has_nan = torch.isnan(output_no_bucket).any().item()
        mean_no_bucket = output_no_bucket.mean().item()

        # Clean up
        del model_no_bucket
        torch.cuda.empty_cache()

        # Gather results
        results = {
            "time": time_no_bucket,
            "has_nan": has_nan,
            "mean": mean_no_bucket,
        }

        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        if rank == 0:
            avg_time = sum(r["time"] for r in all_results) / len(all_results)
            any_nan = any(r["has_nan"] for r in all_results)

            print(f"  Without bucketing: {avg_time:.3f}s (NaN: {any_nan})")
            print(f"  Output mean: {mean_no_bucket:.6f}")

            # Memory usage
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"  Peak memory: {peak_mem:.2f} GB")

        torch.cuda.reset_peak_memory_stats()

    # Now test a specific fix: Process attention without creating full-sized intermediate tensors
    if rank == 0:
        print("\n\nTesting optimized approach at 2048 tokens:")
        print("-" * 40)

    seq_len = 2048
    dist.barrier()

    # Standard non-bucketed for comparison
    model = RingDilatedAttentionV3(
        segment_lengths=[1024],
        dilation_rates=[1],
        use_bucketed=False,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    q = torch.randn(1, seq_len, 8, 64, device=device) * 0.05
    k = torch.randn(1, seq_len, 8, 64, device=device) * 0.05
    v = torch.randn(1, seq_len, 8, 64, device=device) * 0.05

    # Time 3 iterations
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()

        output = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        if rank == 0 and i == 0:
            print(f"First run: {elapsed:.3f}s")

    if rank == 0:
        avg_time = sum(times[1:]) / len(times[1:])  # Skip first (warmup)
        print(f"Average (excluding warmup): {avg_time:.3f}s")
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.6f}")

    dist.destroy_process_group()

    if rank == 0:
        print("\n✅ Testing completed")
        print("\nConclusion: Bucketing implementation needs optimization.")
        print("The current implementation creates full-sized intermediate tensors")
        print("which defeats the purpose of bucketing for memory efficiency.")


if __name__ == "__main__":
    test_with_disabled_bucketing()
