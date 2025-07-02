#!/usr/bin/env python3
"""
Performance diagnostic for Ring V3 to identify bottlenecks.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_performance.py
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_performance():
    """Diagnose performance issues."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_performance.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Ring V3 Performance Diagnostic")
        print("=" * 50)

    # Test different configurations
    configs = [
        (256, 128, "Tiny"),
        (512, 256, "Small"),
        (1024, 256, "Medium"),
    ]

    for seq_len, bucket_size, name in configs:
        if rank == 0:
            print(f"\n{name} test (seq_len={seq_len}):")

        dist.barrier()

        # Time model creation
        start = time.time()
        model = RingDilatedAttentionV3(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            bucket_size=bucket_size,
            use_bucketed=True,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )
        create_time = time.time() - start

        # Create inputs
        q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

        # Warmup
        _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        # Time forward pass
        dist.barrier()
        start = time.time()

        output = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        forward_time = time.time() - start

        # Check output
        has_nan = torch.isnan(output).any().item()

        # Gather timings
        timings = {
            "create_time": create_time,
            "forward_time": forward_time,
            "has_nan": has_nan,
        }

        all_timings = [None] * world_size
        dist.all_gather_object(all_timings, timings)

        if rank == 0:
            avg_forward = sum(t["forward_time"] for t in all_timings) / len(all_timings)
            any_nan = any(t["has_nan"] for t in all_timings)

            print(f"  Model creation: {create_time:.3f}s")
            print(f"  Forward pass: {avg_forward:.3f}s")
            print(f"  Has NaN: {any_nan}")

            # Estimate throughput
            total_flops = seq_len * seq_len * 4 * 32 * 4  # Rough estimate
            tflops = (total_flops / avg_forward) / 1e12
            print(f"  Estimated: {tflops:.2f} TFLOPS")

        del model, q, k, v, output
        torch.cuda.empty_cache()

    # Test bucketed vs non-bucketed
    if rank == 0:
        print("\n\nBucketed vs Non-bucketed Comparison:")
        print("-" * 40)

    seq_len = 512

    for use_bucketed in [False, True]:
        dist.barrier()

        model = RingDilatedAttentionV3(
            segment_lengths=[256],
            dilation_rates=[1],
            bucket_size=128 if use_bucketed else None,
            use_bucketed=use_bucketed,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

        # Time 5 iterations
        torch.cuda.synchronize()
        dist.barrier()

        start = time.time()
        for _ in range(5):
            _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        total_time = time.time() - start
        avg_time = total_time / 5

        all_times = [None] * world_size
        dist.all_gather_object(all_times, avg_time)

        if rank == 0:
            avg_across_gpus = sum(all_times) / len(all_times)
            mode = "Bucketed" if use_bucketed else "Non-bucketed"
            print(f"{mode}: {avg_across_gpus:.3f}s per forward")

    dist.destroy_process_group()

    if rank == 0:
        print("\n✅ Performance diagnostic completed")


if __name__ == "__main__":
    test_performance()
