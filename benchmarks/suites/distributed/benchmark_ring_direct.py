#!/usr/bin/env python3
"""
Direct benchmark of ring attention without Hilbert optimization to find the bottleneck.
"""

import gc
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

# Import the base ring attention directly
from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)


def setup(rank: int, world_size: int):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"[GPU {rank}] Process initialized")


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def benchmark_worker(rank: int, world_size: int, seq_len: int, results_dict):
    """Benchmark worker for each GPU."""

    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    base_segment = min(4096, seq_len // 4)
    segment_lengths = [base_segment, base_segment * 2]
    dilation_rates = [8, 16]

    # Ensure divisibility
    max_segment = max(segment_lengths)
    if seq_len % max_segment != 0:
        seq_len = ((seq_len // max_segment) + 1) * max_segment

    if rank == 0:
        print(f"\nTesting {seq_len:,} tokens on {world_size} GPUs")
        print(f"  Segments: {segment_lengths}")
        print(f"  Dilation: {dilation_rates}")

    try:
        # Create model - using base ring attention
        model = RingDilatedAttentionHybridOptimizedV2(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
        )

        # Create test tensors
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Synchronize
        dist.barrier()

        # Memory tracking
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        if rank == 0:
            print("  Warming up...")

        with torch.no_grad():
            _ = model(q, k, v)
        torch.cuda.synchronize()

        dist.barrier()

        # Benchmark
        if rank == 0:
            print("  Benchmarking 3 iterations...")

        times = []
        for i in range(3):
            torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append(end - start)

            if rank == 0:
                print(f"    Iteration {i + 1}: {end - start:.3f}s")
            del output

        # Calculate metrics
        avg_time = np.mean(times)
        tokens_per_sec = seq_len / avg_time
        peak_mem = torch.cuda.max_memory_allocated(rank) / 1024**3

        # Cleanup
        del q, k, v, model
        torch.cuda.empty_cache()

        dist.barrier()

        if rank == 0:
            results_dict[seq_len] = {
                "success": True,
                "avg_time": avg_time,
                "tokens_per_sec": tokens_per_sec,
                "peak_memory_gb": peak_mem,
            }

            print("\n  ✓ Success!")
            print(f"    Time: {avg_time:.3f}s")
            print(f"    Throughput: {tokens_per_sec:,.0f} tokens/sec")
            print(f"    Memory per GPU: {peak_mem:.2f} GB")

    except Exception as e:
        if rank == 0:
            print(f"  ✗ Error: {str(e)}")
            results_dict[seq_len] = {"success": False, "error": str(e)}
    finally:
        cleanup()


def main():
    """Run direct ring attention benchmark."""

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("Error: This benchmark requires at least 2 GPUs")
        return

    world_size = 2  # Use 2 GPUs

    print("=" * 80)
    print("DIRECT RING ATTENTION BENCHMARK (NO HILBERT)")
    print("=" * 80)

    # Test sequence lengths
    seq_lengths = [16384, 32768, 65536]

    manager = mp.Manager()
    results_dict = manager.dict()

    for seq_len in seq_lengths:
        mp.spawn(
            benchmark_worker,
            args=(world_size, seq_len, results_dict),
            nprocs=world_size,
            join=True,
        )

        if seq_len in results_dict and not results_dict[seq_len]["success"]:
            break

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Seq Len':>10} | {'Time (s)':>10} | {'Throughput':>12} | {'Mem/GPU':>8}")
    print("-" * 50)

    for seq_len, result in results_dict.items():
        if result["success"]:
            print(
                f"{seq_len:>10,} | {result['avg_time']:>10.3f} | {result['tokens_per_sec']:>12,.0f} | {result['peak_memory_gb']:>8.2f}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
