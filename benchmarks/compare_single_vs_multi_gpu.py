#!/usr/bin/env python3
"""
Compare single GPU vs multi-GPU performance for Ring Attention.

This script runs the same workload on:
1. Single GPU (simulated ring)
2. Multiple GPUs (true distributed ring)
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dilated_attention_pytorch import RingDilatedAttentionV2Collective


def run_single_gpu_test(seq_len, batch_size, num_heads, head_dim):
    """Run ring attention on a single GPU."""
    print("\n" + "=" * 60)
    print("Single GPU Test")
    print("=" * 60)

    device = torch.device("cuda:0")
    dtype = torch.float16

    # Create ring attention module with ring_size=1
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Filter segment lengths
    segment_lengths = [s for s in segment_lengths if s <= seq_len]
    dilation_rates = dilation_rates[: len(segment_lengths)]

    model = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,  # Single GPU
        device=device,
        dtype=dtype,
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v)

    torch.cuda.synchronize()

    # Time the forward pass
    start_time = time.perf_counter()

    num_iters = 10
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(q, k, v)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iters * 1000  # ms
    memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print("Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print("\nResults:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(f"  Throughput: {batch_size * seq_len / (avg_time / 1000):.0f} tokens/s")

    return avg_time, memory_mb


def setup(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def run_multi_gpu_worker(
    rank, world_size, seq_len, batch_size, num_heads, head_dim, results_queue
):
    """Worker process for multi-GPU test."""
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    # Create ring attention module
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    segment_lengths = [s for s in segment_lengths if s <= seq_len]
    dilation_rates = dilation_rates[: len(segment_lengths)]

    model = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=dtype,
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v)

    dist.barrier()
    torch.cuda.synchronize()

    # Time the forward pass
    start_time = time.perf_counter()

    num_iters = 10
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(q, k, v)

    torch.cuda.synchronize()
    dist.barrier()

    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iters * 1000  # ms
    memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Send results to main process
    if rank == 0:
        results_queue.put((avg_time, memory_mb))

    cleanup()


def run_multi_gpu_test(world_size, seq_len, batch_size, num_heads, head_dim):
    """Run ring attention on multiple GPUs."""
    print("\n" + "=" * 60)
    print(f"Multi-GPU Test ({world_size} GPUs)")
    print("=" * 60)

    # Create queue for results
    results_queue = mp.Queue()

    # Spawn processes
    mp.spawn(
        run_multi_gpu_worker,
        args=(world_size, seq_len, batch_size, num_heads, head_dim, results_queue),
        nprocs=world_size,
        join=True,
    )

    # Get results
    avg_time, memory_mb = results_queue.get()

    print("Configuration:")
    print(f"  World size: {world_size} GPUs")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print("\nResults:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Memory per GPU: {memory_mb:.2f} MB")
    print(f"  Total memory: {memory_mb * world_size:.2f} MB")
    print(f"  Throughput: {batch_size * seq_len / (avg_time / 1000):.0f} tokens/s")

    return avg_time, memory_mb


def main():
    parser = argparse.ArgumentParser(
        description="Compare single vs multi-GPU Ring Attention"
    )
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")

    args = parser.parse_args()

    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run single GPU test
    single_time, single_memory = run_single_gpu_test(
        args.seq_len, args.batch_size, args.num_heads, args.head_dim
    )

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Run multi-GPU test if we have multiple GPUs
    if n_gpus >= 2:
        multi_time, multi_memory_per_gpu = run_multi_gpu_test(
            2, args.seq_len, args.batch_size, args.num_heads, args.head_dim
        )

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print("Single GPU:")
        print(f"  Time: {single_time:.2f} ms")
        print(f"  Memory: {single_memory:.2f} MB")
        print("\nMulti-GPU (2 GPUs):")
        print(f"  Time: {multi_time:.2f} ms")
        print(f"  Memory per GPU: {multi_memory_per_gpu:.2f} MB")
        print(f"  Total memory: {multi_memory_per_gpu * 2:.2f} MB")
        print(f"\nSpeedup: {single_time / multi_time:.2f}x")
        print(f"Memory per GPU reduction: {single_memory / multi_memory_per_gpu:.2f}x")
        print(f"Communication overhead: {multi_time - single_time:.2f} ms")
    else:
        print("\nOnly 1 GPU available, skipping multi-GPU test")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
