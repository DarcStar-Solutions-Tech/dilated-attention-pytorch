#!/usr/bin/env python3
"""
Simple multi-GPU test for Ring Attention V2 Collective.

This script tests the actual multi-GPU functionality of ring attention
using PyTorch distributed and the collective operations implementation.
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


def setup(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def run_ring_attention_test(rank, world_size, seq_len, batch_size, num_heads, head_dim):
    """Run ring attention on a single GPU in the distributed setup."""
    print(f"[Rank {rank}] Starting on GPU {rank}")

    # Setup distributed
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    # Create ring attention module
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Filter segment lengths to be <= seq_len
    segment_lengths = [s for s in segment_lengths if s <= seq_len]
    dilation_rates = dilation_rates[: len(segment_lengths)]

    try:
        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=dtype,
        )

        # Create inputs - each rank has the full Q but only a chunk of K/V
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

        # Synchronize before timing
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

        # Get memory usage
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Only rank 0 prints results
        if rank == 0:
            print(f"\n{'=' * 60}")
            print("Ring Attention V2 Collective - Multi-GPU Test")
            print(f"{'=' * 60}")
            print(f"World size: {world_size} GPUs")
            print(f"Sequence length: {seq_len}")
            print(f"Batch size: {batch_size}")
            print(f"Num heads: {num_heads}")
            print(f"Head dim: {head_dim}")
            print(f"Segment lengths: {segment_lengths}")
            print(f"Dilation rates: {dilation_rates}")
            print("\nResults:")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Memory per GPU: {memory_mb:.2f} MB")
            print(f"  Total memory: {memory_mb * world_size:.2f} MB")
            print(
                f"  Throughput: {batch_size * seq_len / (avg_time / 1000):.0f} tokens/s"
            )
            print(f"{'=' * 60}")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Test Ring Attention on multiple GPUs")
    parser.add_argument(
        "--world-size", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")

    args = parser.parse_args()

    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus < args.world_size:
        print(f"Error: Requested {args.world_size} GPUs but only {n_gpus} available")
        return

    print(f"Starting multi-GPU test with {args.world_size} GPUs")
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Spawn processes
    mp.spawn(
        run_ring_attention_test,
        args=(
            args.world_size,
            args.seq_len,
            args.batch_size,
            args.num_heads,
            args.head_dim,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
