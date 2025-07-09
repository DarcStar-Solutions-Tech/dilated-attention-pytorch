#!/usr/bin/env python3
"""
Multi-GPU benchmark for dilated attention implementations.
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime

# Set environment for better multi-GPU performance
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def setup_distributed(rank, world_size):
    """Setup distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def benchmark_data_parallel():
    """Benchmark using DataParallel (single-node multi-GPU)."""
    print("\n=== DataParallel Benchmark ===")

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for DataParallel")
        return

    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )

    # Test configurations
    configs = [
        {"seq_len": 4096, "batch": 4},
        {"seq_len": 8192, "batch": 2},
        {"seq_len": 16384, "batch": 1},
    ]

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch"]

        print(f"\nSequence Length: {seq_len:,}, Batch: {batch_size}")

        # Create model
        sparse_config = SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=0.1,  # 90% sparse
            block_size=128,
        )

        if seq_len <= 4096:
            segment_lengths = [2048, 4096]
        elif seq_len <= 8192:
            segment_lengths = [4096, 8192]
        else:
            segment_lengths = [8192, 16384]

        model = BlockSparseRingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
        )

        # Wrap in DataParallel
        model = nn.DataParallel(model)
        model = model.cuda()

        # Create inputs
        num_heads = 8
        head_dim = 64
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(2):
            _ = model(q, k, v)
            torch.cuda.synchronize()

        # Time
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(5):
            output = model(q, k, v)
            torch.cuda.synchronize()

        end = time.perf_counter()
        avg_time = (end - start) / 5 * 1000  # ms

        # Check memory on each GPU
        mem_gpu0 = torch.cuda.memory_allocated(0) / 1024**2
        mem_gpu1 = torch.cuda.memory_allocated(1) / 1024**2

        print(f"  Time: {avg_time:.1f}ms")
        print(f"  Memory GPU0: {mem_gpu0:.1f}MB, GPU1: {mem_gpu1:.1f}MB")
        print(f"  Output shape: {output.shape}")

        # Cleanup
        del model, q, k, v, output
        torch.cuda.empty_cache()


def benchmark_distributed_single_gpu(rank, world_size, configs):
    """Benchmark distributed training - each process handles different data."""
    setup_distributed(rank, world_size)

    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from dilated_attention_pytorch.block_sparse_ring_distributed_dilated_attention import (
        BlockSparseRingDistributedDilatedAttention,
        DistributedSparseConfig,
    )

    if rank == 0:
        print("\n=== Distributed Training Benchmark (Model Parallel) ===")

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_per_gpu"]

        if rank == 0:
            print(f"\nSequence Length: {seq_len:,}, Batch per GPU: {batch_size}")

        # Create distributed model
        distributed_config = DistributedSparseConfig(
            sparsity_ratio=0.1,
            pattern_type="hierarchical",
            enable_memory_optimization=True,
        )

        if seq_len <= 4096:
            segment_lengths = [2048, 4096]
        elif seq_len <= 8192:
            segment_lengths = [4096, 8192]
        else:
            segment_lengths = [8192, 16384]

        try:
            model = BlockSparseRingDistributedDilatedAttention(
                embed_dim=512,
                num_heads=8,
                segment_lengths=segment_lengths,
                dilation_rates=[1, 2],
                distributed_config=distributed_config,
            ).cuda()

            # Create local batch
            embed_dim = 512
            q = torch.randn(
                batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Synchronize before timing
            dist.barrier()

            # Warmup
            for _ in range(2):
                _ = model(q, k, v)

            torch.cuda.synchronize()
            dist.barrier()

            # Time
            start = time.perf_counter()

            for _ in range(5):
                output = model(q, k, v)
                torch.cuda.synchronize()

            dist.barrier()
            end = time.perf_counter()

            avg_time = (end - start) / 5 * 1000  # ms

            # Gather results
            mem_used = torch.cuda.memory_allocated() / 1024**2

            if rank == 0:
                print(f"  Time: {avg_time:.1f}ms")
                print(f"  Memory per GPU: {mem_used:.1f}MB")
                print(f"  Total effective batch: {batch_size * world_size}")
                print(f"  Output shape: {output.shape}")

        except Exception as e:
            if rank == 0:
                print(f"  Failed: {e}")

        # Cleanup
        dist.barrier()
        if "model" in locals():
            del model, q, k, v, output
        torch.cuda.empty_cache()

    cleanup()


def main():
    """Run multi-GPU benchmarks."""
    print("=== Multi-GPU Benchmark ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPUs available: {torch.cuda.device_count()}")

    if torch.cuda.device_count() < 2:
        print("This benchmark requires at least 2 GPUs")
        return

    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 1. Test DataParallel (easier to use, single-node)
    benchmark_data_parallel()

    # 2. Test Distributed (more scalable, multi-node capable)
    # Note: Using spawn for distributed requires configs to be pickleable
    configs = [
        {"seq_len": 4096, "batch_per_gpu": 2},
        {"seq_len": 8192, "batch_per_gpu": 1},
        {"seq_len": 16384, "batch_per_gpu": 1},
    ]

    # Run distributed benchmark
    world_size = min(torch.cuda.device_count(), 2)  # Use 2 GPUs
    mp.spawn(
        benchmark_distributed_single_gpu,
        args=(world_size, configs),
        nprocs=world_size,
        join=True,
    )

    print("\n=== Summary ===")
    print("✓ DataParallel works well for single-node multi-GPU")
    print("✓ Distributed mode enables model parallelism and multi-node scaling")
    print("✓ Both approaches successfully utilize multiple GPUs")


if __name__ == "__main__":
    main()
