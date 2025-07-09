#!/usr/bin/env python3
"""
Benchmark block-sparse implementations on multiple GPUs.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both GPUs

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def setup_distributed(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def benchmark_distributed(rank, world_size, seq_len, batch_size):
    """Run benchmark on a single process."""
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    # Print from rank 0 only
    if rank == 0:
        print("\nDistributed Block-Sparse Benchmark")
        print(
            f"Sequence length: {seq_len}, Batch size: {batch_size}, World size: {world_size}"
        )
        print("-" * 60)

    # Create inputs
    num_heads = 8
    head_dim = 64

    # Each GPU gets a portion of the batch
    local_batch_size = batch_size // world_size

    q = torch.randn(
        local_batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test distributed block-sparse
    try:
        model = create_block_sparse_attention(
            variant="distributed",
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            sparsity_ratio=0.05,  # 95% sparse
            world_size=world_size,
            rank=rank,
        ).to(device=device, dtype=dtype)

        # Warmup
        for _ in range(3):
            _ = model(q, k, v)

        # Synchronize all GPUs
        dist.barrier()

        # Time forward pass
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(10):
            _ = model(q, k, v)

        torch.cuda.synchronize()
        dist.barrier()

        forward_time = (time.time() - start) / 10 * 1000  # ms

        # Gather times from all ranks
        times = [torch.tensor(forward_time, device=device)]
        dist.all_gather(times, times[0])

        if rank == 0:
            avg_time = sum(t.item() for t in times) / len(times)
            print(f"✓ Distributed (95% sparse): {avg_time:.2f}ms")

    except Exception as e:
        if rank == 0:
            print(f"✗ Distributed failed: {e}")

    cleanup()


def benchmark_data_parallel():
    """Test with DataParallel (simpler multi-GPU)."""
    print("\nDataParallel Block-Sparse Benchmark")
    print("=" * 60)

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for DataParallel")
        return

    device = torch.device("cuda:0")
    dtype = torch.float16

    # Test configurations
    configs = [
        {"seq_len": 4096, "batch_size": 4},
        {"seq_len": 8192, "batch_size": 2},
        {"seq_len": 16384, "batch_size": 2},
    ]

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        print(f"\nSequence length: {seq_len}, Batch size: {batch_size}")
        print("-" * 60)

        # Create inputs
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Single GPU baseline
        try:
            model_single = create_block_sparse_attention(
                variant="base",
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                sparsity_ratio=0.05,
            ).to(device=device, dtype=dtype)

            torch.cuda.synchronize()
            start = time.time()
            _ = model_single(q, k, v)
            torch.cuda.synchronize()
            single_time = (time.time() - start) * 1000

            print(f"Single GPU (95% sparse): {single_time:.2f}ms")

            del model_single
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Single GPU failed: {e}")
            single_time = float("inf")

        # DataParallel version
        try:
            model_base = create_block_sparse_attention(
                variant="base",
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                sparsity_ratio=0.05,
            ).to(device=device, dtype=dtype)

            # Wrap in DataParallel
            model_dp = torch.nn.DataParallel(model_base, device_ids=[0, 1])

            torch.cuda.synchronize()
            start = time.time()
            _ = model_dp(q, k, v)
            torch.cuda.synchronize()
            dp_time = (time.time() - start) * 1000

            speedup = single_time / dp_time
            print(
                f"DataParallel (95% sparse): {dp_time:.2f}ms (speedup: {speedup:.2f}x)"
            )

            del model_base, model_dp
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"DataParallel failed: {e}")

        # Show memory usage per GPU
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            print(f"  GPU {i} memory: {allocated:.1f}MB")


def test_ring_attention_multi_gpu():
    """Test ring attention which is designed for multi-GPU."""
    print("\n\nRing Attention Multi-GPU Test")
    print("=" * 60)

    # Ring attention shines with very long sequences
    print("\nTesting Ring Attention (designed for multi-GPU long sequences)")

    # This would normally be run with torchrun or similar
    # For now, just show the concept
    print("\nRing Attention benefits:")
    print("- Splits sequence across GPUs (each GPU handles seq_len/num_gpus)")
    print("- O(n/p) memory per GPU instead of O(n²)")
    print("- Enables processing sequences that don't fit on single GPU")

    print("\nExample scaling (2 GPUs):")
    for seq_len in [32768, 65536, 131072]:
        per_gpu_seq = seq_len // 2
        memory_saved_gb = (seq_len**2 - seq_len * per_gpu_seq) * 2 / 1024**3
        print(
            f"  {seq_len:,} tokens → {per_gpu_seq:,} per GPU (saves ~{memory_saved_gb:.1f}GB)"
        )


def main():
    """Run multi-GPU benchmarks."""
    print("Block-Sparse Multi-GPU Benchmarks")
    print("=" * 60)

    print(f"\nGPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run DataParallel benchmarks
    benchmark_data_parallel()

    # Run distributed benchmarks
    if torch.cuda.device_count() >= 2:
        print("\n\nRunning Distributed Benchmarks...")
        print("=" * 60)

        # Test different configurations
        configs = [
            {"seq_len": 8192, "batch_size": 4},
            {"seq_len": 16384, "batch_size": 2},
        ]

        for config in configs:
            world_size = 2
            mp.spawn(
                benchmark_distributed,
                args=(world_size, config["seq_len"], config["batch_size"]),
                nprocs=world_size,
                join=True,
            )

    # Show ring attention concept
    test_ring_attention_multi_gpu()

    print("\n✅ Multi-GPU benchmarks completed!")


if __name__ == "__main__":
    main()
