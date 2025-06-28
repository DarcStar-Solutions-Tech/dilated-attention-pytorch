#!/usr/bin/env python3
"""
Multi-GPU extreme sequence benchmark for Ring and Block Sparse attention.
Pushes the limits using both available GPUs.
"""

import gc
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from dilated_attention_pytorch import (
    RingDilatedAttentionV2,
    BlockSparseRingDilatedAttention,
)


@dataclass
class BenchmarkResult:
    seq_len: int
    implementation: str
    num_gpus: int
    success: bool
    time_ms: Optional[float] = None
    memory_per_gpu_gb: Optional[List[float]] = None
    total_memory_gb: Optional[float] = None
    memory_per_token_mb: Optional[float] = None
    tokens_per_second: Optional[float] = None
    error: Optional[str] = None
    sparsity_ratio: Optional[float] = None


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_memory_stats(device_id: int) -> Tuple[float, float]:
    """Get memory stats for a specific GPU in GB."""
    torch.cuda.synchronize(device_id)
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    return allocated, reserved


def clear_memory(device_id: Optional[int] = None):
    """Clear memory on specific or all GPUs."""
    gc.collect()
    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    else:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def benchmark_ring_attention_multi_gpu(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
) -> BenchmarkResult:
    """Benchmark Ring attention using multiple GPUs."""

    num_gpus = torch.cuda.device_count()

    try:
        # Initialize distributed if needed
        if not dist.is_initialized() and num_gpus > 1:
            mp.spawn(setup_distributed, args=(num_gpus,), nprocs=num_gpus, join=False)

        clear_memory()

        # Create attention on GPU 0 first
        device = torch.device("cuda:0")

        attention = RingDilatedAttentionV2(
            segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
            dilation_rates=[1, 2, 4],
            ring_size=num_gpus,  # Use all GPUs in ring
            enable_memory_pool=True,
            lightweight_pool=False,
            device=device,
            dtype=torch.float16,
        )

        # Create inputs
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # Warmup
        _ = attention(q, k, v)
        torch.cuda.synchronize()

        # Get initial memory on all GPUs
        initial_memory = []
        for i in range(num_gpus):
            allocated, _ = get_memory_stats(i)
            initial_memory.append(allocated)

        # Benchmark
        start = time.perf_counter()
        output = attention(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()

        time_ms = (end - start) * 1000

        # Get final memory usage on all GPUs
        final_memory = []
        for i in range(num_gpus):
            allocated, _ = get_memory_stats(i)
            final_memory.append(allocated)

        # Calculate metrics
        memory_per_gpu = [
            final - initial for final, initial in zip(final_memory, initial_memory)
        ]
        total_memory = sum(memory_per_gpu)
        memory_per_token_mb = (total_memory * 1024) / seq_len
        tokens_per_second = seq_len / (time_ms / 1000)

        # Cleanup
        del attention, q, k, v, output
        clear_memory()

        return BenchmarkResult(
            seq_len=seq_len,
            implementation="RingDilatedAttentionV2",
            num_gpus=num_gpus,
            success=True,
            time_ms=time_ms,
            memory_per_gpu_gb=memory_per_gpu,
            total_memory_gb=total_memory,
            memory_per_token_mb=memory_per_token_mb,
            tokens_per_second=tokens_per_second,
        )

    except Exception as e:
        clear_memory()
        return BenchmarkResult(
            seq_len=seq_len,
            implementation="RingDilatedAttentionV2",
            num_gpus=num_gpus,
            success=False,
            error=str(e),
        )


def benchmark_block_sparse_multi_gpu(
    seq_len: int,
    sparsity_ratio: float,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
) -> BenchmarkResult:
    """Benchmark Block Sparse attention, potentially using model parallelism."""

    try:
        clear_memory()

        # For now, use single GPU but we could extend to model parallel
        device = torch.device("cuda:0")

        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
            dilation_rates=[1, 2, 4],
            sparsity_ratio=sparsity_ratio,
            enable_memory_pool=True,
            lightweight_pool=False,
        )

        # Create inputs
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # Warmup
        _ = attention(q, k, v)
        torch.cuda.synchronize()

        # Get initial memory
        initial_allocated, _ = get_memory_stats(0)

        # Benchmark
        start = time.perf_counter()
        output = attention(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()

        time_ms = (end - start) * 1000

        # Get final memory
        final_allocated, _ = get_memory_stats(0)
        memory_used = final_allocated - initial_allocated

        # Calculate metrics
        memory_per_token_mb = (memory_used * 1024) / seq_len
        tokens_per_second = seq_len / (time_ms / 1000)

        # Cleanup
        del attention, q, k, v, output
        clear_memory()

        return BenchmarkResult(
            seq_len=seq_len,
            implementation="BlockSparseRingDilatedAttention",
            num_gpus=1,
            success=True,
            time_ms=time_ms,
            memory_per_gpu_gb=[memory_used],
            total_memory_gb=memory_used,
            memory_per_token_mb=memory_per_token_mb,
            tokens_per_second=tokens_per_second,
            sparsity_ratio=sparsity_ratio,
        )

    except Exception as e:
        clear_memory()
        return BenchmarkResult(
            seq_len=seq_len,
            implementation="BlockSparseRingDilatedAttention",
            num_gpus=1,
            success=False,
            error=str(e),
            sparsity_ratio=sparsity_ratio,
        )


def find_max_sequence_length(
    implementation: str,
    use_multi_gpu: bool = True,
    sparsity_ratio: Optional[float] = None,
) -> Tuple[int, List[BenchmarkResult]]:
    """Binary search to find maximum supported sequence length."""

    results = []

    # Start with reasonable bounds
    min_len = 8192
    max_len = 10_000_000  # 10M tokens theoretical max

    # First, find upper bound that fails
    test_len = 32768
    while test_len <= max_len:
        print(f"  Testing {test_len:,} tokens...")

        if implementation == "ring":
            result = benchmark_ring_attention_multi_gpu(test_len)
        else:  # block_sparse
            result = benchmark_block_sparse_multi_gpu(test_len, sparsity_ratio or 0.99)

        results.append(result)

        if not result.success:
            max_len = test_len
            break

        # Exponential increase
        test_len *= 2

    # Binary search for exact limit
    while max_len - min_len > 8192:
        mid_len = (min_len + max_len) // 2
        mid_len = (mid_len // 8192) * 8192  # Round to nearest 8K

        print(f"  Testing {mid_len:,} tokens...")

        if implementation == "ring":
            result = benchmark_ring_attention_multi_gpu(mid_len)
        else:
            result = benchmark_block_sparse_multi_gpu(mid_len, sparsity_ratio or 0.99)

        results.append(result)

        if result.success:
            min_len = mid_len
        else:
            max_len = mid_len

    return min_len, results


def run_multi_gpu_benchmark():
    """Run comprehensive multi-GPU benchmark."""

    print("=" * 100)
    print("MULTI-GPU EXTREME SEQUENCE BENCHMARK")
    print("=" * 100)
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(
            f"  GPU {i}: {torch.cuda.get_device_name(i)} "
            f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)"
        )
    print()

    # Test configurations
    test_configs = [
        # (seq_len, description)
        (32768, "32K tokens"),
        (65536, "64K tokens"),
        (131072, "128K tokens"),
        (262144, "256K tokens"),
        (524288, "512K tokens"),
        (1048576, "1M tokens"),
        (2097152, "2M tokens"),
    ]

    # Ring Attention Tests
    print("\nRING ATTENTION (Multi-GPU)")
    print("-" * 100)

    ring_results = []
    for seq_len, description in test_configs:
        print(f"\n{description}:")
        result = benchmark_ring_attention_multi_gpu(seq_len)
        ring_results.append(result)

        if result.success:
            print("  ✓ Success!")
            print(f"  Time: {result.time_ms:.1f}ms")
            print(
                f"  Memory per GPU: {[f'{m:.2f}GB' for m in result.memory_per_gpu_gb]}"
            )
            print(f"  Total memory: {result.total_memory_gb:.2f}GB")
            print(f"  Memory per token: {result.memory_per_token_mb:.3f}MB")
            print(f"  Throughput: {result.tokens_per_second:.0f} tokens/sec")
        else:
            print(f"  ✗ Failed: {result.error}")
            break

    # Block Sparse Tests
    print("\n\nBLOCK SPARSE RING ATTENTION")
    print("-" * 100)

    sparse_configs = [
        # (seq_len, sparsity, description)
        (131072, 0.95, "128K tokens, 95% sparse"),
        (262144, 0.98, "256K tokens, 98% sparse"),
        (524288, 0.99, "512K tokens, 99% sparse"),
        (1048576, 0.995, "1M tokens, 99.5% sparse"),
        (2097152, 0.998, "2M tokens, 99.8% sparse"),
        (4194304, 0.999, "4M tokens, 99.9% sparse"),
    ]

    sparse_results = []
    for seq_len, sparsity, description in sparse_configs:
        print(f"\n{description}:")
        result = benchmark_block_sparse_multi_gpu(seq_len, sparsity)
        sparse_results.append(result)

        if result.success:
            print("  ✓ Success!")
            print(f"  Time: {result.time_ms:.1f}ms")
            print(f"  Memory: {result.total_memory_gb:.2f}GB")
            print(f"  Memory per token: {result.memory_per_token_mb:.3f}MB")
            print(f"  Throughput: {result.tokens_per_second:.0f} tokens/sec")
            effective_compute = seq_len * seq_len * (1 - sparsity) / 1e9
            print(f"  Effective compute: {effective_compute:.1f}B operations")
        else:
            print(f"  ✗ Failed: {result.error}")

    # Find maximum sequence lengths
    print("\n\nMAXIMUM SEQUENCE LENGTH SEARCH")
    print("-" * 100)

    print("\nSearching for Ring Attention maximum...")
    max_ring, _ = find_max_sequence_length("ring", use_multi_gpu=True)
    print(f"Maximum Ring Attention sequence: {max_ring:,} tokens")

    print("\nSearching for Block Sparse maximum (99.9% sparse)...")
    max_sparse, _ = find_max_sequence_length(
        "block_sparse", use_multi_gpu=False, sparsity_ratio=0.999
    )
    print(f"Maximum Block Sparse sequence: {max_sparse:,} tokens")

    # Analysis
    print("\n\nANALYSIS")
    print("=" * 100)

    # Find best performing configurations
    successful_ring = [r for r in ring_results if r.success]
    successful_sparse = [r for r in sparse_results if r.success]

    if successful_ring:
        print("\nRing Attention Performance:")
        print(f"  Maximum sequence: {max(r.seq_len for r in successful_ring):,} tokens")
        best_throughput = max(successful_ring, key=lambda r: r.tokens_per_second)
        print(
            f"  Best throughput: {best_throughput.tokens_per_second:.0f} tokens/sec "
            f"at {best_throughput.seq_len:,} tokens"
        )

        # Memory efficiency
        avg_memory_per_token = np.mean([r.memory_per_token_mb for r in successful_ring])
        print(f"  Average memory per token: {avg_memory_per_token:.3f}MB")

    if successful_sparse:
        print("\nBlock Sparse Attention Performance:")
        print(
            f"  Maximum sequence: {max(r.seq_len for r in successful_sparse):,} tokens"
        )
        best_sparse = max(successful_sparse, key=lambda r: r.seq_len)
        print(f"  Achieved with {best_sparse.sparsity_ratio * 100:.1f}% sparsity")

        # Memory efficiency
        avg_memory_per_token = np.mean(
            [r.memory_per_token_mb for r in successful_sparse]
        )
        print(f"  Average memory per token: {avg_memory_per_token:.3f}MB")

    # Recommendations
    print("\n\nRECOMMENDATIONS")
    print("-" * 100)
    print(f"\n1. For sequences up to {max_ring:,} tokens:")
    print("   - Use Ring Attention with multi-GPU")
    print("   - Provides best memory distribution")
    print(f"\n2. For sequences up to {max_sparse:,} tokens:")
    print("   - Use Block Sparse Ring Attention")
    print("   - Adjust sparsity based on sequence length")
    print("\n3. For maximum efficiency:")
    print("   - Combine both: Block Sparse + Multi-GPU Ring")
    print("   - Could theoretically reach 10M+ tokens")

    print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    # Set environment for better multi-GPU performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1"  # GTX 1080 architecture

    run_multi_gpu_benchmark()
