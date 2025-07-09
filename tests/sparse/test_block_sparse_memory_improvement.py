#!/usr/bin/env python3
"""Test memory improvements in block sparse attention with UnifiedMemoryPool."""

import gc
import time

import torch

from dilated_attention_pytorch import BlockSparseRingMultiheadDilatedAttention
from dilated_attention_pytorch import (
    SparsePatternConfig,
)
from dilated_attention_pytorch.core.memory_pool import (
    get_global_memory_pool,
    reset_global_memory_pool,
)


def measure_memory_usage(use_global_pool=True):
    """Measure memory usage with and without global pool."""
    # Reset everything
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Reset or disable global pool
    if use_global_pool:
        reset_global_memory_pool()
        pool = get_global_memory_pool()
    else:
        # Disable by clearing after each use
        reset_global_memory_pool()

    # Create block sparse multihead attention
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,  # 90% sparse
        block_size=128,
    )

    attention = BlockSparseRingMultiheadDilatedAttention(
        embed_dim=768,
        num_heads=12,
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        batch_first=True,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)

    # Test parameters
    batch_size = 2
    seq_len = 1024

    # Track timing and memory
    times = []
    memory_used = []

    # Run multiple iterations
    num_iterations = 10
    for i in range(num_iterations):
        # Create new input each time
        x = torch.randn(batch_size, seq_len, 768, device=device)

        # Measure time
        start_time = time.time()

        # Forward pass
        output = attention(x, x, x)
        if isinstance(output, tuple):
            output = output[0]

        # Sync for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        times.append(elapsed)

        # Measure memory
        if device.type == "cuda":
            memory_used.append(torch.cuda.memory_allocated() / 1024**2)  # MB

        # Clear output to free memory
        del output

        # Disable pool after each iteration if not using it
        if not use_global_pool:
            reset_global_memory_pool()

    # Get final stats
    if use_global_pool:
        pool_stats = pool.get_stats()
    else:
        pool_stats = None

    # Clean up
    del attention
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "times": times,
        "memory_used": memory_used,
        "pool_stats": pool_stats,
        "avg_time": sum(times[2:]) / (len(times) - 2),  # Skip warmup
        "peak_memory": max(memory_used) if memory_used else 0,
    }


def main():
    """Compare block sparse performance with and without memory pool."""
    print("Testing Block Sparse Attention Memory Pool Performance")
    print("=" * 60)

    # Test without pool (baseline)
    print("\n1. WITHOUT UnifiedMemoryPool (baseline):")
    results_without = measure_memory_usage(use_global_pool=False)
    print(f"   Average time: {results_without['avg_time'] * 1000:.2f} ms")
    print(f"   Peak memory: {results_without['peak_memory']:.1f} MB")

    # Test with pool
    print("\n2. WITH UnifiedMemoryPool:")
    results_with = measure_memory_usage(use_global_pool=True)
    print(f"   Average time: {results_with['avg_time'] * 1000:.2f} ms")
    print(f"   Peak memory: {results_with['peak_memory']:.1f} MB")

    if results_with["pool_stats"]:
        stats = results_with["pool_stats"]
        print(f"   Total buffers cached: {stats['total_buffers']}")
        print(f"   Hot cache size: {stats['hot_cache_size']}")
        print(f"   Memory in pool: {stats['total_allocated_bytes'] / 1024**2:.1f} MB")

    # Calculate improvements
    print("\n3. Improvements:")
    time_improvement = (
        (results_without["avg_time"] - results_with["avg_time"])
        / results_without["avg_time"]
        * 100
    )
    memory_improvement = (
        (results_without["peak_memory"] - results_with["peak_memory"])
        / results_without["peak_memory"]
        * 100
    )

    print(f"   Speed improvement: {time_improvement:+.1f}%")
    print(f"   Memory improvement: {memory_improvement:+.1f}%")

    # Note about block sparse
    print("\n4. Analysis:")
    print("   Block sparse attention benefits from UnifiedMemoryPool because:")
    print("   - It inherits from RingDilatedAttention which uses the global pool")
    print("   - Pattern caching reduces computation overhead")
    print("   - Buffer reuse helps with temporary allocations")
    print("   - The sparse nature means fewer but more frequent allocations")


if __name__ == "__main__":
    main()
