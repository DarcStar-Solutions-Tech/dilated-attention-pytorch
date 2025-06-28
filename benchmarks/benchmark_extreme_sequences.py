#!/usr/bin/env python3
"""
Benchmark for extreme sequence length processing with memory pools.
Tests the limits of what's possible with and without memory optimization.
"""

import gc
import time
import torch
import psutil
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dilated_attention_pytorch.core import create_dilated_attention
from dilated_attention_pytorch import (
    RingDilatedAttentionV2,
    BlockSparseRingDilatedAttention,
)


@dataclass
class SequenceTestResult:
    seq_len: int
    batch_size: int
    success: bool
    time_ms: Optional[float]
    memory_gb: Optional[float]
    error: Optional[str]
    implementation: str
    pool_enabled: bool


def get_memory_info():
    """Get current memory usage in GB."""
    if torch.cuda.is_available():
        # GPU memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    else:
        # CPU memory
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3, 0


def clear_memory():
    """Aggressively clear all memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_sequence_length(
    impl_type: str,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    enable_pool: Optional[bool] = None,
    use_ring_size: Optional[int] = None,
) -> SequenceTestResult:
    """Test a specific sequence length configuration."""
    clear_memory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Create attention module
        if impl_type == "ring_manual":
            # Manual ring attention for extreme sequences
            attention = RingDilatedAttentionV2(
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
                ring_size=use_ring_size,
                enable_memory_pool=enable_pool if enable_pool is not None else True,
                lightweight_pool=False,  # Full pool for extreme sequences
            )
            pool_enabled = enable_pool if enable_pool is not None else True
        elif impl_type == "block_sparse":
            # Block sparse for extreme sequences
            attention = BlockSparseRingDilatedAttention(
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
                sparsity_ratio=0.95,  # 95% sparse for extreme sequences
                enable_memory_pool=enable_pool if enable_pool is not None else True,
            )
            pool_enabled = enable_pool if enable_pool is not None else True
        else:
            # Use factory
            kwargs = {}
            if enable_pool is not None:
                kwargs["enable_memory_pool"] = enable_pool

            attention = create_dilated_attention(
                impl_type,
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
                **kwargs,
            )
            pool_enabled = getattr(attention, "enable_memory_pool", None)

        # Create inputs with reduced precision for extreme sequences
        dtype = torch.float16 if seq_len > 32768 else torch.float32
        shape = (batch_size, seq_len, num_heads, head_dim)

        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)

        # Warmup
        _ = attention(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time the forward pass
        start = time.perf_counter()
        output = attention(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        time_ms = (end - start) * 1000

        # Get memory usage
        mem_allocated, mem_reserved = get_memory_info()

        # Cleanup
        del q, k, v, output, attention
        clear_memory()

        return SequenceTestResult(
            seq_len=seq_len,
            batch_size=batch_size,
            success=True,
            time_ms=time_ms,
            memory_gb=mem_allocated,
            error=None,
            implementation=impl_type,
            pool_enabled=pool_enabled,
        )

    except Exception as e:
        return SequenceTestResult(
            seq_len=seq_len,
            batch_size=batch_size,
            success=False,
            time_ms=None,
            memory_gb=None,
            error=str(e),
            implementation=impl_type,
            pool_enabled=enable_pool,
        )


def find_max_sequence_length(
    impl_type: str, enable_pool: bool
) -> Tuple[int, List[SequenceTestResult]]:
    """Binary search to find maximum supported sequence length."""
    results = []

    # Start with reasonable bounds
    min_len = 1024
    max_len = 1_000_000  # 1M tokens

    # First, find upper bound that fails
    test_len = 8192
    while test_len <= max_len:
        result = test_sequence_length(impl_type, test_len, enable_pool=enable_pool)
        results.append(result)

        if not result.success:
            max_len = test_len
            break

        test_len *= 2

    # Binary search for exact limit
    while max_len - min_len > 1024:
        mid_len = (min_len + max_len) // 2
        mid_len = (mid_len // 1024) * 1024  # Round to nearest 1024

        result = test_sequence_length(impl_type, mid_len, enable_pool=enable_pool)
        results.append(result)

        if result.success:
            min_len = mid_len
        else:
            max_len = mid_len

    return min_len, results


def benchmark_extreme_sequences():
    """Run comprehensive extreme sequence benchmarks."""
    print("=" * 80)
    print("EXTREME SEQUENCE LENGTH BENCHMARK")
    print("=" * 80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print()

    # Test configurations for extreme sequences
    extreme_configs = [
        # (seq_len, batch_size, description)
        (16384, 1, "16K tokens"),
        (32768, 1, "32K tokens"),
        (65536, 1, "64K tokens"),
        (131072, 1, "128K tokens"),
        (262144, 1, "256K tokens"),
        (524288, 1, "512K tokens"),
        (1048576, 1, "1M tokens"),
    ]

    implementations = [
        ("improved", "Improved Attention"),
        ("ring_manual", "Ring Attention"),
        ("block_sparse", "Block Sparse (95% sparse)"),
    ]

    results_by_impl = {}

    for impl_type, impl_name in implementations:
        print(f"\n{impl_name}")
        print("-" * 60)

        results = []

        for seq_len, batch_size, description in extreme_configs:
            print(f"\nTesting {description}...")

            # Test with memory pool
            result_with_pool = test_sequence_length(
                impl_type,
                seq_len,
                batch_size,
                enable_pool=True,
                use_ring_size=8 if "ring" in impl_type else None,
            )
            results.append(("With Pool", result_with_pool))

            # Test without memory pool (only if with pool succeeded and seq < 128K)
            if result_with_pool.success and seq_len <= 131072:
                result_no_pool = test_sequence_length(
                    impl_type, seq_len, batch_size, enable_pool=False
                )
                results.append(("No Pool", result_no_pool))

                # Calculate speedup
                if result_no_pool.success:
                    speedup = result_no_pool.time_ms / result_with_pool.time_ms
                    mem_saving = (
                        (result_no_pool.memory_gb - result_with_pool.memory_gb)
                        / result_no_pool.memory_gb
                        * 100
                    )
                    print(
                        f"  Pool Impact: {speedup:.2f}x speedup, {mem_saving:.1f}% memory reduction"
                    )

            # Print results
            for config, result in results[-2:]:
                if result.success:
                    print(
                        f"  {config}: ✓ {result.time_ms:.1f}ms, {result.memory_gb:.2f}GB"
                    )
                else:
                    print(f"  {config}: ✗ {result.error}")

        results_by_impl[impl_name] = results

    # Find maximum sequence lengths
    print("\n" + "=" * 80)
    print("MAXIMUM SEQUENCE LENGTH ANALYSIS")
    print("=" * 80)

    for impl_type, impl_name in implementations:
        print(f"\n{impl_name}:")

        # With memory pool
        max_with_pool, _ = find_max_sequence_length(impl_type, enable_pool=True)
        print(f"  Max with pool: {max_with_pool:,} tokens")

        # Without memory pool (only for standard implementations)
        if impl_type == "improved":
            max_no_pool, _ = find_max_sequence_length(impl_type, enable_pool=False)
            print(f"  Max without pool: {max_no_pool:,} tokens")
            print(f"  Pool enables {max_with_pool / max_no_pool:.1f}x longer sequences")

    # Extreme sequence recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR EXTREME SEQUENCES")
    print("=" * 80)

    print("\n1. For 16K-64K tokens:")
    print("   - ImprovedDilatedAttention with memory pools")
    print("   - Provides best balance of speed and memory")

    print("\n2. For 64K-256K tokens:")
    print("   - Ring Attention becomes necessary")
    print("   - O(n) memory complexity enables these lengths")

    print("\n3. For 256K-1M tokens:")
    print("   - Block Sparse Ring Attention")
    print("   - 95% sparsity dramatically reduces compute")

    print("\n4. For >1M tokens:")
    print("   - Distributed Block Sparse Ring Attention")
    print("   - Requires multi-GPU setup")

    print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    benchmark_extreme_sequences()
