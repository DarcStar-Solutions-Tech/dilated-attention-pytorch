"""
Test script for the cached Hilbert implementation to verify correctness and performance.
"""

import torch
import time

from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert import (
    BlockSparseRingDilatedAttentionHilbert,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_cached import (
    BlockSparseRingDilatedAttentionHilbertCached,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    SparsePatternConfig,
)


def benchmark_forward_pass(
    attention_module,
    seq_length: int,
    batch_size: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
) -> float:
    """Benchmark forward pass performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_module = attention_module.to(device)

    # Create random input tensors
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(warmup_iterations):
        _ = attention_module(q, k, v)

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time the forward passes
    start_time = time.time()
    for _ in range(num_iterations):
        _ = attention_module(q, k, v)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations

    return avg_time


def test_cached_hilbert_correctness():
    """Test that cached and non-cached versions produce the same output."""
    print("Testing correctness of cached Hilbert implementation...")

    # Configuration
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,
        block_size=64,
    )

    # Create both versions
    standard = BlockSparseRingDilatedAttentionHilbert(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_hilbert=True,
        hilbert_block_level=True,
    )

    cached = BlockSparseRingDilatedAttentionHilbertCached(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_hilbert=True,
        hilbert_block_level=True,
    )

    # Test on different sequence lengths
    test_seq_lengths = [2048, 4096, 8192]

    for seq_len in test_seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        # Create input tensors
        torch.manual_seed(42)  # For reproducibility
        q = torch.randn(2, seq_len, 8, 64)
        k = torch.randn(2, seq_len, 8, 64)
        v = torch.randn(2, seq_len, 8, 64)

        # Forward pass through both
        output_standard = standard(q, k, v)
        output_cached = cached(q, k, v)

        # Check if outputs match
        max_diff = (output_standard - output_cached).abs().max().item()
        print(f"  Max difference: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("  ✓ Outputs match!")
        else:
            print("  ✗ Outputs differ significantly!")

    # Print cache statistics
    stats = cached.get_pattern_stats()
    print("\nCache statistics:")
    print(f"  Cached orderings: {stats['hilbert_optimization']['cached_orderings']}")
    print(
        f"  Cached sequence lengths: {stats['hilbert_optimization']['cached_seq_lengths']}"
    )
    print(
        f"  Memory usage: {stats['hilbert_optimization']['memory_usage_bytes']} bytes"
    )


def benchmark_performance():
    """Benchmark performance improvement of cached implementation."""
    print("\n\nBenchmarking performance improvement...")

    # Configuration
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,
        block_size=64,
    )

    # Test different sequence lengths
    test_configs = [
        (4096, "4K tokens"),
        (8192, "8K tokens"),
        (16384, "16K tokens"),
    ]

    results = []

    for seq_len, desc in test_configs:
        print(f"\n{desc} (sequence length: {seq_len}):")

        # Create both versions
        standard = BlockSparseRingDilatedAttentionHilbert(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
            use_hilbert=True,
            hilbert_block_level=True,
        )

        cached = BlockSparseRingDilatedAttentionHilbertCached(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
            use_hilbert=True,
            hilbert_block_level=True,
            common_seq_lengths=[seq_len],  # Pre-compute for this length
        )

        # Benchmark
        time_standard = benchmark_forward_pass(standard, seq_len)
        time_cached = benchmark_forward_pass(cached, seq_len)

        speedup = time_standard / time_cached

        print(f"  Standard Hilbert: {time_standard * 1000:.2f} ms")
        print(f"  Cached Hilbert:   {time_cached * 1000:.2f} ms")
        print(f"  Speedup:          {speedup:.2f}x")

        results.append(
            {
                "seq_len": seq_len,
                "desc": desc,
                "time_standard": time_standard,
                "time_cached": time_cached,
                "speedup": speedup,
            }
        )

    # Summary
    print("\n\nSummary:")
    print("-" * 60)
    print(f"{'Sequence':<15} {'Standard (ms)':<15} {'Cached (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['desc']:<15} {r['time_standard'] * 1000:<15.2f} {r['time_cached'] * 1000:<15.2f} {r['speedup']:<10.2f}x"
        )

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 60)
    print(f"Average speedup: {avg_speedup:.2f}x")


def test_memory_efficiency():
    """Test memory efficiency of pre-computed orderings."""
    print("\n\nTesting memory efficiency...")

    # Configuration
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,
        block_size=64,
    )

    # Common sequence lengths in practice
    common_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # Create cached version with pre-computation
    cached = BlockSparseRingDilatedAttentionHilbertCached(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_hilbert=True,
        hilbert_block_level=True,
        common_seq_lengths=common_lengths,
    )

    # Get statistics
    stats = cached.get_pattern_stats()
    hilbert_stats = stats["hilbert_optimization"]

    print(f"Pre-computed {hilbert_stats['cached_orderings']} unique orderings")
    print(f"Covering {hilbert_stats['cached_seq_lengths']} sequence lengths")
    print(f"Total memory usage: {hilbert_stats['memory_usage_bytes'] / 1024:.2f} KB")
    print(
        f"Average per ordering: {hilbert_stats['memory_usage_bytes'] / max(1, hilbert_stats['cached_orderings']) / 1024:.2f} KB"
    )

    # Test cache clearing
    print("\nClearing cache...")
    cached.clear_cache()
    stats_after = cached.get_pattern_stats()
    print(
        f"After clearing: {stats_after['hilbert_optimization']['cached_orderings']} orderings"
    )


if __name__ == "__main__":
    # Run all tests
    test_cached_hilbert_correctness()
    benchmark_performance()
    test_memory_efficiency()

    print("\n\nAll tests completed!")
