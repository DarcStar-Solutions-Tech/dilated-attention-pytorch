"""
Benchmark pattern caching performance impact.

This script measures the performance improvement from pattern caching
in dilated attention modules.
"""

import time
import torch
import torch.nn as nn
from typing import Dict

from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention
from dilated_attention_pytorch.core import clear_global_cache, get_global_pattern_cache


def benchmark_attention(
    attention_module: nn.Module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark attention module with and without cache."""
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(warmup_iterations):
        _ = attention_module(q, k, v)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark with cold cache
    clear_global_cache()
    cache = get_global_pattern_cache()
    cache.reset_stats()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = attention_module(q, k, v)

    if device == "cuda":
        torch.cuda.synchronize()

    cold_cache_time = time.perf_counter() - start_time
    cold_cache_stats = cache.get_stats()

    # Benchmark with warm cache (patterns already cached)
    cache.reset_stats()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = attention_module(q, k, v)

    if device == "cuda":
        torch.cuda.synchronize()

    warm_cache_time = time.perf_counter() - start_time
    warm_cache_stats = cache.get_stats()

    return {
        "cold_cache_time": cold_cache_time,
        "warm_cache_time": warm_cache_time,
        "speedup": cold_cache_time / warm_cache_time,
        "cold_cache_misses": cold_cache_stats["misses"],
        "warm_cache_hits": warm_cache_stats["hits"],
        "cache_hit_rate": warm_cache_stats["hit_rate"],
        "patterns_cached": warm_cache_stats["size"],
    }


def main():
    """Run pattern caching benchmarks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    print("=" * 60)

    # Test configurations
    configs = [
        # (name, segment_lengths, dilation_rates, seq_len, batch_size, num_heads)
        ("Small", [256, 512], [1, 2], 512, 4, 8),
        ("Medium", [512, 1024, 2048], [1, 2, 4], 2048, 2, 16),
        ("Large", [1024, 2048, 4096], [1, 2, 4], 4096, 1, 32),
    ]

    attention_classes = [
        ("DilatedAttention", DilatedAttention),
        ("ImprovedDilatedAttention", ImprovedDilatedAttention),
    ]

    for (
        config_name,
        segment_lengths,
        dilation_rates,
        seq_len,
        batch_size,
        num_heads,
    ) in configs:
        print(f"\n{config_name} Configuration:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of heads: {num_heads}")
        print(f"  Segment lengths: {segment_lengths}")
        print(f"  Dilation rates: {dilation_rates}")
        print()

        for attn_name, AttnClass in attention_classes:
            # Create attention module
            attention = AttnClass(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            ).to(device)
            attention.eval()

            # Run benchmark
            results = benchmark_attention(
                attention,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=64,
                num_iterations=50,
                device=device,
            )

            print(f"  {attn_name}:")
            print(f"    Cold cache time: {results['cold_cache_time']:.4f}s")
            print(f"    Warm cache time: {results['warm_cache_time']:.4f}s")
            print(f"    Speedup: {results['speedup']:.2f}x")
            print(f"    Cache hit rate: {results['cache_hit_rate']:.2%}")
            print(f"    Patterns cached: {results['patterns_cached']}")
            print()

    # Memory usage analysis
    print("\nMemory Usage Analysis:")
    cache = get_global_pattern_cache()
    stats = cache.get_stats()

    print(f"Total patterns cached: {stats['size']}")
    print(f"Total cache accesses: {stats['total_accesses']}")
    print(f"Overall hit rate: {stats['hit_rate']:.2%}")
    print(f"Total evictions: {stats['evictions']}")

    # Estimate memory saved
    # Each pattern typically saves creating a tensor of indices
    avg_pattern_size = 1000  # conservative estimate in elements
    bytes_per_element = 8  # int64
    memory_saved_mb = (stats["hits"] * avg_pattern_size * bytes_per_element) / (
        1024 * 1024
    )
    print(f"Estimated memory operations saved: {memory_saved_mb:.2f} MB")


if __name__ == "__main__":
    main()
