"""
Focused benchmark to measure pattern generation overhead vs cache lookup.
"""

import time
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from dilated_attention_pytorch.core import DilatedPatternCache


def benchmark_pattern_generation(
    segment_lengths: List[int],
    dilation_rates: List[int],
    device: str = "cuda",
    iterations: int = 10000,
) -> Dict[str, float]:
    """Benchmark raw pattern generation time."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Generate patterns (what happens without cache)
        patterns = []
        for s, r in zip(segment_lengths, dilation_rates):
            for offset in range(r):
                indices = torch.arange(offset, s, r, device=device)
                patterns.append(indices)

        if device == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return {
        "mean": np.mean(times) * 1000,  # Convert to ms
        "std": np.std(times) * 1000,
        "min": np.min(times) * 1000,
        "max": np.max(times) * 1000,
    }


def benchmark_cache_operations(
    segment_lengths: List[int],
    dilation_rates: List[int],
    device: str = "cuda",
    iterations: int = 10000,
) -> Dict[str, Dict[str, float]]:
    """Benchmark cache put and get operations."""
    cache = DilatedPatternCache()

    # Prepare pattern
    seq_len = max(segment_lengths)
    test_pattern = torch.arange(0, segment_lengths[0], dilation_rates[0], device=device)

    # Benchmark cache put
    put_times = []
    for i in range(iterations):
        key = f"test_pattern_{i}"
        start = time.perf_counter()
        cache.put(key, test_pattern, move_to_cpu=True)
        if device == "cuda":
            torch.cuda.synchronize()
        put_times.append(time.perf_counter() - start)

    # Clear cache for get benchmark
    cache.clear()

    # Pre-populate cache for get benchmark
    cache.put_dilated_indices(
        test_pattern, seq_len, tuple(segment_lengths), tuple(dilation_rates)
    )

    # Benchmark cache get
    get_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = cache.get_dilated_indices(
            seq_len,
            tuple(segment_lengths),
            tuple(dilation_rates),
            device=torch.device(device),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        get_times.append(time.perf_counter() - start)

    return {
        "put": {
            "mean": np.mean(put_times) * 1000,
            "std": np.std(put_times) * 1000,
        },
        "get": {
            "mean": np.mean(get_times) * 1000,
            "std": np.std(get_times) * 1000,
        },
    }


def compare_overhead():
    """Compare pattern generation overhead with cache overhead."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running overhead comparison on {device}")
    print("=" * 60)

    configurations = [
        ("Small", [128, 256], [1, 2]),
        ("Medium", [256, 512, 1024], [1, 2, 4]),
        ("Large", [512, 1024, 2048], [1, 2, 4]),
        ("XLarge", [1024, 2048, 4096], [1, 2, 4]),
    ]

    results = []

    for name, segment_lengths, dilation_rates in configurations:
        print(f"\n{name} Configuration:")
        print(f"  Segments: {segment_lengths}")
        print(f"  Dilations: {dilation_rates}")

        # Benchmark pattern generation
        gen_stats = benchmark_pattern_generation(
            segment_lengths, dilation_rates, device
        )
        print(
            f"  Pattern generation: {gen_stats['mean']:.4f} ± {gen_stats['std']:.4f} ms"
        )

        # Benchmark cache operations
        cache_stats = benchmark_cache_operations(
            segment_lengths, dilation_rates, device
        )
        print(
            f"  Cache put: {cache_stats['put']['mean']:.4f} ± {cache_stats['put']['std']:.4f} ms"
        )
        print(
            f"  Cache get: {cache_stats['get']['mean']:.4f} ± {cache_stats['get']['std']:.4f} ms"
        )

        # Calculate speedup
        speedup = gen_stats["mean"] / cache_stats["get"]["mean"]
        print(f"  Speedup (generation/get): {speedup:.2f}x")

        results.append(
            {
                "config": name,
                "gen_mean": gen_stats["mean"],
                "gen_std": gen_stats["std"],
                "cache_get_mean": cache_stats["get"]["mean"],
                "cache_get_std": cache_stats["get"]["std"],
                "speedup": speedup,
            }
        )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    configs = [r["config"] for r in results]
    gen_times = [r["gen_mean"] for r in results]
    cache_times = [r["cache_get_mean"] for r in results]
    speedups = [r["speedup"] for r in results]

    # Time comparison
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, gen_times, width, label="Pattern Generation", alpha=0.8
    )
    bars2 = ax1.bar(x + width / 2, cache_times, width, label="Cache Lookup", alpha=0.8)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Pattern Generation vs Cache Lookup Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.set_yscale("log")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Speedup
    bars3 = ax2.bar(configs, speedups, alpha=0.8, color="green")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_title("Cache Speedup (Pattern Generation / Cache Lookup)")
    ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Break-even")

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}x",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        "benchmarks/pattern_generation_overhead.png", dpi=300, bbox_inches="tight"
    )
    print("\nVisualization saved to: benchmarks/pattern_generation_overhead.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_speedup = np.mean(speedups)
    print(f"Average speedup from caching: {avg_speedup:.2f}x")
    print(f"Pattern generation is {avg_speedup:.1f}x slower than cache lookup")


def measure_cache_memory_overhead():
    """Measure memory overhead of the cache itself."""
    import sys

    cache = DilatedPatternCache(max_size=100)

    # Measure empty cache size
    empty_size = sys.getsizeof(cache._cache) + sys.getsizeof(cache)

    # Fill cache with patterns
    for i in range(100):
        pattern = torch.arange(1000)  # 1000 int64 elements = 8KB
        cache.put(f"pattern_{i}", pattern)

    # Measure full cache size
    full_size = sys.getsizeof(cache._cache) + sys.getsizeof(cache)

    # Estimate pattern storage
    pattern_size = 100 * 1000 * 8  # 100 patterns * 1000 elements * 8 bytes

    print("\nCache Memory Overhead:")
    print(f"  Empty cache overhead: {empty_size / 1024:.2f} KB")
    print(f"  Full cache overhead: {full_size / 1024:.2f} KB")
    print(f"  Pattern data size: {pattern_size / 1024:.2f} KB")
    print(f"  Total memory usage: {(full_size + pattern_size) / 1024:.2f} KB")


if __name__ == "__main__":
    compare_overhead()
    measure_cache_memory_overhead()
