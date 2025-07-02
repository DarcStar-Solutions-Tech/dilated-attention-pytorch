#!/usr/bin/env python3
"""
Benchmark script to measure Ring Attention V2Collective performance improvements.

This script compares:
1. Old vs new causal mask handling
2. Performance with and without pattern caching
3. Memory usage comparisons
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import the Ring Attention implementations
try:
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )
    from dilated_attention_pytorch.improved_dilated_attention import (
        ImprovedDilatedAttention,
    )
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def measure_performance(func, *args, num_warmup=3, num_runs=10, **kwargs):
    """Measure performance with warmup runs."""
    # Warmup
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        _ = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "times": times,
    }


def benchmark_causal_mask_caching():
    """Benchmark causal mask caching improvements."""
    print("=" * 70)
    print("Benchmark 1: Causal Mask Caching Performance")
    print("=" * 70)

    seq_lengths = [512, 1024, 2048, 4096, 8192]
    results = []

    for seq_len in seq_lengths:
        print(f"\nTesting seq_len={seq_len}...")

        # Determine appropriate segment lengths
        if seq_len <= 1024:
            segment_lengths = [256, 512]
        elif seq_len <= 2048:
            segment_lengths = [512, 1024]
        else:
            segment_lengths = [1024, 2048]

        # Create attention instance
        ring_attn = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=[1, 2],
            dropout=0.0,
            use_pattern_cache=True,
        )

        # Create inputs
        batch_size = 2
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Measure first run (no cache)
        print("  Measuring first run (no cache)...")
        first_run = measure_performance(ring_attn, q, k, v, is_causal=True, num_runs=1)

        # Measure subsequent runs (with cache)
        print("  Measuring subsequent runs (with cache)...")
        cached_runs = measure_performance(ring_attn, q, k, v, is_causal=True)

        speedup = first_run["mean"] / cached_runs["mean"]

        results.append(
            {
                "seq_len": seq_len,
                "first_run": first_run["mean"],
                "cached_mean": cached_runs["mean"],
                "cached_std": cached_runs["std"],
                "speedup": speedup,
            }
        )

        print(f"  First run: {first_run['mean']:.4f}s")
        print(f"  Cached runs: {cached_runs['mean']:.4f}s ± {cached_runs['std']:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def benchmark_pattern_caching():
    """Benchmark pattern caching on/off."""
    print("\n" + "=" * 70)
    print("Benchmark 2: Pattern Caching Performance")
    print("=" * 70)

    seq_len = 4096
    segment_lengths = [512, 1024, 2048]
    dilation_rates = [1, 2, 4]

    # Create inputs
    batch_size = 2
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Test with pattern cache enabled
    print("\nWith pattern caching enabled:")
    ring_with_cache = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        use_pattern_cache=True,
    )

    perf_with_cache = measure_performance(ring_with_cache, q, k, v, is_causal=True)
    print(f"  Time: {perf_with_cache['mean']:.4f}s ± {perf_with_cache['std']:.4f}s")

    # Test with pattern cache disabled
    print("\nWith pattern caching disabled:")
    ring_no_cache = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        use_pattern_cache=False,
    )

    perf_no_cache = measure_performance(ring_no_cache, q, k, v, is_causal=True)
    print(f"  Time: {perf_no_cache['mean']:.4f}s ± {perf_no_cache['std']:.4f}s")

    speedup = perf_no_cache["mean"] / perf_with_cache["mean"]
    print(f"\nSpeedup from pattern caching: {speedup:.2f}x")

    return {
        "with_cache": perf_with_cache,
        "without_cache": perf_no_cache,
        "speedup": speedup,
    }


def benchmark_vs_baseline():
    """Benchmark Ring Attention V2 vs baseline implementation."""
    print("\n" + "=" * 70)
    print("Benchmark 3: Ring Attention V2 vs Baseline")
    print("=" * 70)

    seq_lengths = [1024, 2048, 4096]
    results = []

    for seq_len in seq_lengths:
        print(f"\nTesting seq_len={seq_len}...")

        # Determine segment lengths
        segment_lengths = [min(512, seq_len), min(1024, seq_len)]
        dilation_rates = [1, 2]

        # Create inputs
        batch_size = 2
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Ring Attention V2
        ring_attn = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            use_pattern_cache=True,
        )

        ring_perf = measure_performance(ring_attn, q, k, v, is_causal=True)

        # Baseline (ImprovedDilatedAttention)
        baseline_attn = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
        )

        baseline_perf = measure_performance(baseline_attn, q, k, v, is_causal=True)

        speedup = baseline_perf["mean"] / ring_perf["mean"]

        results.append(
            {
                "seq_len": seq_len,
                "ring_time": ring_perf["mean"],
                "baseline_time": baseline_perf["mean"],
                "speedup": speedup,
            }
        )

        print(f"  Ring V2: {ring_perf['mean']:.4f}s ± {ring_perf['std']:.4f}s")
        print(f"  Baseline: {baseline_perf['mean']:.4f}s ± {baseline_perf['std']:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def benchmark_memory_usage():
    """Benchmark memory usage improvements."""
    print("\n" + "=" * 70)
    print("Benchmark 4: Memory Usage")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return None

    seq_lengths = [1024, 2048, 4096, 8192]
    results = []

    for seq_len in seq_lengths:
        print(f"\nTesting seq_len={seq_len}...")

        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Determine segment lengths
        if seq_len <= 2048:
            segment_lengths = [512, 1024]
        else:
            segment_lengths = [1024, 2048]

        # Create inputs on GPU
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")

        # Measure Ring V2 memory
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()

        ring_attn = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=[1, 2],
            dropout=0.0,
            use_pattern_cache=True,
        ).cuda()

        _ = ring_attn(q, k, v, is_causal=True)

        torch.cuda.synchronize()
        ring_mem = torch.cuda.memory_allocated() - start_mem

        # Clean up
        del ring_attn
        torch.cuda.empty_cache()

        # Measure baseline memory
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()

        baseline_attn = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=[1, 2],
            dropout=0.0,
        ).cuda()

        _ = baseline_attn(q, k, v, is_causal=True)

        torch.cuda.synchronize()
        baseline_mem = torch.cuda.memory_allocated() - start_mem

        # Calculate metrics
        ring_mem_mb = ring_mem / (1024**2)
        baseline_mem_mb = baseline_mem / (1024**2)
        mem_ratio = ring_mem / baseline_mem if baseline_mem > 0 else 0

        results.append(
            {
                "seq_len": seq_len,
                "ring_mem_mb": ring_mem_mb,
                "baseline_mem_mb": baseline_mem_mb,
                "mem_ratio": mem_ratio,
            }
        )

        print(f"  Ring V2: {ring_mem_mb:.2f} MB")
        print(f"  Baseline: {baseline_mem_mb:.2f} MB")
        print(f"  Memory ratio: {mem_ratio:.2f}x")

        # Clean up
        del q, k, v, baseline_attn
        torch.cuda.empty_cache()

    return results


def plot_results(causal_results, vs_baseline_results):
    """Plot benchmark results."""
    # Plot causal mask caching speedup
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    seq_lens = [r["seq_len"] for r in causal_results]
    speedups = [r["speedup"] for r in causal_results]

    plt.bar(range(len(seq_lens)), speedups, color="skyblue")
    plt.xticks(range(len(seq_lens)), seq_lens)
    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup")
    plt.title("Causal Mask Caching Speedup")
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.5, f"{v:.1f}x", ha="center")

    # Plot Ring V2 vs Baseline speedup
    plt.subplot(1, 2, 2)
    seq_lens = [r["seq_len"] for r in vs_baseline_results]
    speedups = [r["speedup"] for r in vs_baseline_results]

    plt.bar(range(len(seq_lens)), speedups, color="lightgreen")
    plt.xticks(range(len(seq_lens)), seq_lens)
    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup")
    plt.title("Ring V2 vs Baseline Speedup")
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.02, f"{v:.2f}x", ha="center")

    plt.tight_layout()
    plt.savefig("ring_v2_benchmark_results.png", dpi=150)
    print("\nResults saved to ring_v2_benchmark_results.png")


def main():
    """Run all benchmarks."""
    print("Ring Attention V2Collective Performance Benchmarks")
    print("=" * 70)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    print(f"Running on device: {device}")

    # Run benchmarks
    results = {}

    # Benchmark 1: Causal mask caching
    results["causal_caching"] = benchmark_causal_mask_caching()

    # Benchmark 2: Pattern caching
    results["pattern_caching"] = benchmark_pattern_caching()

    # Benchmark 3: Ring V2 vs baseline
    results["vs_baseline"] = benchmark_vs_baseline()

    # Benchmark 4: Memory usage
    if torch.cuda.is_available():
        results["memory"] = benchmark_memory_usage()

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Causal mask caching
    avg_speedup = np.mean([r["speedup"] for r in results["causal_caching"]])
    print("\nCausal Mask Caching:")
    print(f"  Average speedup: {avg_speedup:.1f}x")

    # Pattern caching
    print("\nPattern Caching:")
    print(f"  Speedup: {results['pattern_caching']['speedup']:.2f}x")

    # vs Baseline
    avg_speedup = np.mean([r["speedup"] for r in results["vs_baseline"]])
    print("\nRing V2 vs Baseline:")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    # Memory usage
    if "memory" in results and results["memory"]:
        avg_ratio = np.mean([r["mem_ratio"] for r in results["memory"]])
        print("\nMemory Usage:")
        print(f"  Average memory ratio: {avg_ratio:.2f}x")

    # Plot results
    try:
        plot_results(results["causal_caching"], results["vs_baseline"])
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

    print("\n✅ All benchmarks completed successfully!")

    return results


if __name__ == "__main__":
    results = main()
