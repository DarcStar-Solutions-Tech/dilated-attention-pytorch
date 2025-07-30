#!/usr/bin/env python3
"""
Comprehensive benchmark comparing Enhanced vs Refactored Enhanced kernels.
Tests various configurations and provides detailed performance analysis.
"""

import torch
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def benchmark_model(model, x, warmup=10, runs=50):
    """Benchmark a model with proper warmup and timing."""

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Time runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            _ = model(x)
        end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        times.append(elapsed)

    times = np.array(times)
    return {
        "mean": times.mean(),
        "std": times.std(),
        "min": times.min(),
        "max": times.max(),
        "median": np.median(times),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
    }


def create_models(
    hidden_dim, num_heads, segment_size, dilation_rate, optimization_level
):
    """Create Enhanced and Refactored models with same configuration."""

    # Original Enhanced
    enhanced = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            enable_multi_row=(optimization_level != OptimizationLevel.NONE),
            enable_8k_optimization=(optimization_level == OptimizationLevel.AGGRESSIVE),
            enable_4k_optimization=(optimization_level == OptimizationLevel.AGGRESSIVE),
            enable_sparse_optimization=(optimization_level != OptimizationLevel.NONE),
        )
        .cuda()
        .eval()
    )

    # Refactored Enhanced
    refactored = (
        UnifiedHilbertAttentionOptimizedEnhancedRefactored(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            optimization_level=optimization_level,
        )
        .cuda()
        .eval()
    )

    # Sync weights
    refactored.qkv_proj.weight.data = enhanced.qkv_proj.weight.data.clone()
    refactored.out_proj.weight.data = enhanced.out_proj.weight.data.clone()

    return enhanced, refactored


def verify_correctness(enhanced, refactored, x):
    """Verify outputs match between implementations."""

    with torch.no_grad():
        out_enhanced = enhanced(x)
        out_refactored = refactored(x)

    diff = (out_enhanced - out_refactored).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "correct": max_diff < 1e-3,
    }


def benchmark_configuration(config):
    """Benchmark a specific configuration."""

    seq_len, dilation_rate, opt_level, batch_size = config

    # Fixed parameters
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    # Create models
    enhanced, refactored = create_models(
        hidden_dim, num_heads, segment_size, dilation_rate, opt_level
    )

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Verify correctness
    correctness = verify_correctness(enhanced, refactored, x)

    if not correctness["correct"]:
        print(f"WARNING: Correctness check failed for {config}")
        print(f"  Max diff: {correctness['max_diff']:.6f}")

    # Benchmark
    enhanced_stats = benchmark_model(enhanced, x)
    refactored_stats = benchmark_model(refactored, x)

    # Calculate speedup
    speedup = enhanced_stats["mean"] / refactored_stats["mean"]

    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = enhanced(x)
    enhanced_memory = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = refactored(x)
    refactored_memory = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "config": config,
        "enhanced": enhanced_stats,
        "refactored": refactored_stats,
        "speedup": speedup,
        "enhanced_memory_mb": enhanced_memory,
        "refactored_memory_mb": refactored_memory,
        "correctness": correctness,
    }


def print_results_table(results):
    """Print results in a formatted table."""

    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS: Enhanced vs Refactored Enhanced")
    print("=" * 120)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("\n")

    # Group by optimization level
    for opt_level in [
        OptimizationLevel.NONE,
        OptimizationLevel.BASIC,
        OptimizationLevel.AGGRESSIVE,
    ]:
        level_results = [r for r in results if r["config"][2] == opt_level]
        if not level_results:
            continue

        print(f"\n{opt_level.name} Optimization:")
        print("-" * 100)
        print(
            f"{'SeqLen':>6} | {'d':>2} | {'Batch':>5} | {'Enhanced':>8} | {'Refactored':>10} | {'Speedup':>7} | {'Mem E':>6} | {'Mem R':>6} | Status"
        )
        print("-" * 100)

        for result in level_results:
            seq_len, dilation_rate, _, batch_size = result["config"]
            e_mean = result["enhanced"]["mean"]
            r_mean = result["refactored"]["mean"]
            speedup = result["speedup"]
            e_mem = result["enhanced_memory_mb"]
            r_mem = result["refactored_memory_mb"]

            # Status icon
            if speedup > 1.1:
                status = "🚀"  # Faster
            elif speedup > 0.9:
                status = "✅"  # Similar
            else:
                status = "❌"  # Slower

            print(
                f"{seq_len:>6} | {dilation_rate:>2} | {batch_size:>5} | {e_mean:>8.2f} | {r_mean:>10.2f} | {speedup:>6.2f}x | {e_mem:>6.1f} | {r_mem:>6.1f} | {status}"
            )

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    all_speedups = [r["speedup"] for r in results]
    dense_speedups = [r["speedup"] for r in results if r["config"][1] == 1]
    sparse_speedups = [r["speedup"] for r in results if r["config"][1] > 1]

    print("\nOverall Performance:")
    print(f"  Average speedup: {np.mean(all_speedups):.2f}x")
    print(f"  Median speedup: {np.median(all_speedups):.2f}x")
    print(f"  Min speedup: {np.min(all_speedups):.2f}x")
    print(f"  Max speedup: {np.max(all_speedups):.2f}x")

    if dense_speedups:
        print("\nDense Patterns (d=1):")
        print(f"  Average speedup: {np.mean(dense_speedups):.2f}x")
        print(f"  Median speedup: {np.median(dense_speedups):.2f}x")

    if sparse_speedups:
        print("\nSparse Patterns (d>1):")
        print(f"  Average speedup: {np.mean(sparse_speedups):.2f}x")
        print(f"  Median speedup: {np.median(sparse_speedups):.2f}x")

    # Memory comparison
    total_e_mem = sum(r["enhanced_memory_mb"] for r in results)
    total_r_mem = sum(r["refactored_memory_mb"] for r in results)

    print("\nMemory Usage:")
    print(f"  Total Enhanced: {total_e_mem:.1f} MB")
    print(f"  Total Refactored: {total_r_mem:.1f} MB")
    print(f"  Memory reduction: {(1 - total_r_mem / total_e_mem) * 100:.1f}%")


def create_performance_plots(results):
    """Create performance visualization plots."""

    # Prepare data
    configs = []
    speedups = []

    for r in results:
        seq_len, dilation_rate, opt_level, batch_size = r["config"]
        config_str = f"{seq_len}\nd={dilation_rate}\n{opt_level.name[:3]}"
        configs.append(config_str)
        speedups.append(r["speedup"])

    # Create figure
    plt.figure(figsize=(15, 8))

    # Bar plot
    plt.subplot(2, 1, 1)
    colors = ["green" if s > 1.1 else "orange" if s > 0.9 else "red" for s in speedups]
    bars = plt.bar(range(len(configs)), speedups, color=colors)
    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    plt.xticks(range(len(configs)), configs, rotation=45, ha="right")
    plt.ylabel("Speedup (Enhanced / Refactored)")
    plt.title("Performance Comparison: Enhanced vs Refactored Enhanced")
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
        )

    # Time comparison
    plt.subplot(2, 1, 2)
    enhanced_times = [r["enhanced"]["mean"] for r in results]
    refactored_times = [r["refactored"]["mean"] for r in results]

    x = np.arange(len(configs))
    width = 0.35

    plt.bar(x - width / 2, enhanced_times, width, label="Enhanced", alpha=0.8)
    plt.bar(x + width / 2, refactored_times, width, label="Refactored", alpha=0.8)

    plt.xticks(x, configs, rotation=45, ha="right")
    plt.ylabel("Time (ms)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"benchmark_enhanced_vs_refactored_{timestamp}.png", dpi=150)
    print(f"\nPlot saved as: benchmark_enhanced_vs_refactored_{timestamp}.png")


def main():
    """Run comprehensive benchmark suite."""

    # Test configurations: (seq_len, dilation_rate, optimization_level, batch_size)
    configs = [
        # Dense patterns - varying optimization levels
        (1024, 1, OptimizationLevel.NONE, 4),
        (1024, 1, OptimizationLevel.BASIC, 4),
        (2048, 1, OptimizationLevel.BASIC, 4),
        (4096, 1, OptimizationLevel.BASIC, 2),
        (4096, 1, OptimizationLevel.AGGRESSIVE, 2),  # 4K optimization
        (8192, 1, OptimizationLevel.BASIC, 2),
        (8192, 1, OptimizationLevel.AGGRESSIVE, 2),  # 8K optimization
        (16384, 1, OptimizationLevel.BASIC, 1),
        # Sparse patterns - d=2
        (2048, 2, OptimizationLevel.BASIC, 4),
        (4096, 2, OptimizationLevel.BASIC, 2),
        (4096, 2, OptimizationLevel.AGGRESSIVE, 2),  # 4K d=2 special
        (8192, 2, OptimizationLevel.BASIC, 2),
        (16384, 2, OptimizationLevel.BASIC, 1),
        # Very sparse patterns - d=4
        (2048, 4, OptimizationLevel.BASIC, 4),
        (4096, 4, OptimizationLevel.BASIC, 2),
        (4096, 4, OptimizationLevel.AGGRESSIVE, 2),  # 4K d=4 special
        (8192, 4, OptimizationLevel.BASIC, 2),
        (16384, 4, OptimizationLevel.BASIC, 1),
        # Extreme sparse - d=8
        (8192, 8, OptimizationLevel.BASIC, 2),
        (16384, 8, OptimizationLevel.BASIC, 1),
    ]

    results = []

    print("Running benchmarks...")
    for i, config in enumerate(configs):
        print(
            f"\rProgress: {i + 1}/{len(configs)} - Testing {config}    ",
            end="",
            flush=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

        try:
            result = benchmark_configuration(config)
            results.append(result)
        except Exception as e:
            print(f"\nERROR with config {config}: {str(e)}")

    print("\n")

    # Print results
    print_results_table(results)

    # Create plots
    create_performance_plots(results)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"benchmark_results_{timestamp}.txt", "w") as f:
        f.write("=" * 120 + "\n")
        f.write("DETAILED BENCHMARK RESULTS\n")
        f.write("=" * 120 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name()}\n")
        f.write(f"PyTorch: {torch.__version__}\n\n")

        for result in results:
            f.write(f"\nConfiguration: {result['config']}\n")
            f.write(
                f"  Enhanced mean: {result['enhanced']['mean']:.2f}ms (std: {result['enhanced']['std']:.2f})\n"
            )
            f.write(
                f"  Refactored mean: {result['refactored']['mean']:.2f}ms (std: {result['refactored']['std']:.2f})\n"
            )
            f.write(f"  Speedup: {result['speedup']:.2f}x\n")
            f.write(
                f"  Memory - Enhanced: {result['enhanced_memory_mb']:.1f}MB, Refactored: {result['refactored_memory_mb']:.1f}MB\n"
            )
            f.write(f"  Correctness: {result['correctness']}\n")

    print(f"\nDetailed results saved to: benchmark_results_{timestamp}.txt")


if __name__ == "__main__":
    main()
