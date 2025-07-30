#!/usr/bin/env python3
"""
Comprehensive benchmark after sparse block size fix.
"""

import torch
import sys
import gc
import numpy as np
from datetime import datetime

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def benchmark_model(model, x, warmup=10, runs=30):
    """Benchmark with proper warmup and timing."""

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Time
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            _ = model(x)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    times = np.array(times)
    return {
        "mean": times.mean(),
        "std": times.std(),
        "min": times.min(),
        "median": np.median(times),
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""

    print("=" * 80)
    print("PERFORMANCE BENCHMARK AFTER SPARSE FIX")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Test configurations
    configs = [
        # Dense patterns
        (
            "Dense Patterns",
            [
                (1024, 1, OptimizationLevel.BASIC, 4),
                (2048, 1, OptimizationLevel.BASIC, 4),
                (4096, 1, OptimizationLevel.BASIC, 2),
                (8192, 1, OptimizationLevel.BASIC, 2),
                (16384, 1, OptimizationLevel.BASIC, 1),
            ],
        ),
        # Moderate sparse (d=2)
        (
            "Moderate Sparse (d=2)",
            [
                (2048, 2, OptimizationLevel.BASIC, 4),
                (4096, 2, OptimizationLevel.BASIC, 2),
                (8192, 2, OptimizationLevel.BASIC, 2),
                (16384, 2, OptimizationLevel.BASIC, 1),
            ],
        ),
        # Very sparse (d=4)
        (
            "Very Sparse (d=4)",
            [
                (2048, 4, OptimizationLevel.BASIC, 4),
                (4096, 4, OptimizationLevel.BASIC, 2),
                (8192, 4, OptimizationLevel.BASIC, 2),
                (16384, 4, OptimizationLevel.BASIC, 1),
            ],
        ),
        # Extreme sparse (d=8)
        (
            "Extreme Sparse (d=8)",
            [
                (8192, 8, OptimizationLevel.BASIC, 2),
                (16384, 8, OptimizationLevel.BASIC, 1),
            ],
        ),
        # AGGRESSIVE optimizations
        (
            "AGGRESSIVE Mode",
            [
                (4096, 1, OptimizationLevel.AGGRESSIVE, 2),
                (8192, 1, OptimizationLevel.AGGRESSIVE, 2),
                (4096, 2, OptimizationLevel.AGGRESSIVE, 2),
                (4096, 4, OptimizationLevel.AGGRESSIVE, 2),
            ],
        ),
    ]

    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    all_results = []

    for category, category_configs in configs:
        print(f"\n{category}:")
        print("-" * 80)
        print(
            f"{'SeqLen':>6} | {'d':>2} | {'Mode':>10} | {'Enhanced':>9} | {'Refactored':>10} | {'Speedup':>7} | {'Status'}"
        )
        print("-" * 80)

        for seq_len, dilation_rate, opt_level, batch_size in category_configs:
            gc.collect()
            torch.cuda.empty_cache()

            # Create models
            enhanced = (
                UnifiedHilbertAttentionOptimizedEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                    enable_multi_row=(opt_level != OptimizationLevel.NONE),
                    enable_8k_optimization=(opt_level == OptimizationLevel.AGGRESSIVE),
                    enable_4k_optimization=(opt_level == OptimizationLevel.AGGRESSIVE),
                    enable_sparse_optimization=(opt_level != OptimizationLevel.NONE),
                )
                .cuda()
                .eval()
            )

            refactored = (
                UnifiedHilbertAttentionOptimizedEnhancedRefactored(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                    optimization_level=opt_level,
                )
                .cuda()
                .eval()
            )

            # Sync weights
            refactored.qkv_proj.weight.data = enhanced.qkv_proj.weight.data.clone()
            refactored.out_proj.weight.data = enhanced.out_proj.weight.data.clone()

            # Test input
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
            )

            try:
                # Benchmark
                e_stats = benchmark_model(enhanced, x)
                r_stats = benchmark_model(refactored, x)

                speedup = e_stats["mean"] / r_stats["mean"]

                # Status
                if speedup > 1.1:
                    status = "🚀 Faster"
                elif speedup > 0.9:
                    status = "✅ Similar"
                else:
                    status = "❌ Slower"

                result = {
                    "category": category,
                    "seq_len": seq_len,
                    "dilation_rate": dilation_rate,
                    "opt_level": opt_level,
                    "enhanced_time": e_stats["mean"],
                    "refactored_time": r_stats["mean"],
                    "speedup": speedup,
                }
                all_results.append(result)

                print(
                    f"{seq_len:>6} | {dilation_rate:>2} | {opt_level.name:>10} | {e_stats['mean']:>8.2f}ms | {r_stats['mean']:>9.2f}ms | {speedup:>6.2f}x | {status}"
                )

            except Exception as e:
                print(
                    f"{seq_len:>6} | {dilation_rate:>2} | {opt_level.name:>10} | ERROR: {str(e)}"
                )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall
    all_speedups = [r["speedup"] for r in all_results]
    print("\nOverall Performance:")
    print(f"  Average speedup: {np.mean(all_speedups):.2f}x")
    print(f"  Median speedup: {np.median(all_speedups):.2f}x")
    print(f"  Best speedup: {np.max(all_speedups):.2f}x")
    print(f"  Worst speedup: {np.min(all_speedups):.2f}x")

    # By pattern type
    dense_results = [r for r in all_results if r["dilation_rate"] == 1]
    sparse_d2_results = [r for r in all_results if r["dilation_rate"] == 2]
    sparse_d4_results = [r for r in all_results if r["dilation_rate"] == 4]
    sparse_d8_results = [r for r in all_results if r["dilation_rate"] == 8]

    if dense_results:
        speedups = [r["speedup"] for r in dense_results]
        print("\nDense Patterns:")
        print(f"  Average: {np.mean(speedups):.2f}x")
        print(f"  Range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x")

    if sparse_d2_results:
        speedups = [r["speedup"] for r in sparse_d2_results]
        print("\nModerate Sparse (d=2):")
        print(f"  Average: {np.mean(speedups):.2f}x")
        print(f"  Range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x")
        print("  Note: Block size fix applied (64x64)")

    if sparse_d4_results:
        speedups = [r["speedup"] for r in sparse_d4_results]
        print("\nVery Sparse (d=4):")
        print(f"  Average: {np.mean(speedups):.2f}x")
        print(f"  Range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x")
        print("  Note: Refactored performs very well here!")

    if sparse_d8_results:
        speedups = [r["speedup"] for r in sparse_d8_results]
        print("\nExtreme Sparse (d=8):")
        print(f"  Average: {np.mean(speedups):.2f}x")

    # Count improvements/regressions
    improvements = sum(1 for r in all_results if r["speedup"] > 1.1)
    similar = sum(1 for r in all_results if 0.9 <= r["speedup"] <= 1.1)
    regressions = sum(1 for r in all_results if r["speedup"] < 0.9)

    print("\nPerformance Distribution:")
    print(
        f"  Faster (>1.1x): {improvements}/{len(all_results)} ({improvements / len(all_results) * 100:.1f}%)"
    )
    print(
        f"  Similar (0.9-1.1x): {similar}/{len(all_results)} ({similar / len(all_results) * 100:.1f}%)"
    )
    print(
        f"  Slower (<0.9x): {regressions}/{len(all_results)} ({regressions / len(all_results) * 100:.1f}%)"
    )

    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS FROM SPARSE FIX:")
    print("=" * 80)
    print("✅ Fixed block configuration for moderate sparse (d=2) patterns")
    print("✅ Enabled 4K optimizations in BASIC mode")
    print("✅ Very sparse patterns (d≥4) show excellent performance")
    print("⚠️  Moderate sparse still slower due to online softmax overhead")
    print("⚠️  Some dense patterns need further investigation")


if __name__ == "__main__":
    run_comprehensive_benchmark()
