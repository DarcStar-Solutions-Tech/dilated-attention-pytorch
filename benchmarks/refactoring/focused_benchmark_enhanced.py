#!/usr/bin/env python3
"""
Focused benchmark comparing Enhanced vs Refactored on key configurations.
"""

import torch
import sys
import gc
import numpy as np

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def benchmark_precise(model, x, warmup=20, runs=100):
    """High-precision benchmark with more runs."""

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
        "median": np.median(times),
        "min": times.min(),
        "max": times.max(),
    }


def test_configuration(seq_len, dilation_rate, opt_level_name, batch_size=2):
    """Test a specific configuration."""

    opt_level = getattr(OptimizationLevel, opt_level_name)
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

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

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Verify correctness
    with torch.no_grad():
        out_e = enhanced(x)
        out_r = refactored(x)

    max_diff = (out_e - out_r).abs().max().item()

    # Benchmark
    e_stats = benchmark_precise(enhanced, x)
    r_stats = benchmark_precise(refactored, x)

    speedup = e_stats["mean"] / r_stats["mean"]

    return {
        "enhanced": e_stats,
        "refactored": r_stats,
        "speedup": speedup,
        "max_diff": max_diff,
    }


def main():
    print("=" * 80)
    print("FOCUSED BENCHMARK: Enhanced vs Refactored Enhanced")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Key configurations to test
    configs = [
        # Dense patterns showing different behaviors
        (1024, 1, "BASIC", "Small dense"),
        (4096, 1, "BASIC", "4K dense basic"),
        (4096, 1, "AGGRESSIVE", "4K dense aggressive"),
        (8192, 1, "BASIC", "8K dense basic"),
        (8192, 1, "AGGRESSIVE", "8K dense aggressive"),
        # Sparse patterns with special optimizations
        (2048, 2, "BASIC", "2K sparse d=2"),
        (4096, 2, "BASIC", "4K sparse d=2 basic"),
        (4096, 2, "AGGRESSIVE", "4K sparse d=2 aggressive"),
        (4096, 4, "BASIC", "4K sparse d=4 basic"),
        (4096, 4, "AGGRESSIVE", "4K sparse d=4 aggressive"),
        (8192, 2, "BASIC", "8K sparse d=2"),
        (8192, 4, "BASIC", "8K sparse d=4"),
    ]

    print("Configuration Performance Analysis:")
    print("-" * 80)
    print(
        f"{'Config':<25} | {'Enhanced':>10} | {'Refactored':>10} | {'Speedup':>8} | {'Verdict':<10}"
    )
    print("-" * 80)

    results = []

    for seq_len, dilation_rate, opt_level, desc in configs:
        gc.collect()
        torch.cuda.empty_cache()

        result = test_configuration(seq_len, dilation_rate, opt_level)
        results.append((desc, result))

        e_mean = result["enhanced"]["mean"]
        r_mean = result["refactored"]["mean"]
        speedup = result["speedup"]

        if speedup > 1.1:
            verdict = "✅ Faster"
        elif speedup > 0.9:
            verdict = "🔄 Similar"
        else:
            verdict = "❌ Slower"

        print(
            f"{desc:<25} | {e_mean:>9.2f}ms | {r_mean:>9.2f}ms | {speedup:>7.2f}x | {verdict}"
        )

        if result["max_diff"] > 1e-3:
            print(f"  ⚠️  WARNING: Max diff = {result['max_diff']:.6f}")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Group by pattern type
    dense_results = [(d, r) for d, r in results if "dense" in d]
    sparse_results = [(d, r) for d, r in results if "sparse" in d]

    print("\nDense Pattern Analysis:")
    dense_speedups = [r["speedup"] for _, r in dense_results]
    print(f"  Average speedup: {np.mean(dense_speedups):.2f}x")
    print(
        f"  Best case: {max(dense_speedups):.2f}x ({dense_results[np.argmax(dense_speedups)][0]})"
    )
    print(
        f"  Worst case: {min(dense_speedups):.2f}x ({dense_results[np.argmin(dense_speedups)][0]})"
    )

    print("\nSparse Pattern Analysis:")
    sparse_speedups = [r["speedup"] for _, r in sparse_results]
    print(f"  Average speedup: {np.mean(sparse_speedups):.2f}x")
    print(
        f"  Best case: {max(sparse_speedups):.2f}x ({sparse_results[np.argmax(sparse_speedups)][0]})"
    )
    print(
        f"  Worst case: {min(sparse_speedups):.2f}x ({sparse_results[np.argmin(sparse_speedups)][0]})"
    )

    # Optimization level analysis
    print("\nOptimization Level Analysis:")
    basic_speedups = [r["speedup"] for d, r in results if "basic" in d.lower()]
    aggressive_speedups = [
        r["speedup"] for d, r in results if "aggressive" in d.lower()
    ]

    print(f"  BASIC average: {np.mean(basic_speedups):.2f}x")
    print(f"  AGGRESSIVE average: {np.mean(aggressive_speedups):.2f}x")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    improvements = [(d, r) for d, r in results if r["speedup"] > 1.1]
    regressions = [(d, r) for d, r in results if r["speedup"] < 0.9]

    if improvements:
        print("\n✅ Significant Improvements:")
        for desc, result in improvements:
            print(f"  - {desc}: {result['speedup']:.2f}x faster")

    if regressions:
        print("\n❌ Performance Regressions:")
        for desc, result in regressions:
            print(
                f"  - {desc}: {result['speedup']:.2f}x ({(1 - result['speedup']) * 100:.1f}% slower)"
            )

    # Overall verdict
    all_speedups = [r["speedup"] for _, r in results]
    avg_speedup = np.mean(all_speedups)

    print("\n" + "=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)
    print(f"Average speedup across all configurations: {avg_speedup:.2f}x")

    if avg_speedup > 1.05:
        print("✅ Refactored version is FASTER overall")
    elif avg_speedup > 0.95:
        print("🔄 Refactored version has SIMILAR performance")
    else:
        print("❌ Refactored version is SLOWER overall")

    print(f"\nConfigurations improved: {len(improvements)}/{len(results)}")
    print(f"Configurations regressed: {len(regressions)}/{len(results)}")
    print(
        f"Configurations similar: {len(results) - len(improvements) - len(regressions)}/{len(results)}"
    )


if __name__ == "__main__":
    main()
