#!/usr/bin/env python3
"""
Comprehensive performance comparison between original Enhanced and refactored Enhanced.
Tests various configurations to ensure refactoring maintains performance.
"""

import torch
import sys
import gc
from typing import Dict
import numpy as np

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def benchmark_model(model, x, warmup=5, runs=20):
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
    }


def verify_correctness(unified, original, refactored, x):
    """Verify all implementations produce similar outputs."""

    with torch.no_grad():
        out_unified = unified(x)
        out_original = original(x)
        out_refactored = refactored(x)

    # Check original vs unified
    diff_orig = (out_unified - out_original).abs()
    max_diff_orig = diff_orig.max().item()

    # Check refactored vs unified
    diff_refact = (out_unified - out_refactored).abs()
    max_diff_refact = diff_refact.max().item()

    # Check refactored vs original
    diff_direct = (out_original - out_refactored).abs()
    max_diff_direct = diff_direct.max().item()

    return {
        "orig_vs_unified": max_diff_orig,
        "refact_vs_unified": max_diff_refact,
        "refact_vs_orig": max_diff_direct,
        "all_close": max_diff_orig < 0.01
        and max_diff_refact < 0.01
        and max_diff_direct < 0.001,
    }


def benchmark_configuration(
    seq_len: int, dilation_rate: int, optimization_level: OptimizationLevel
):
    """Benchmark a specific configuration."""

    # Parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    # Create models
    unified = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    original = (
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

    # Synchronize weights
    original.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
    original.out_proj.weight.data = unified.out_proj.weight.data.clone()
    refactored.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
    refactored.out_proj.weight.data = unified.out_proj.weight.data.clone()

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Verify correctness
    correctness = verify_correctness(unified, original, refactored, x)

    # Benchmark
    unified_stats = benchmark_model(unified, x)
    original_stats = benchmark_model(original, x)
    refactored_stats = benchmark_model(refactored, x)

    return {
        "unified": unified_stats,
        "original": original_stats,
        "refactored": refactored_stats,
        "correctness": correctness,
    }


def print_results(results: Dict, seq_len: int, dilation_rate: int, opt_level: str):
    """Print benchmark results in a nice format."""

    unified_time = results["unified"]["mean"]
    original_time = results["original"]["mean"]
    refactored_time = results["refactored"]["mean"]

    # Calculate speedups
    orig_vs_unified = unified_time / original_time
    refact_vs_unified = unified_time / refactored_time
    refact_vs_orig = original_time / refactored_time

    # Correctness
    correct = "✅" if results["correctness"]["all_close"] else "❌"

    print(
        f"{seq_len:<6} | {dilation_rate:<2} | {opt_level:<10} | "
        f"{unified_time:>7.2f} | {original_time:>7.2f} | {refactored_time:>7.2f} | "
        f"{orig_vs_unified:>6.2f}x | {refact_vs_unified:>6.2f}x | {refact_vs_orig:>6.2f}x | {correct}"
    )


def main():
    """Run comprehensive benchmark suite."""

    print("=== Enhanced Kernel Refactoring Performance Comparison ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test configurations
    configs = [
        # Dense patterns
        (1024, 1, OptimizationLevel.BASIC),
        (2048, 1, OptimizationLevel.BASIC),
        (4096, 1, OptimizationLevel.BASIC),
        (4096, 1, OptimizationLevel.AGGRESSIVE),  # Test 4K optimization
        (8192, 1, OptimizationLevel.BASIC),
        (8192, 1, OptimizationLevel.AGGRESSIVE),  # Test 8K optimization
        # Sparse patterns
        (2048, 2, OptimizationLevel.BASIC),
        (4096, 2, OptimizationLevel.BASIC),
        (4096, 2, OptimizationLevel.AGGRESSIVE),  # Test 4K d=2 special case
        (4096, 4, OptimizationLevel.BASIC),
        (4096, 4, OptimizationLevel.AGGRESSIVE),  # Test 4K d=4 special case
        (8192, 2, OptimizationLevel.BASIC),
        (8192, 4, OptimizationLevel.BASIC),
        (16384, 2, OptimizationLevel.BASIC),
        (16384, 4, OptimizationLevel.BASIC),
    ]

    print("Performance Results (times in ms):")
    print()
    print(
        "SeqLen | d  | OptLevel   | Unified | Original | Refactored | Orig/Uni | Ref/Uni | Ref/Orig | ✓"
    )
    print("-" * 105)

    all_results = []

    for seq_len, dilation_rate, opt_level in configs:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            results = benchmark_configuration(seq_len, dilation_rate, opt_level)
            all_results.append((seq_len, dilation_rate, opt_level, results))
            print_results(results, seq_len, dilation_rate, opt_level.name)
        except Exception as e:
            print(
                f"{seq_len:<6} | {dilation_rate:<2} | {opt_level.name:<10} | ERROR: {str(e)}"
            )

    # Summary statistics
    print("\n=== Summary ===")

    # Calculate average performance ratios
    refact_vs_orig_ratios = []
    for _, _, _, results in all_results:
        if results["correctness"]["all_close"]:
            ratio = results["original"]["mean"] / results["refactored"]["mean"]
            refact_vs_orig_ratios.append(ratio)

    if refact_vs_orig_ratios:
        avg_ratio = np.mean(refact_vs_orig_ratios)
        print(f"\nAverage performance (Refactored vs Original): {avg_ratio:.2f}x")

        if avg_ratio > 0.95:
            print("✅ Refactoring maintains performance (within 5%)")
        elif avg_ratio > 0.9:
            print("⚠️  Minor performance regression (5-10%)")
        else:
            print("❌ Significant performance regression (>10%)")

    # Group by pattern type
    print("\n=== Performance by Pattern Type ===")

    dense_ratios = []
    sparse_ratios = []

    for seq_len, dilation_rate, opt_level, results in all_results:
        if results["correctness"]["all_close"]:
            ratio = results["original"]["mean"] / results["refactored"]["mean"]
            if dilation_rate == 1:
                dense_ratios.append(ratio)
            else:
                sparse_ratios.append(ratio)

    if dense_ratios:
        print(f"Dense patterns: {np.mean(dense_ratios):.2f}x average")
    if sparse_ratios:
        print(f"Sparse patterns: {np.mean(sparse_ratios):.2f}x average")

    # Check specific optimizations
    print("\n=== Special Case Optimizations ===")

    for seq_len, dilation_rate, opt_level, results in all_results:
        if opt_level == OptimizationLevel.AGGRESSIVE:
            if (seq_len == 4096 and dilation_rate in [2, 4]) or seq_len == 8192:
                ratio = results["original"]["mean"] / results["refactored"]["mean"]
                desc = f"{seq_len} d={dilation_rate}"
                status = "✅" if ratio > 0.95 else "❌"
                print(f"{desc}: {ratio:.2f}x {status}")

    # Memory usage comparison
    print("\n=== Memory Usage ===")

    # Test memory with larger batch
    test_seq_len = 8192
    test_batch_size = 8

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Original memory
    original_mem = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
    ).cuda()

    x_large = torch.randn(test_batch_size, test_seq_len, 512, device="cuda")
    with torch.no_grad():
        _ = original_mem(x_large)
    torch.cuda.synchronize()
    original_peak = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()

    # Refactored memory
    refactored_mem = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
    ).cuda()

    with torch.no_grad():
        _ = refactored_mem(x_large)
    torch.cuda.synchronize()
    refactored_peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Original peak memory: {original_peak:.1f} MB")
    print(f"Refactored peak memory: {refactored_peak:.1f} MB")
    print(f"Difference: {refactored_peak - original_peak:+.1f} MB")


if __name__ == "__main__":
    main()
