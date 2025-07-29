#!/usr/bin/env python3
"""
Test integrated sparse optimizations in Enhanced implementation.
"""

import torch
import time
import sys
import gc

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_config(model, x, warmup=3, runs=10):
    """Benchmark a model with given input."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Time runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def main():
    print("=== Testing Integrated Sparse Optimizations ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test configurations - focus on problematic sparse patterns
    configs = [
        # Small sparse (should use PyTorch)
        (1024, 4, "1K d=4"),
        (2048, 4, "2K d=4"),
        # Medium sparse
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        # Large sparse (most problematic)
        (8192, 2, "8K d=2"),
        (8192, 4, "8K d=4"),
        (16384, 2, "16K d=2"),
        (16384, 4, "16K d=4"),
        # Dense for comparison
        (4096, 1, "4K Dense"),
        (8192, 1, "8K Dense"),
    ]

    # Common parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    print(
        f"{'Config':<12} | {'Unified':<12} | {'Original':<12} | {'Optimized':<12} | {'Orig Ratio':<10} | {'Opt Ratio':<10} | {'Improvement':<30}"
    )
    print("-" * 120)

    for seq_len, dilation_rate, desc in configs:
        gc.collect()
        torch.cuda.empty_cache()

        try:
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
                    enable_sparse_optimization=False,  # Disable new optimizations
                )
                .cuda()
                .eval()
            )

            optimized = (
                UnifiedHilbertAttentionOptimizedEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                    enable_sparse_optimization=True,  # Enable new optimizations
                )
                .cuda()
                .eval()
            )

            # Create input
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
            )

            # Get configuration info
            config = optimized._get_optimal_config(seq_len)

            # Check which path will be used
            M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
            use_pytorch = (
                M_padded <= 512
                or not optimized._triton_available
                or (
                    optimized.enable_sparse_optimization
                    and dilation_rate >= 4
                    and M_padded <= 2048
                )
            )

            # Benchmark
            unified_time = benchmark_config(unified, x)
            original_time = benchmark_config(original, x)
            optimized_time = benchmark_config(optimized, x)

            orig_ratio = original_time / unified_time
            opt_ratio = optimized_time / unified_time

            # Analysis
            if opt_ratio < orig_ratio * 0.9:
                improvement = (
                    f"✓ {((orig_ratio - opt_ratio) / orig_ratio * 100):.0f}% better"
                )
            elif opt_ratio < orig_ratio * 1.1:
                improvement = "≈ Similar performance"
            else:
                improvement = (
                    f"✗ {((opt_ratio - orig_ratio) / orig_ratio * 100):.0f}% worse"
                )

            # Add info about path used
            if use_pytorch:
                improvement += " (PyTorch)"
            else:
                improvement += f" (Triton {config['block_m']}x{config['block_n']})"

            print(
                f"{desc:<12} | {unified_time:<12.2f} | {original_time:<12.2f} | "
                f"{optimized_time:<12.2f} | {orig_ratio:<10.2f}x | {opt_ratio:<10.2f}x | {improvement:<30}"
            )

        except Exception as e:
            print(f"{desc:<12} | Error: {str(e)}")
            continue

    # Summary by pattern type
    print("\n=== Summary ===")
    print("\n1. Small Very Sparse (d=4, seq <= 2K):")
    print("   - Should use PyTorch path for better performance")
    print("   - Avoids Triton overhead for small computations")

    print("\n2. Medium Sparse (d=2):")
    print("   - Uses asymmetric blocks (64x32) for better efficiency")
    print("   - Disables fused softmax to reduce overhead")

    print("\n3. Large Sparse:")
    print("   - Adaptive block sizing based on sparsity level")
    print("   - Smaller blocks for very sparse patterns")

    print("\n4. Dense Patterns:")
    print("   - Should maintain excellent performance")
    print("   - No regression from sparse optimizations")


if __name__ == "__main__":
    main()
