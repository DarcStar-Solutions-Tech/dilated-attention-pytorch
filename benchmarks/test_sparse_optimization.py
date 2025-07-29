#!/usr/bin/env python3
"""
Test the optimized sparse handling in Enhanced implementation.
"""

import torch
import time
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_sparse_patterns():
    """Benchmark sparse patterns with the optimized Enhanced implementation."""

    print("=== Testing Optimized Sparse Handling ===\n")

    # Test configurations
    configs = [
        # (seq_len, dilation_rate, description)
        (2048, 2, "2K d=2 (effective: 1024)"),
        (4096, 2, "4K d=2 (effective: 2048)"),
        (4096, 4, "4K d=4 (effective: 1024)"),
        (8192, 2, "8K d=2 (effective: 4096)"),
        (8192, 4, "8K d=4 (effective: 2048)"),
        (16384, 4, "16K d=4 (effective: 4096)"),
        (16384, 8, "16K d=8 (effective: 2048)"),
    ]

    # Common parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    warmup_runs = 3
    test_runs = 10

    print(f"Batch: {batch_size}, Hidden: {hidden_dim}, Heads: {num_heads}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Results header
    print(
        f"{'Config':<25} | {'Unified (ms)':<12} | {'Enhanced (ms)':<13} | {'Ratio':<10} | {'Improvement':<15}"
    )
    print("-" * 80)

    for seq_len, dilation_rate, desc in configs:
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

        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        # Create input (use float32 for compatibility)
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
        )

        # Check configuration
        config = enhanced._get_optimal_config(seq_len)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = unified(x)
                _ = enhanced(x)
        torch.cuda.synchronize()

        # Benchmark Unified
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(test_runs):
                _ = unified(x)
        torch.cuda.synchronize()
        unified_time = (time.perf_counter() - start) / test_runs * 1000

        # Benchmark Enhanced
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(test_runs):
                _ = enhanced(x)
        torch.cuda.synchronize()
        enhanced_time = (time.perf_counter() - start) / test_runs * 1000

        # Calculate improvement
        ratio = enhanced_time / unified_time
        improvement = "Faster" if ratio < 1.0 else f"{ratio:.2f}x slower"

        print(
            f"{desc:<25} | {unified_time:<12.2f} | {enhanced_time:<13.2f} | {ratio:<10.2f} | {improvement:<15}"
        )

        # Show configuration used
        if dilation_rate > 1:
            print(
                f"  → Config: block_m={config['block_m']}, block_n={config['block_n']}, "
                f"fused_softmax={config['use_fused_softmax']}"
            )

    print("\n=== Configuration Analysis ===")
    print("\nEnhanced now uses adaptive configuration for sparse patterns:")
    print("- Effective ≤ 512: Small blocks (32x32), simple softmax")
    print("- Effective ≤ 2048: Medium blocks (64x64), fused softmax")
    print("- Effective > 2048: Large blocks (128x128), fused softmax")
    print("\nOptimizations applied:")
    print("- Removed unnecessary type conversion (p.to(v.dtype))")
    print("- Simplified accumulator update")
    print("- Adaptive block sizing based on sparsity")


if __name__ == "__main__":
    benchmark_sparse_patterns()
