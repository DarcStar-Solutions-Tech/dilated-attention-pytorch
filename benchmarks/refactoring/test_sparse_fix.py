#!/usr/bin/env python3
"""
Test the sparse pattern fix specifically.
"""

import torch
import sys
import gc

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def verify_config_fix():
    """Verify the configuration fix for sparse patterns."""

    print("=== Verifying Sparse Configuration Fix ===")

    # Test 4K d=2 configuration
    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    config = refactored._get_attention_config(4096)

    print("\n4K d=2 Configuration (After Fix):")
    print(f"  block_m: {config.block_config.block_m}")
    print(f"  block_n: {config.block_config.block_n}")
    print(f"  block_d: {config.block_config.block_d}")
    print(f"  num_warps: {config.block_config.num_warps}")

    # Check if it matches expected
    if config.block_config.block_n == 64:
        print("  ✅ Block size fixed! Now using 64x64 blocks")
    else:
        print("  ❌ Block size not fixed, still using asymmetric blocks")


def benchmark_sparse_patterns():
    """Benchmark sparse patterns after fix."""

    print("\n\n=== Sparse Pattern Performance After Fix ===")

    configs = [
        (2048, 2, "2K d=2"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (8192, 2, "8K d=2"),
        (8192, 4, "8K d=4"),
    ]

    print(f"{'Config':<10} | {'Enhanced':>8} | {'Refact':>8} | {'Speedup':>7} | Status")
    print("-" * 50)

    batch_size = 2
    hidden_dim = 512
    num_heads = 8

    for seq_len, dilation_rate, desc in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Create models
        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dilation_rate=dilation_rate,
                enable_sparse_optimization=True,
            )
            .cuda()
            .eval()
        )

        refactored = (
            UnifiedHilbertAttentionOptimizedEnhancedRefactored(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dilation_rate=dilation_rate,
                optimization_level=OptimizationLevel.BASIC,
            )
            .cuda()
            .eval()
        )

        # Sync weights
        refactored.qkv_proj.weight.data = enhanced.qkv_proj.weight.data.clone()
        refactored.out_proj.weight.data = enhanced.out_proj.weight.data.clone()

        # Test input
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = enhanced(x)
                _ = refactored(x)

        # Time
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            with torch.no_grad():
                _ = enhanced(x)
        end.record()
        torch.cuda.synchronize()
        e_time = start.elapsed_time(end) / 10

        start.record()
        for _ in range(10):
            with torch.no_grad():
                _ = refactored(x)
        end.record()
        torch.cuda.synchronize()
        r_time = start.elapsed_time(end) / 10

        speedup = e_time / r_time
        status = "🚀" if speedup > 1.1 else "✅" if speedup > 0.9 else "❌"

        print(
            f"{desc:<10} | {e_time:>7.1f}ms | {r_time:>7.1f}ms | {speedup:>6.2f}x | {status}"
        )


def check_all_configs():
    """Check configurations for all sparse patterns."""

    print("\n\n=== All Sparse Configurations ===")

    test_cases = [
        (2048, 2, OptimizationLevel.BASIC),
        (4096, 2, OptimizationLevel.BASIC),
        (4096, 4, OptimizationLevel.BASIC),
        (8192, 2, OptimizationLevel.BASIC),
    ]

    for seq_len, dilation_rate, opt_level in test_cases:
        refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
            hidden_dim=512,
            num_heads=8,
            dilation_rate=dilation_rate,
            optimization_level=opt_level,
        ).cuda()

        config = refactored._get_attention_config(seq_len)

        print(f"\n{seq_len} d={dilation_rate} ({opt_level.name}):")
        print(f"  block: {config.block_config.block_m}x{config.block_config.block_n}")
        print(f"  warps: {config.block_config.num_warps}")


def main():
    print("=== Testing Sparse Pattern Fix ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    verify_config_fix()
    benchmark_sparse_patterns()
    check_all_configs()

    print("\n\n=== Summary ===")
    print(
        "The configuration fix should restore symmetric blocks for moderate sparse patterns."
    )
    print("This addresses the primary cause of the performance regression.")


if __name__ == "__main__":
    main()
