#!/usr/bin/env python3
"""
Quick comparison of Enhanced vs Refactored on important configurations.
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


def quick_bench(model, x, runs=10):
    """Quick benchmark."""
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)

    # Time
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(x)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def main():
    print("=" * 70)
    print("QUICK COMPARISON: Enhanced vs Refactored")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Most important configurations
    configs = [
        # Config format: (seq_len, dilation, opt_level, description)
        (1024, 1, "BASIC", "1K dense"),
        (4096, 1, "BASIC", "4K dense"),
        (8192, 1, "BASIC", "8K dense"),
        (4096, 2, "BASIC", "4K d=2"),
        (4096, 4, "BASIC", "4K d=4"),
        (8192, 2, "BASIC", "8K d=2"),
        (4096, 1, "AGGRESSIVE", "4K dense AGG"),
        (8192, 1, "AGGRESSIVE", "8K dense AGG"),
        (4096, 4, "AGGRESSIVE", "4K d=4 AGG"),
    ]

    print(
        f"{'Config':<15} | {'Enhanced':>8} | {'Refact':>8} | {'Speedup':>7} | {'Status'}"
    )
    print("-" * 60)

    hidden_dim = 512
    num_heads = 8
    batch_size = 2

    for seq_len, dilation_rate, opt_name, desc in configs:
        gc.collect()
        torch.cuda.empty_cache()

        opt_level = getattr(OptimizationLevel, opt_name)

        # Create models
        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
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
                dilation_rate=dilation_rate,
                optimization_level=opt_level,
            )
            .cuda()
            .eval()
        )

        # Sync weights
        refactored.qkv_proj.weight.data = enhanced.qkv_proj.weight.data.clone()
        refactored.out_proj.weight.data = enhanced.out_proj.weight.data.clone()

        # Test
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        e_time = quick_bench(enhanced, x)
        r_time = quick_bench(refactored, x)
        speedup = e_time / r_time

        status = "🚀" if speedup > 1.1 else "✅" if speedup > 0.9 else "❌"

        print(
            f"{desc:<15} | {e_time:>7.1f}ms | {r_time:>7.1f}ms | {speedup:>6.2f}x | {status}"
        )

    print("\n" + "=" * 60)
    print("Summary:")
    print("🚀 = Refactored >10% faster")
    print("✅ = Similar performance (±10%)")
    print("❌ = Refactored >10% slower")


if __name__ == "__main__":
    main()
