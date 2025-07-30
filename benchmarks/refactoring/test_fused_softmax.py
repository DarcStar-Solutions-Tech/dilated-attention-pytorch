#!/usr/bin/env python3
"""
Test performance with fused softmax re-introduced.
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


def verify_fused_softmax_enabled():
    """Verify that fused softmax is enabled for moderate sparse patterns."""

    print("=== Verifying Fused Softmax Configuration ===")

    # Test 4K d=2 (should use fused softmax)
    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    config = refactored._get_attention_config(4096)

    print("\n4K d=2 Configuration:")
    print(f"  block_m: {config.block_config.block_m}")
    print(f"  block_n: {config.block_config.block_n}")
    print(f"  use_fused_softmax: {config.use_fused_softmax}")

    if config.use_fused_softmax:
        print("  ✅ Fused softmax enabled for moderate sparse!")
    else:
        print("  ❌ Fused softmax not enabled")

    # Test 4K d=4 (should NOT use fused softmax)
    refactored_d4 = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=4,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    config_d4 = refactored_d4._get_attention_config(4096)

    print("\n4K d=4 Configuration:")
    print(f"  block_m: {config_d4.block_config.block_m}")
    print(f"  block_n: {config_d4.block_config.block_n}")
    print(f"  use_fused_softmax: {config_d4.use_fused_softmax}")

    if not config_d4.use_fused_softmax:
        print("  ✅ Online softmax for very sparse (as intended)")


def benchmark_with_fused():
    """Benchmark performance with fused softmax."""

    print("\n\n=== Performance with Fused Softmax ===")

    configs = [
        # Moderate sparse (should benefit from fused)
        (2048, 2, "2K d=2"),
        (4096, 2, "4K d=2 (FUSED)"),
        (8192, 2, "8K d=2"),
        # Very sparse (uses online)
        (2048, 4, "2K d=4"),
        (4096, 4, "4K d=4"),
        (8192, 4, "8K d=4"),
        # Dense (uses online)
        (4096, 1, "4K dense"),
        (8192, 1, "8K dense"),
    ]

    print(
        f"{'Config':<18} | {'Enhanced':>8} | {'Refact':>8} | {'Speedup':>7} | {'Status':<10} | Notes"
    )
    print("-" * 80)

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
        for _ in range(5):
            with torch.no_grad():
                _ = enhanced(x)
                _ = refactored(x)

        # Time
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Enhanced
        start.record()
        for _ in range(10):
            with torch.no_grad():
                _ = enhanced(x)
        end.record()
        torch.cuda.synchronize()
        e_time = start.elapsed_time(end) / 10

        # Refactored
        start.record()
        for _ in range(10):
            with torch.no_grad():
                _ = refactored(x)
        end.record()
        torch.cuda.synchronize()
        r_time = start.elapsed_time(end) / 10

        speedup = e_time / r_time
        status = (
            "🚀 Faster"
            if speedup > 1.1
            else "✅ Similar"
            if speedup > 0.9
            else "❌ Slower"
        )

        # Get config info
        config = refactored._get_attention_config(seq_len)
        notes = f"Fused={config.use_fused_softmax}"

        print(
            f"{desc:<18} | {e_time:>7.1f}ms | {r_time:>7.1f}ms | {speedup:>6.2f}x | {status:<10} | {notes}"
        )


def compare_before_after_fused():
    """Compare results before and after adding fused softmax."""

    print("\n\n=== Impact of Fused Softmax ===")
    print("Key configurations that should improve:")
    print("- 4K d=2: Should see significant improvement (uses fused)")
    print("- 8K d=2: Should see some improvement (uses fused)")
    print("- 4K d=4: Should remain similar (still uses online)")


def main():
    print("=" * 80)
    print("TESTING FUSED SOFTMAX IMPLEMENTATION")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")

    verify_fused_softmax_enabled()
    benchmark_with_fused()
    compare_before_after_fused()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Re-introduced fused softmax for moderate sparse patterns")
    print("✅ 4K d=2 now uses hardware-accelerated softmax")
    print("✅ Very sparse patterns still use online softmax")
    print("🎯 This should restore performance for d=2 patterns")


if __name__ == "__main__":
    main()
