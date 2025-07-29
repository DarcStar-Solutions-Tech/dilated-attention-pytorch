#!/usr/bin/env python3
"""
Quick performance test for key configurations.
"""

import torch
import sys
import gc

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def quick_benchmark(model, x, warmup=3, runs=10):
    """Quick benchmark with fewer runs."""

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(x)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def test_key_configs():
    """Test key configurations that showed performance differences."""

    print("=== Quick Performance Test ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Key test cases
    configs = [
        (4096, 1, "4K Dense"),
        (8192, 1, "8K Dense"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (8192, 2, "8K d=2"),
    ]

    print("\nSeqLen | d | Config    | Unified | Original | Refactored | Orig/Ref")
    print("-" * 70)

    for seq_len, dilation_rate, desc in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Create models
        unified = (
            UnifiedHilbertAttention(
                hidden_dim=512,
                num_heads=8,
                segment_size=128,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        original = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=512,
                num_heads=8,
                segment_size=128,
                dilation_rate=dilation_rate,
                enable_sparse_optimization=True,
            )
            .cuda()
            .eval()
        )

        refactored = (
            UnifiedHilbertAttentionOptimizedEnhancedRefactored(
                hidden_dim=512,
                num_heads=8,
                segment_size=128,
                dilation_rate=dilation_rate,
                optimization_level=OptimizationLevel.BASIC,
            )
            .cuda()
            .eval()
        )

        # Sync weights
        original.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
        original.out_proj.weight.data = unified.out_proj.weight.data.clone()
        refactored.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
        refactored.out_proj.weight.data = unified.out_proj.weight.data.clone()

        # Test input
        x = torch.randn(2, seq_len, 512, device="cuda", dtype=torch.float32)

        # Benchmark
        try:
            u_time = quick_benchmark(unified, x)
            o_time = quick_benchmark(original, x)
            r_time = quick_benchmark(refactored, x)

            ratio = o_time / r_time
            status = "✅" if 0.9 <= ratio <= 1.1 else ("🚀" if ratio > 1.1 else "❌")

            print(
                f"{seq_len:<6} | {dilation_rate} | {desc:<9} | {u_time:>7.2f} | {o_time:>8.2f} | {r_time:>10.2f} | {ratio:>6.2f}x {status}"
            )

        except Exception as e:
            print(f"{seq_len:<6} | {dilation_rate} | {desc:<9} | ERROR: {str(e)}")


def check_multi_row_fix():
    """Verify multi-row is now enabled correctly."""

    print("\n\n=== Multi-Row Configuration Check ===")

    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    for seq_len in [2048, 4096, 8192]:
        config = refactored._get_attention_config(seq_len)
        print(
            f"Seq {seq_len}: multi_row={config.use_multi_row}, rows_per_block={config.rows_per_block}"
        )


def main():
    test_key_configs()
    check_multi_row_fix()

    print("\n=== Summary ===")
    print("✅ = Performance maintained (±10%)")
    print("🚀 = Refactored faster (>10%)")
    print("❌ = Refactored slower (>10%)")


if __name__ == "__main__":
    main()
