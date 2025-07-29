#!/usr/bin/env python3
"""
Debug kernel parameter differences.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def trace_kernel_calls():
    """Trace kernel calls to see parameter differences."""

    print("=== Kernel Parameter Analysis ===")

    # Test case
    seq_len = 4096
    batch_size = 2
    hidden_dim = 512
    num_heads = 8

    # Create models
    original = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
        )
        .cuda()
        .eval()
    )

    refactored = (
        UnifiedHilbertAttentionOptimizedEnhancedRefactored(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
            optimization_level=OptimizationLevel.BASIC,
        )
        .cuda()
        .eval()
    )

    # Create input
    _ = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Get configurations
    orig_config = original._get_optimal_config(seq_len)
    refact_config = refactored._get_attention_config(seq_len)

    print("\nFor 4K Dense:")
    print("\nOriginal kernel parameters:")
    print(f"  block_m: {orig_config['block_m']}")
    print(f"  block_n: {orig_config['block_n']}")
    print(f"  block_d: {orig_config['block_d']}")
    print(f"  num_warps: {orig_config['num_warps']}")
    print(f"  use_fused_softmax: {orig_config.get('use_fused_softmax', 'not set')}")
    print(f"  enable_prefetch: {orig_config.get('enable_prefetch', False)}")
    print(f"  rows_per_block: {orig_config.get('rows_per_block', 1)}")

    print("\nRefactored kernel parameters:")
    print(f"  block_m: {refact_config.block_config.block_m}")
    print(f"  block_n: {refact_config.block_config.block_n}")
    print(f"  block_d: {refact_config.block_config.block_d}")
    print(f"  num_warps: {refact_config.block_config.num_warps}")
    print("  use_fused_softmax: REMOVED (always online softmax)")
    print(f"  enable_prefetch: {refact_config.enable_prefetch}")
    print(f"  rows_per_block: {refact_config.rows_per_block}")

    # Check grid sizes
    M_padded = ((seq_len + 128 - 1) // 128) * 128
    num_blocks_m_orig = (M_padded + orig_config["block_m"] - 1) // orig_config[
        "block_m"
    ]
    num_blocks_m_refact = (
        M_padded + refact_config.block_config.block_m - 1
    ) // refact_config.block_config.block_m

    print("\nGrid configuration:")
    print(f"  M_padded: {M_padded}")
    print(f"  Original blocks: {num_blocks_m_orig}")
    print(f"  Refactored blocks: {num_blocks_m_refact}")

    if orig_config.get("rows_per_block", 1) > 1:
        grid_orig = (
            (num_blocks_m_orig // orig_config["rows_per_block"])
            * batch_size
            * num_heads
        )
    else:
        grid_orig = num_blocks_m_orig * batch_size * num_heads

    if refact_config.rows_per_block > 1:
        grid_refact = (
            (num_blocks_m_refact // refact_config.rows_per_block)
            * batch_size
            * num_heads
        )
    else:
        grid_refact = num_blocks_m_refact * batch_size * num_heads

    print(f"  Original grid size: {grid_orig}")
    print(f"  Refactored grid size: {grid_refact}")


def check_kernel_differences():
    """Check differences in kernel code."""

    print("\n\n=== Kernel Code Differences ===")

    print("\n1. Original kernel parameters (23 + meta):")
    print("   - Includes USE_FUSED_SOFTMAX")
    print("   - Includes ENABLE_PREFETCH")
    print("   - Includes mask_value parameter")

    print("\n2. Refactored kernel parameters (reduced):")
    print("   - Removed USE_FUSED_SOFTMAX (always online)")
    print("   - Removed ENABLE_PREFETCH (dead code)")
    print("   - Moved mask_value to constant")

    print("\n3. Softmax implementation:")
    print("   - Original: Two paths (fused vs standard)")
    print("   - Refactored: Single unified path")

    print("\n4. Potential performance impact:")
    print("   - Fewer parameters = less register pressure")
    print("   - Single path = better instruction cache")
    print("   - But: Always using online softmax (no simple path)")


def test_small_sequence():
    """Test with small sequence to isolate overhead."""

    print("\n\n=== Small Sequence Test ===")

    import time

    seq_len = 512  # Should use PyTorch path
    batch_size = 1
    hidden_dim = 128
    num_heads = 4

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    original = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        .cuda()
        .eval()
    )

    refactored = (
        UnifiedHilbertAttentionOptimizedEnhancedRefactored(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        .cuda()
        .eval()
    )

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original(x)
            _ = refactored(x)

    # Time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = original(x)
    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - start) * 10  # ms per call

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = refactored(x)
    torch.cuda.synchronize()
    refact_time = (time.perf_counter() - start) * 10

    print("Small sequence (512) performance:")
    print(f"  Original: {orig_time:.3f}ms")
    print(f"  Refactored: {refact_time:.3f}ms")
    print(f"  Overhead: {refact_time - orig_time:.3f}ms")

    # Check which path
    print("\nExecution path:")
    print(f"  Uses PyTorch: {seq_len <= 512}")


def main():
    print("=== Debugging Kernel Parameters ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    trace_kernel_calls()
    check_kernel_differences()
    test_small_sequence()

    print("\n\n=== Hypothesis ===")
    print("Performance differences may be due to:")
    print("1. Always using online softmax (no simple path)")
    print("2. Strategy pattern overhead for configuration")
    print("3. Different kernel parameter packing")
    print("4. Potential compiler optimization differences")


if __name__ == "__main__":
    main()
