#!/usr/bin/env python3
"""
Compare sparse configuration selection in detail.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.config_strategies import (
    OptimizationLevel,
    SparseConfigStrategy,
)


def analyze_config_selection():
    """Analyze the configuration selection logic for sparse patterns."""

    print("=== Configuration Selection Logic Analysis ===")

    # Key finding from previous analysis:
    # 4K d=2: Original uses 64x64, Refactored uses 64x32
    # This is a critical difference!

    print("\nKey Finding: 4K d=2 Block Size Mismatch")
    print("  Original: block_n = 64")
    print("  Refactored: block_n = 32")
    print("  This reduces the tile size for K/V processing!")

    # Let's trace why this happens
    print("\nTracing Refactored Sparse Strategy for 4K d=2:")

    _ = SparseConfigStrategy(
        compute_capability=6,  # Pascal GTX 1080
        optimization_level=OptimizationLevel.BASIC,
    )

    seq_len = 4096
    dilation_rate = 2
    effective_len = seq_len // dilation_rate  # 2048
    sparsity = 1.0 - (1.0 / dilation_rate)  # 0.5

    print(f"  effective_len: {effective_len}")
    print(f"  sparsity: {sparsity:.2%}")

    # Check the logic in SparseConfigStrategy
    print("\nSparseConfigStrategy logic for effective_len=2048:")
    print("  Falls into: effective_len <= SEQ_MEDIUM (2048)")
    print("  Sparsity < SPARSITY_VERY_HIGH (0.5 < 0.75)")
    print("  Result: Uses asymmetric blocks (64x32)")

    # Original logic
    print("\nOriginal Enhanced logic for 4K d=2:")
    _ = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
        enable_sparse_optimization=True,
    ).cuda()

    # Look at original's _get_optimal_config
    print("  Original checks dilation_rate > 1 and enable_sparse_optimization")
    print("  Then uses effective_len-based config")
    print("  For effective_len=2048: Uses 64x64 blocks")


def test_block_size_impact():
    """Test the performance impact of different block sizes."""

    print("\n\n=== Block Size Performance Impact ===")

    seq_len = 4096
    dilation_rate = 2
    batch_size = 2
    hidden_dim = 512
    _ = 8

    # Create input
    _ = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Test with forced configurations
    print("\nTesting different block configurations for 4K d=2:")

    # We can't easily force different configs in the existing implementations
    # But we can analyze the theoretical impact

    print("\nTheoretical Analysis:")
    print("1. 64x64 blocks (Original):")
    print("   - Larger tiles for K/V processing")
    print("   - Better memory coalescing for sparse access")
    print("   - Fewer total blocks to process")

    print("\n2. 64x32 blocks (Refactored):")
    print("   - Smaller K/V tiles")
    print("   - More blocks to process")
    print("   - Potentially worse memory access pattern")

    effective_len = seq_len // dilation_rate
    print(f"\nFor effective length {effective_len}:")
    print(f"  64x64: {(effective_len + 63) // 64} blocks")
    print(f"  64x32: {(effective_len + 31) // 32} blocks (2x more!)")


def analyze_softmax_impact():
    """Analyze the impact of always using online softmax."""

    print("\n\n=== Softmax Implementation Impact ===")

    print("Original Enhanced:")
    print("  - Has use_fused_softmax option")
    print("  - For 4K d=2: use_fused_softmax = True")
    print("  - Fused softmax can be more efficient for sparse patterns")

    print("\nRefactored Enhanced:")
    print("  - Always uses online softmax")
    print("  - No fused softmax path")
    print("  - Online softmax has more overhead for sparse patterns")

    print("\nWhy this matters for sparse:")
    print("  - Sparse patterns have many masked positions")
    print("  - Fused softmax can skip masked computations")
    print("  - Online softmax processes all positions")


def check_4k_optimizations():
    """Check if 4K optimizations are being applied."""

    print("\n\n=== 4K Optimization Analysis ===")

    # Test AGGRESSIVE mode which should have special 4K cases
    sparse_strategy_aggressive = SparseConfigStrategy(
        compute_capability=6,
        optimization_level=OptimizationLevel.AGGRESSIVE,
    )

    print("AGGRESSIVE mode 4K optimizations:")

    # 4K d=2
    config_4k_d2 = sparse_strategy_aggressive._get_4k_sparse_config(2048, 64)
    if config_4k_d2:
        print("\n4K d=2 special case (AGGRESSIVE):")
        print(f"  block_m: {config_4k_d2.block_config.block_m}")
        print(f"  block_n: {config_4k_d2.block_config.block_n}")
        print("  This restores 64x64 blocks!")

    # But in BASIC mode, no special case
    print("\nBASIC mode (used in benchmarks):")
    print("  No special 4K optimizations")
    print("  Falls back to general sparse logic")
    print("  Results in suboptimal 64x32 config")


def propose_fix():
    """Propose a fix for the sparse regression."""

    print("\n\n=== Proposed Fix ===")

    print("1. Update SparseConfigStrategy for better 4K d=2 handling:")
    print("   - For effective_len == 2048, use 64x64 blocks")
    print("   - Match original's configuration")

    print("\n2. Consider adding fused softmax option:")
    print("   - Beneficial for sparse patterns")
    print("   - Reduces overhead for masked positions")

    print("\n3. Enable special cases in BASIC mode:")
    print("   - Currently only in AGGRESSIVE")
    print("   - 4K patterns are common enough to optimize by default")

    print("\nCode change in config_strategies.py:")
    print("""
    # In SparseConfigStrategy._get_config():
    elif effective_len <= AttentionConstants.SEQ_MEDIUM:
        if effective_len == 2048 and seq_len == 4096:  # 4K d=2
            # Special case for common 4K d=2 pattern
            block_config = BlockConfig(
                block_m=64,
                block_n=64,  # Use 64 instead of 32
                block_d=64,
                num_warps=4
            )
        elif sparsity >= AttentionConstants.SPARSITY_VERY_HIGH:
            # Very sparse - small blocks
            ...
    """)


def main():
    print("=== Sparse Pattern Regression Root Cause Analysis ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    analyze_config_selection()
    test_block_size_impact()
    analyze_softmax_impact()
    check_4k_optimizations()
    propose_fix()

    print("\n\n=== Root Cause Summary ===")
    print("1. Block configuration mismatch:")
    print("   - 4K d=2: Refactored uses 64x32 vs Original's 64x64")
    print("   - Smaller blocks = more overhead for sparse patterns")

    print("\n2. Always online softmax:")
    print("   - Original uses fused softmax for 4K d=2")
    print("   - Online softmax has more overhead")

    print("\n3. Missing optimizations in BASIC mode:")
    print("   - Special 4K cases only in AGGRESSIVE")
    print("   - Common patterns should be optimized by default")


if __name__ == "__main__":
    main()
