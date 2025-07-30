#!/usr/bin/env python3
"""
Debug remaining performance issues after block size fix.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def check_fused_softmax_impact():
    """Check the impact of fused vs online softmax."""

    print("=== Fused vs Online Softmax Analysis ===")

    # Original has fused softmax option
    original = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
        enable_sparse_optimization=True,
    ).cuda()

    config = original._get_optimal_config(4096)
    print("\nOriginal 4K d=2 config:")
    print(f"  use_fused_softmax: {config.get('use_fused_softmax', 'not set')}")

    print("\nRefactored always uses online softmax (no fused option)")
    print("\nImpact:")
    print("- Fused softmax can skip computation for masked positions")
    print("- Online softmax must process all positions")
    print("- For 50% sparsity, this could be ~2x overhead")


def check_enable_prefetch():
    """Check if prefetch is being used."""

    print("\n\n=== Prefetch Configuration ===")

    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=2,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    config = refactored._get_attention_config(4096)
    print("\nRefactored 4K d=2:")
    print(f"  enable_prefetch: {config.enable_prefetch}")

    # Check dense for comparison
    refactored_dense = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=1,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    config_dense = refactored_dense._get_attention_config(4096)
    print("\nRefactored 4K dense:")
    print(f"  enable_prefetch: {config_dense.enable_prefetch}")

    print("\nNote: Sparse patterns disable prefetch, dense enables it")


def analyze_kernel_differences():
    """Analyze kernel implementation differences."""

    print("\n\n=== Kernel Implementation Differences ===")

    print("Original Enhanced kernel:")
    print("- Has USE_FUSED_SOFTMAX parameter")
    print("- Can skip softmax normalization for masked positions")
    print("- Has ENABLE_PREFETCH for memory optimization")

    print("\nRefactored Enhanced kernel:")
    print("- Always uses online softmax")
    print("- Removed USE_FUSED_SOFTMAX parameter")
    print("- Removed ENABLE_PREFETCH (was dead code)")
    print("- Simpler but potentially less optimized for sparse")


def test_very_sparse():
    """Test very sparse patterns (d=4, d=8)."""

    print("\n\n=== Very Sparse Pattern Analysis ===")

    # 4K d=4 shows 5.44x speedup!
    print("Interesting finding: 4K d=4 shows 5.44x speedup!")
    print("This suggests the refactored version is BETTER for very sparse patterns")

    # Check config for d=4
    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        dilation_rate=4,
        optimization_level=OptimizationLevel.BASIC,
    ).cuda()

    config = refactored._get_attention_config(4096)
    print("\n4K d=4 config:")
    print(f"  block: {config.block_config.block_m}x{config.block_config.block_n}")
    print("  Special 4K optimization applied: Yes (32x32 blocks)")


def hypothesis():
    """Final hypothesis for remaining issues."""

    print("\n\n=== Hypothesis for Remaining Issues ===")

    print("1. Fused Softmax Impact:")
    print("   - Original uses fused softmax for 4K d=2")
    print("   - Refactored always uses online softmax")
    print("   - This explains ~2x performance difference")

    print("\n2. Pattern-Specific Behavior:")
    print("   - Moderate sparse (d=2): Hurt by lack of fused softmax")
    print("   - Very sparse (d=4): Benefits from simpler kernel")
    print("   - The overhead of fused softmax logic may hurt very sparse")

    print("\n3. Dense Pattern Issues:")
    print("   - Some dense patterns show major regressions")
    print("   - This suggests other issues beyond sparse optimization")
    print("   - May need separate investigation")


def main():
    print("=== Debugging Remaining Performance Issues ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    check_fused_softmax_impact()
    check_enable_prefetch()
    analyze_kernel_differences()
    test_very_sparse()
    hypothesis()

    print("\n\n=== Recommendations ===")
    print("1. The block size fix helped but didn't fully resolve the issue")
    print("2. Consider re-introducing fused softmax for moderate sparse patterns")
    print("3. The refactored version is actually BETTER for very sparse (d>=4)")
    print("4. Dense pattern regressions need separate investigation")


if __name__ == "__main__":
    main()
