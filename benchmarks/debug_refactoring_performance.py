#!/usr/bin/env python3
"""
Debug performance differences between original and refactored Enhanced.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def compare_configurations():
    """Compare configurations selected by original vs refactored."""

    print("=== Configuration Comparison ===")

    # Test cases where performance differs
    test_cases = [
        (4096, 1, "4K Dense - Basic"),
        (4096, 1, "4K Dense - Aggressive"),
        (8192, 1, "8K Dense - Basic"),
        (8192, 1, "8K Dense - Aggressive"),
        (4096, 2, "4K d=2 - Aggressive"),
        (4096, 4, "4K d=4 - Aggressive"),
    ]

    for seq_len, dilation_rate, desc in test_cases:
        print(f"\n{desc}:")

        # Original with appropriate flags
        if "Aggressive" in desc:
            original = UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=512,
                num_heads=8,
                dilation_rate=dilation_rate,
                enable_4k_optimization=True,
                enable_8k_optimization=True,
                enable_sparse_optimization=True,
            ).cuda()
            refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
                hidden_dim=512,
                num_heads=8,
                dilation_rate=dilation_rate,
                optimization_level=OptimizationLevel.AGGRESSIVE,
            ).cuda()
        else:
            original = UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=512,
                num_heads=8,
                dilation_rate=dilation_rate,
                enable_sparse_optimization=True,
            ).cuda()
            refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
                hidden_dim=512,
                num_heads=8,
                dilation_rate=dilation_rate,
                optimization_level=OptimizationLevel.BASIC,
            ).cuda()

        # Get configs
        orig_config = original._get_optimal_config(seq_len)
        refact_config = refactored._get_attention_config(seq_len)

        print("  Original:")
        print(f"    block_m={orig_config['block_m']}, block_n={orig_config['block_n']}")
        print(f"    num_warps={orig_config['num_warps']}")
        print(
            f"    use_fused_softmax={orig_config.get('use_fused_softmax', 'not set')}"
        )
        print(f"    rows_per_block={orig_config.get('rows_per_block', 1)}")

        print("  Refactored:")
        print(
            f"    block_m={refact_config.block_config.block_m}, block_n={refact_config.block_config.block_n}"
        )
        print(f"    num_warps={refact_config.block_config.num_warps}")
        print(f"    use_multi_row={refact_config.use_multi_row}")
        print(f"    rows_per_block={refact_config.rows_per_block}")

        # Check for differences
        if (
            orig_config["block_m"] != refact_config.block_config.block_m
            or orig_config["block_n"] != refact_config.block_config.block_n
        ):
            print("  ⚠️  BLOCK SIZE MISMATCH!")

        if orig_config["num_warps"] != refact_config.block_config.num_warps:
            print("  ⚠️  WARP COUNT MISMATCH!")

        if orig_config.get("rows_per_block", 1) != refact_config.rows_per_block:
            print("  ⚠️  ROWS PER BLOCK MISMATCH!")


def check_kernel_execution_path():
    """Check which execution path is being taken."""

    print("\n\n=== Execution Path Analysis ===")

    # Create test case
    seq_len = 4096
    batch_size = 2
    hidden_dim = 512

    _ = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Original
    original = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=hidden_dim,
        num_heads=8,
    ).cuda()

    # Refactored
    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=hidden_dim,
        num_heads=8,
    ).cuda()

    print(f"\nFor seq_len={seq_len}:")

    # Check thresholds
    print(
        f"  Hilbert threshold: {original.hilbert_threshold} / {refactored.hilbert_threshold}"
    )
    print(f"  PyTorch threshold: 512 / {refactored._get_attention_config(512)}")

    # Check if Triton is available
    print(
        f"  Triton available: {original._triton_available} / {refactored._triton_available}"
    )

    # Get padded length
    M_padded = ((seq_len + 128 - 1) // 128) * 128
    print(f"  Padded length: {M_padded}")

    # Determine path
    use_pytorch_orig = M_padded <= 512 or not original._triton_available
    use_pytorch_refact = M_padded <= 512 or not refactored._triton_available

    print(f"  Original uses PyTorch: {use_pytorch_orig}")
    print(f"  Refactored uses PyTorch: {use_pytorch_refact}")


def profile_kernel_launch():
    """Profile kernel launch overhead."""

    print("\n\n=== Kernel Launch Profiling ===")

    import time

    # Small sequence to minimize compute time
    seq_len = 1024
    batch_size = 1
    hidden_dim = 256
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

    # Time configuration generation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        _ = original._get_optimal_config(seq_len)
    config_time_orig = (time.perf_counter() - start) * 1000

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        _ = refactored._get_attention_config(seq_len)
    config_time_refact = (time.perf_counter() - start) * 1000

    print("Configuration generation time (1000 calls):")
    print(f"  Original: {config_time_orig:.2f}ms")
    print(f"  Refactored: {config_time_refact:.2f}ms")
    print(f"  Overhead: {config_time_refact - config_time_orig:.2f}ms")


def main():
    print("=== Debugging Refactoring Performance ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    compare_configurations()
    check_kernel_execution_path()
    profile_kernel_launch()

    print("\n\n=== Analysis ===")
    print("Potential issues identified:")
    print("1. Configuration mismatches for some sequences")
    print("2. Additional overhead from strategy pattern")
    print("3. Different execution paths being selected")


if __name__ == "__main__":
    main()
