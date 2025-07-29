#!/usr/bin/env python3
"""
Test that the refactored Enhanced implementation produces identical results.
"""

import torch
import pytest
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttentionOptimizedEnhanced,
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


@pytest.mark.parametrize(
    "seq_len,dilation_rate",
    [
        (512, 1),  # Small dense
        (2048, 1),  # Medium dense
        (4096, 1),  # Large dense
        (8192, 1),  # XLarge dense
        (1024, 2),  # Small sparse
        (4096, 2),  # 4K d=2 (special case)
        (4096, 4),  # 4K d=4 (special case)
        (8192, 2),  # Large sparse
        (8192, 4),  # Large very sparse
    ],
)
@pytest.mark.parametrize(
    "optimization_level",
    [
        OptimizationLevel.NONE,
        OptimizationLevel.BASIC,
        OptimizationLevel.AGGRESSIVE,
    ],
)
def test_refactored_equivalence(seq_len, dilation_rate, optimization_level):
    """Test that refactored version produces identical results."""

    # Skip if no CUDA
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    # Create models
    original = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            enable_multi_row=(optimization_level != OptimizationLevel.NONE),
            enable_8k_optimization=(optimization_level == OptimizationLevel.AGGRESSIVE),
            enable_4k_optimization=(optimization_level == OptimizationLevel.AGGRESSIVE),
            enable_sparse_optimization=(optimization_level != OptimizationLevel.NONE),
        )
        .cuda()
        .eval()
    )

    refactored = (
        UnifiedHilbertAttentionOptimizedEnhancedRefactored(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            optimization_level=optimization_level,
        )
        .cuda()
        .eval()
    )

    # Synchronize weights
    refactored.qkv_proj.weight.data = original.qkv_proj.weight.data.clone()
    refactored.out_proj.weight.data = original.out_proj.weight.data.clone()

    # Test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        out_original = original(x)
        out_refactored = refactored(x)

    # Check outputs match
    diff = (out_original - out_refactored).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nSeq={seq_len}, d={dilation_rate}, opt={optimization_level.name}:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # Tolerance based on numerical precision
    assert max_diff < 1e-4, f"Max difference {max_diff} exceeds tolerance"
    assert mean_diff < 1e-5, f"Mean difference {mean_diff} exceeds tolerance"


def test_config_extraction():
    """Test that configuration extraction works correctly."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create models
    original = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=512,
        num_heads=8,
        enable_4k_optimization=True,
        enable_sparse_optimization=True,
    ).cuda()

    refactored = UnifiedHilbertAttentionOptimizedEnhancedRefactored(
        hidden_dim=512,
        num_heads=8,
        optimization_level=OptimizationLevel.AGGRESSIVE,
    ).cuda()

    # Test configuration for various sequence lengths
    test_configs = [
        (4096, 4),  # 4K d=4 special case
        (4096, 2),  # 4K d=2 special case
        (8192, 1),  # 8K dense
        (2048, 2),  # Medium sparse
    ]

    for seq_len, dilation_rate in test_configs:
        # Update dilation rate
        original.dilation_rate = dilation_rate
        refactored.dilation_rate = dilation_rate

        # Get configs
        orig_config = original._get_optimal_config(seq_len)
        refactored_config = refactored._get_attention_config(seq_len)

        print(f"\nConfig for seq={seq_len}, d={dilation_rate}:")
        print(
            f"  Original: block_m={orig_config['block_m']}, block_n={orig_config['block_n']}"
        )
        print(
            f"  Refactored: block_m={refactored_config.block_config.block_m}, "
            f"block_n={refactored_config.block_config.block_n}"
        )

        # Check key parameters match
        assert orig_config["block_m"] == refactored_config.block_config.block_m
        assert orig_config["block_n"] == refactored_config.block_config.block_n
        assert orig_config["num_warps"] == refactored_config.block_config.num_warps


def test_performance_comparison():
    """Compare performance of original vs refactored."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import time

    # Test configuration
    seq_len = 4096
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    dilation_rate = 4

    # Create models
    original = (
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

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = original(x)
            _ = refactored(x)

    # Time original
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            _ = original(x)
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) * 1000 / 20

    # Time refactored
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            _ = refactored(x)
    torch.cuda.synchronize()
    refactored_time = (time.perf_counter() - start) * 1000 / 20

    print("\nPerformance comparison (4K d=4):")
    print(f"  Original: {original_time:.2f}ms")
    print(f"  Refactored: {refactored_time:.2f}ms")
    print(f"  Ratio: {refactored_time / original_time:.2f}x")

    # Should be within 10% of original performance
    assert abs(refactored_time - original_time) / original_time < 0.1


if __name__ == "__main__":
    print("Testing Enhanced kernel refactoring...")

    # Run basic equivalence test
    test_refactored_equivalence(4096, 4, OptimizationLevel.AGGRESSIVE)

    # Test configuration extraction
    test_config_extraction()

    # Test performance
    test_performance_comparison()

    print("\nAll tests passed!")
