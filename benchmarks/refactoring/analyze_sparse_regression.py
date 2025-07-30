#!/usr/bin/env python3
"""
Analyze sparse pattern performance regression in refactored Enhanced kernel.
"""

import torch
import sys
import time

sys.path.append("..")

from dilated_attention_pytorch.kernels import UnifiedHilbertAttentionOptimizedEnhanced
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced_refactored import (
    UnifiedHilbertAttentionOptimizedEnhancedRefactored,
)
from dilated_attention_pytorch.kernels.config_strategies import OptimizationLevel


def analyze_sparse_config():
    """Analyze configuration differences for sparse patterns."""

    print("=== Sparse Configuration Analysis ===")

    # Test cases that showed regression
    test_cases = [
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (2048, 4, "2K d=4"),
    ]

    for seq_len, dilation_rate, desc in test_cases:
        print(f"\n{desc} Configuration:")

        # Create models
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

        # Get configurations
        orig_config = original._get_optimal_config(seq_len)
        refact_config = refactored._get_attention_config(seq_len)

        print("  Original config:")
        print(f"    block_m: {orig_config['block_m']}")
        print(f"    block_n: {orig_config['block_n']}")
        print(f"    block_d: {orig_config['block_d']}")
        print(f"    num_warps: {orig_config['num_warps']}")
        print(
            f"    use_fused_softmax: {orig_config.get('use_fused_softmax', 'not set')}"
        )
        print(f"    rows_per_block: {orig_config.get('rows_per_block', 1)}")

        print("  Refactored config:")
        print(f"    block_m: {refact_config.block_config.block_m}")
        print(f"    block_n: {refact_config.block_config.block_n}")
        print(f"    block_d: {refact_config.block_config.block_d}")
        print(f"    num_warps: {refact_config.block_config.num_warps}")
        print(f"    use_multi_row: {refact_config.use_multi_row}")
        print(f"    rows_per_block: {refact_config.rows_per_block}")

        # Calculate effective sequence length
        effective_len = seq_len // dilation_rate
        sparsity = 1.0 - (1.0 / dilation_rate)
        print("  Sparse properties:")
        print(f"    effective_len: {effective_len}")
        print(f"    sparsity: {sparsity:.2%}")


def trace_sparse_execution():
    """Trace execution paths for sparse patterns."""

    print("\n\n=== Sparse Execution Path Analysis ===")

    # Test 4K d=2 which showed regression
    seq_len = 4096
    dilation_rate = 2
    batch_size = 2
    hidden_dim = 512
    num_heads = 8

    _ = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Original implementation
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

    print("\nOriginal Enhanced - Sparse Optimization Logic:")

    # Check original's sparse optimization path
    _ = original._get_optimal_config(seq_len)
    print(f"  enable_sparse_optimization: {original.enable_sparse_optimization}")
    print(f"  dilation_rate: {dilation_rate}")

    # The original has this logic in _get_optimal_config:
    effective_len = seq_len // dilation_rate
    print(f"  effective_len for config: {effective_len}")

    # Refactored implementation
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

    print("\nRefactored Enhanced - Strategy Selection:")
    print(f"  Using sparse strategy: {refactored.dilation_rate > 1}")
    print(f"  optimization_level: {refactored.optimization_level}")


def benchmark_sparse_kernels():
    """Benchmark specific sparse pattern executions."""

    print("\n\n=== Sparse Kernel Performance Analysis ===")

    # Focus on 4K d=2 which showed regression
    seq_len = 4096
    dilation_rate = 2
    batch_size = 2
    hidden_dim = 512
    num_heads = 8

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

    # Sync weights
    refactored.qkv_proj.weight.data = original.qkv_proj.weight.data.clone()
    refactored.out_proj.weight.data = original.out_proj.weight.data.clone()

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Profile kernel launches
    print("\nKernel Launch Analysis:")

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = original(x)
            _ = refactored(x)

    # Time individual components
    torch.cuda.synchronize()

    # Time QKV projection
    start = time.perf_counter()
    with torch.no_grad():
        _ = original.qkv_proj(x)
    torch.cuda.synchronize()
    qkv_time_orig = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with torch.no_grad():
        _ = refactored.qkv_proj(x)
    torch.cuda.synchronize()
    qkv_time_refact = (time.perf_counter() - start) * 1000

    print("  QKV projection time:")
    print(f"    Original: {qkv_time_orig:.2f}ms")
    print(f"    Refactored: {qkv_time_refact:.2f}ms")

    # Get grid sizes
    M_padded = ((seq_len + 128 - 1) // 128) * 128

    orig_config = original._get_optimal_config(seq_len)
    refact_config = refactored._get_attention_config(seq_len)

    num_blocks_m_orig = (M_padded + orig_config["block_m"] - 1) // orig_config[
        "block_m"
    ]
    num_blocks_m_refact = (
        M_padded + refact_config.block_config.block_m - 1
    ) // refact_config.block_config.block_m

    grid_orig = num_blocks_m_orig * batch_size * num_heads
    grid_refact = num_blocks_m_refact * batch_size * num_heads

    print("\n  Grid configuration:")
    print(f"    Original grid size: {grid_orig}")
    print(f"    Refactored grid size: {grid_refact}")
    print(f"    Grid difference: {grid_refact - grid_orig} blocks")


def check_sparse_special_cases():
    """Check if sparse special cases are being applied correctly."""

    print("\n\n=== Sparse Special Case Analysis ===")

    # Check 4K sparse optimizations
    from dilated_attention_pytorch.kernels.config_strategies import (
        SparseConfigStrategy,
    )

    sparse_strategy = SparseConfigStrategy(
        compute_capability=6,  # Pascal
        optimization_level=OptimizationLevel.AGGRESSIVE,
    )

    print("\n4K Sparse Special Cases (AGGRESSIVE):")

    # Test 4K d=4 special case
    config_4k_d4 = sparse_strategy._get_4k_sparse_config(
        1024, 64
    )  # effective_len=1024 for 4K d=4
    if config_4k_d4:
        print("  4K d=4 special config found:")
        print(f"    block_m: {config_4k_d4.block_config.block_m}")
        print(f"    block_n: {config_4k_d4.block_config.block_n}")
        print(f"    num_warps: {config_4k_d4.block_config.num_warps}")

    # Test 4K d=2 special case
    config_4k_d2 = sparse_strategy._get_4k_sparse_config(
        2048, 64
    )  # effective_len=2048 for 4K d=2
    if config_4k_d2:
        print("  4K d=2 special config found:")
        print(f"    block_m: {config_4k_d2.block_config.block_m}")
        print(f"    block_n: {config_4k_d2.block_config.block_n}")
        print(f"    num_warps: {config_4k_d2.block_config.num_warps}")


def analyze_memory_access_patterns():
    """Analyze memory access patterns for sparse kernels."""

    print("\n\n=== Memory Access Pattern Analysis ===")

    seq_len = 4096
    dilation_rate = 2
    segment_size = 128

    print(f"\nFor {seq_len} tokens with dilation={dilation_rate}:")

    # Calculate sparse access pattern
    num_segments = (seq_len + segment_size - 1) // segment_size
    print(f"  Number of segments: {num_segments}")

    total_active = 0
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)
        seg_len = seg_end - seg_start

        # Active positions in this segment
        active_in_segment = (seg_len + dilation_rate - 1) // dilation_rate
        total_active += active_in_segment

    print(f"  Total active positions: {total_active}")
    print(f"  Sparsity: {1 - total_active / seq_len:.2%}")
    print(f"  Memory accesses per segment: ~{segment_size // dilation_rate}")

    # Compare block sizes
    print("\nBlock size impact on sparse patterns:")
    print(f"  With 32x32 blocks: {(total_active + 31) // 32} blocks needed")
    print(f"  With 64x32 blocks: {(total_active + 63) // 64} blocks needed")
    print(f"  With 64x64 blocks: {(total_active + 63) // 64} blocks needed")


def main():
    print("=== Debugging Sparse Pattern Performance Regression ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    analyze_sparse_config()
    trace_sparse_execution()
    benchmark_sparse_kernels()
    check_sparse_special_cases()
    analyze_memory_access_patterns()

    print("\n\n=== Hypothesis ===")
    print("Possible causes of sparse pattern regression:")
    print("1. Different block configurations affecting memory coalescing")
    print("2. Strategy pattern overhead for configuration selection")
    print("3. Missing sparse-specific optimizations in refactored version")
    print("4. Different grid sizes leading to suboptimal GPU utilization")
    print(
        "5. Always using online softmax (no fused path) may hurt sparse patterns more"
    )


if __name__ == "__main__":
    main()
