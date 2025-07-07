#!/usr/bin/env python3
"""
Test post-pattern Hilbert optimization implementation.
"""

import torch
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
    PostPatternHilbertOptimizer,
    create_post_pattern_hilbert_attention,
)
from dilated_attention_pytorch import create_block_sparse_attention, SparsePatternConfig


def test_pattern_preservation():
    """Test that post-pattern optimization preserves the sparse pattern."""
    print("Testing Pattern Preservation")
    print("=" * 50)

    # Create model
    model = create_post_pattern_hilbert_attention(
        segment_lengths=[1024],
        dilation_rates=[4],
        sparsity_ratio=0.1,
        block_size=64,
    )

    # Analyze optimization impact
    analysis = model.analyze_optimization_impact(1024)

    print(f"Number of blocks: {analysis['num_blocks']}")
    print(f"Number of connections: {analysis['num_connections']}")
    print(f"Standard cache metric: {analysis['standard_cache_metric']:.2f}")
    print(f"Optimized cache metric: {analysis['optimized_cache_metric']:.2f}")
    print(f"Cache improvement: {analysis['improvement']:.1f}%")
    print(f"Pattern preserved: {analysis['pattern_preserved']}")

    assert analysis["pattern_preserved"], "Pattern must be preserved!"
    print("\n✓ Pattern preservation test passed!")


def test_optimizer_logic():
    """Test the optimizer's pattern analysis and ordering logic."""
    print("\n\nTesting Optimizer Logic")
    print("=" * 50)

    optimizer = PostPatternHilbertOptimizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different pattern types
    test_cases = [
        # Diagonal pattern
        {
            "name": "Diagonal pattern",
            "rows": [0, 1, 2, 3, 4, 5],
            "cols": [0, 1, 2, 3, 4, 5],
            "num_blocks": 8,
        },
        # Long-range pattern
        {
            "name": "Long-range pattern",
            "rows": [0, 0, 1, 1, 7, 7],
            "cols": [4, 7, 5, 6, 0, 3],
            "num_blocks": 8,
        },
        # Mixed pattern
        {
            "name": "Mixed pattern",
            "rows": [0, 0, 1, 1, 2, 2, 3, 3],
            "cols": [0, 2, 1, 3, 2, 4, 3, 5],
            "num_blocks": 8,
        },
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        row_tensor = torch.tensor(case["rows"], device=device)
        col_tensor = torch.tensor(case["cols"], device=device)

        # Analyze pattern
        pattern_info = optimizer._analyze_access_pattern(
            row_tensor, col_tensor, case["num_blocks"]
        )
        print(f"  Pattern type: {pattern_info['type']}")
        print(f"  Avg distance: {pattern_info['avg_distance']:.2f}")

        # Optimize order
        opt_rows, opt_cols = optimizer.optimize_block_processing_order(
            row_tensor, col_tensor, case["num_blocks"]
        )

        print(f"  Original order: {list(zip(case['rows'], case['cols']))}")
        print(f"  Optimized order: {list(zip(opt_rows.tolist(), opt_cols.tolist()))}")


def test_forward_pass():
    """Test forward pass with post-pattern optimization."""
    print("\n\nTesting Forward Pass")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_post_pattern_hilbert_attention(
        segment_lengths=[1024],
        dilation_rates=[2],
        sparsity_ratio=0.15,
        block_size=64,
    ).to(device)

    # Create inputs
    batch_size = 2
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Forward pass
    try:
        output, attention_info = model(q, k, v, return_attention_weights=True)
        print(f"Output shape: {output.shape}")
        print(f"Optimization: {attention_info.get('optimization', 'none')}")
        print(f"Pattern optimized: {attention_info.get('pattern_optimized', False)}")
        print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


def benchmark_processing_order():
    """Benchmark the impact of optimized processing order."""
    print("\n\nBenchmarking Processing Order Impact")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two models - with and without optimization
    config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,
        block_size=64,
    )

    standard_model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[1024],
        dilation_rates=[4],
        sparse_config=config,
    ).to(device)

    optimized_model = create_post_pattern_hilbert_attention(
        segment_lengths=[1024],
        dilation_rates=[4],
        sparsity_ratio=0.1,
        block_size=64,
    ).to(device)

    # Test inputs
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _ = standard_model(q, k, v)
            _ = optimized_model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    num_runs = 10

    # Standard timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_runs):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out_std = standard_model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    standard_time = (time.perf_counter() - start) / num_runs * 1000

    # Optimized timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_runs):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out_opt = optimized_model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    optimized_time = (time.perf_counter() - start) / num_runs * 1000

    print(f"Standard time: {standard_time:.2f}ms")
    print(f"Optimized time: {optimized_time:.2f}ms")
    print(f"Speedup: {standard_time / optimized_time:.2f}x")

    # Verify outputs are equivalent
    max_diff = (out_std - out_opt).abs().max().item()
    print(f"\nMax output difference: {max_diff}")
    print(f"Outputs match: {torch.allclose(out_std, out_opt, atol=1e-3)}")


def visualize_processing_order():
    """Visualize how the processing order changes."""
    print("\n\nVisualizing Processing Order")
    print("=" * 50)

    model = create_post_pattern_hilbert_attention(
        segment_lengths=[256],
        dilation_rates=[2],
        sparsity_ratio=0.2,
        block_size=32,
    )

    # Get indices
    num_blocks = 8  # Small for visualization
    device = torch.device("cpu")

    # Get standard order
    model.use_post_pattern_optimization = False
    std_indices = model._get_optimized_sparse_block_indices(num_blocks, 1, device)

    # Get optimized order
    model.use_post_pattern_optimization = True
    opt_indices = model._get_optimized_sparse_block_indices(num_blocks, 1, device)

    print("Standard processing order:")
    std_rows, std_cols = std_indices
    if len(std_rows.shape) > 1:
        std_rows, std_cols = std_rows[0], std_cols[0]
    for i in range(min(10, len(std_rows))):
        print(f"  Step {i}: Block ({std_rows[i]}, {std_cols[i]})")

    print("\nOptimized processing order:")
    opt_rows, opt_cols = opt_indices
    if len(opt_rows.shape) > 1:
        opt_rows, opt_cols = opt_rows[0], opt_cols[0]
    for i in range(min(10, len(opt_rows))):
        print(f"  Step {i}: Block ({opt_rows[i]}, {opt_cols[i]})")


if __name__ == "__main__":
    test_pattern_preservation()
    test_optimizer_logic()
    test_forward_pass()
    benchmark_processing_order()
    visualize_processing_order()
