#!/usr/bin/env python3
"""
Test dilation-aware Hilbert ordering implementation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_dilation_aware import (
    DilationAwareHilbertOrdering,
    create_dilation_aware_hilbert_attention,
)


def test_dilation_access_groups():
    """Test that access groups are created correctly for different dilation rates."""
    ordering = DilationAwareHilbertOrdering()

    print("Testing Dilation Access Groups")
    print("=" * 50)

    # Test with different configurations
    test_cases = [
        (16, 1, "No dilation - should create local groups"),
        (16, 2, "Dilation 2 - blocks 0,2,4,6... grouped together"),
        (16, 4, "Dilation 4 - blocks 0,4,8,12... grouped together"),
        (20, 5, "Dilation 5 - non-power-of-2"),
    ]

    for num_blocks, dilation_rate, description in test_cases:
        print(f"\n{description}")
        print(f"Num blocks: {num_blocks}, Dilation rate: {dilation_rate}")

        groups = ordering.get_dilation_access_groups(num_blocks, dilation_rate)

        print(f"Number of groups: {len(groups)}")
        for i, group in enumerate(groups[:5]):  # Show first 5 groups
            print(f"  Group {i}: {group}")
        if len(groups) > 5:
            print(f"  ... and {len(groups) - 5} more groups")

    print("\n" + "=" * 50)


def test_hilbert_within_groups():
    """Test that Hilbert ordering is applied correctly within groups."""
    ordering = DilationAwareHilbertOrdering()

    print("\nTesting Hilbert Ordering Within Groups")
    print("=" * 50)

    num_blocks = 16
    dilation_rate = 4

    # Create a simple block pattern
    # Diagonal blocks + some dilated connections
    row_indices = []
    col_indices = []

    # Add diagonal
    for i in range(num_blocks):
        row_indices.append(i)
        col_indices.append(i)

    # Add dilated connections
    for i in range(0, num_blocks - dilation_rate, dilation_rate):
        for j in range(
            i + dilation_rate, min(i + 3 * dilation_rate, num_blocks), dilation_rate
        ):
            row_indices.append(i)
            col_indices.append(j)
            row_indices.append(j)
            col_indices.append(i)

    row_tensor = torch.tensor(row_indices)
    col_tensor = torch.tensor(col_indices)

    print(f"Original pattern has {len(row_indices)} connections")
    print(f"First 10 connections: {list(zip(row_indices[:10], col_indices[:10]))}")

    # Get access groups
    groups = ordering.get_dilation_access_groups(num_blocks, dilation_rate)
    print(f"\nAccess groups: {groups[:3]}...")

    # Apply Hilbert within groups
    reordered_row, reordered_col = ordering.apply_hilbert_within_groups(
        (row_tensor, col_tensor), groups, num_blocks
    )

    print("\nReordered pattern:")
    print(
        f"First 10 connections: {list(zip(reordered_row[:10].tolist(), reordered_col[:10].tolist()))}"
    )

    # Verify all connections are preserved
    original_set = set(zip(row_indices, col_indices))
    reordered_set = set(zip(reordered_row.tolist(), reordered_col.tolist()))

    print("\nVerification:")
    print(f"Original connections: {len(original_set)}")
    print(f"Reordered connections: {len(reordered_set)}")
    print(f"All connections preserved: {original_set == reordered_set}")


def test_forward_pass():
    """Test the forward pass with dilation-aware Hilbert."""
    print("\n\nTesting Forward Pass")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_dilation_aware_hilbert_attention(
        segment_lengths=[1024],
        dilation_rates=[4],
        sparsity_ratio=0.1,
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

    print(f"Input shape: {q.shape}")
    print(f"Device: {device}")

    # Forward pass
    try:
        output, attention_info = model(q, k, v, return_attention_weights=True)
        print(f"Output shape: {output.shape}")
        print(f"Optimization: {attention_info.get('optimization', 'none')}")
        print(f"Dilation rate: {attention_info.get('dilation_rate', 'unknown')}")
        print(
            f"Number of access groups: {attention_info.get('access_groups', 'unknown')}"
        )
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


def test_visualization():
    """Test pattern visualization to understand the optimization."""
    print("\n\nTesting Pattern Visualization")
    print("=" * 50)

    # Create model with specific dilation
    model = create_dilation_aware_hilbert_attention(
        segment_lengths=[512],
        dilation_rates=[4],
        sparsity_ratio=0.2,
        block_size=64,
    )

    # Visualize for a small sequence
    viz = model.visualize_access_pattern(512)

    print(f"Standard pattern shape: {viz['standard_pattern'].shape}")
    print(f"Optimized pattern shape: {viz['optimized_pattern'].shape}")
    print(f"Number of access groups: {viz['num_groups']}")

    # Show pattern density
    standard_density = (
        viz["standard_pattern"].sum().item() / viz["standard_pattern"].numel()
    )
    optimized_density = (
        viz["optimized_pattern"].sum().item() / viz["optimized_pattern"].numel()
    )

    print("\nPattern density:")
    print(f"Standard: {standard_density:.4f}")
    print(f"Optimized: {optimized_density:.4f}")
    print(
        f"Patterns identical: {torch.allclose(viz['standard_pattern'], viz['optimized_pattern'])}"
    )

    # Show which blocks are in the same access group
    print(f"\nAccess group visualization shape: {viz['access_groups'].shape}")
    print(f"Unique groups: {viz['access_groups'].unique().tolist()}")


def compare_with_standard():
    """Compare outputs with standard implementation."""
    print("\n\nComparing with Standard Implementation")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create both models
    from dilated_attention_pytorch import (
        create_block_sparse_attention,
        SparsePatternConfig,
    )

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

    dilation_aware_model = create_dilation_aware_hilbert_attention(
        segment_lengths=[1024],
        dilation_rates=[4],
        sparsity_ratio=0.1,
        block_size=64,
    ).to(device)

    # Create inputs
    batch_size = 1
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Forward passes
    with torch.no_grad():
        output_standard = standard_model(q, k, v)
        output_dilation_aware = dilation_aware_model(q, k, v)

    # Compare outputs
    max_diff = (output_standard - output_dilation_aware).abs().max().item()
    mean_diff = (output_standard - output_dilation_aware).abs().mean().item()

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(
        f"Outputs match (within tolerance): {torch.allclose(output_standard, output_dilation_aware, atol=1e-5)}"
    )


if __name__ == "__main__":
    test_dilation_access_groups()
    test_hilbert_within_groups()
    test_forward_pass()
    test_visualization()
    compare_with_standard()
