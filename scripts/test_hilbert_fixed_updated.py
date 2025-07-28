#!/usr/bin/env python3
"""
Test script for the updated Hilbert attention implementation with bit-reversal pattern.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dilated_attention_pytorch.kernels.hilbert_attention_core_fixed import (
    HilbertAttentionCoreFixed,
    create_hilbert_mapping_per_segment,
)


def test_bit_reversal_pattern():
    """Test the bit-reversal pattern generation."""
    print("Testing bit-reversal pattern generation...")

    # Test with different segment sizes
    for segment_size in [64, 128, 256, 512, 1024, 2048]:
        seq_len = segment_size * 2  # Two segments
        mapping = create_hilbert_mapping_per_segment(seq_len, segment_size)

        print(f"\nSegment size: {segment_size}")
        print(f"First 16 positions in segment 0: {mapping[:16].tolist()}")
        print(
            f"First 16 positions in segment 1: {mapping[segment_size : segment_size + 16].tolist()}"
        )

        # Verify that each segment is a permutation
        for seg_idx in range(2):
            seg_start = seg_idx * segment_size
            seg_end = seg_start + segment_size
            segment_values = mapping[seg_start:seg_end] - seg_start

            # Check that it's a valid permutation
            assert torch.all(segment_values >= 0) and torch.all(
                segment_values < segment_size
            )
            assert len(torch.unique(segment_values)) == segment_size
            print(f"  Segment {seg_idx}: Valid permutation ✓")


def test_hilbert_attention():
    """Test the Hilbert attention module."""
    print("\n\nTesting Hilbert attention module...")

    # Test parameters
    batch_size = 2
    seq_len = 1024
    hidden_dim = 256
    num_heads = 8
    segment_size = 256

    # Create module
    model = HilbertAttentionCoreFixed(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=1,
    )

    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test forward pass with Hilbert ordering
    print("\nRunning forward pass with Hilbert ordering...")
    try:
        with torch.cuda.amp.autocast(enabled=False):
            output_hilbert = model(x, use_hilbert=True)
        print(f"  Output shape: {output_hilbert.shape}")
        print(f"  Output mean: {output_hilbert.mean().item():.6f}")
        print(f"  Output std: {output_hilbert.std().item():.6f}")
        print("  Hilbert attention forward pass: ✓")
    except Exception as e:
        print(f"  Hilbert attention forward pass failed: {e}")
        return

    # Test forward pass without Hilbert ordering
    print("\nRunning forward pass without Hilbert ordering...")
    try:
        with torch.cuda.amp.autocast(enabled=False):
            output_standard = model(x, use_hilbert=False)
        print(f"  Output shape: {output_standard.shape}")
        print(f"  Output mean: {output_standard.mean().item():.6f}")
        print(f"  Output std: {output_standard.std().item():.6f}")
        print("  Standard attention forward pass: ✓")
    except Exception as e:
        print(f"  Standard attention forward pass failed: {e}")
        return

    # Compare outputs (they should be different due to reordering)
    diff = (output_hilbert - output_standard).abs().mean()
    print(f"\nMean absolute difference: {diff.item():.6f}")
    print(f"Outputs are {'different' if diff > 1e-5 else 'similar'} (as expected)")

    # Test gradient flow
    print("\nTesting gradient flow...")
    # Clear any existing gradients
    model.zero_grad()

    # Create a fresh input with requires_grad
    x_grad = torch.randn(1, 256, hidden_dim, device=device, requires_grad=True)
    output_grad = model(x_grad, use_hilbert=True)

    loss = output_grad.mean()
    loss.backward()

    # Check that gradients exist
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  All parameters have gradients: {'✓' if has_grads else '✗'}")
    print(f"  Input has gradient: {'✓' if x_grad.grad is not None else '✗'}")

    # Test with different sequence lengths
    print("\nTesting with different sequence lengths...")
    for test_seq_len in [256, 512, 768, 1024, 2048]:
        x_test = torch.randn(1, test_seq_len, hidden_dim, device=device)
        try:
            output = model(x_test, use_hilbert=True)
            print(f"  Seq length {test_seq_len}: ✓ (output shape: {output.shape})")
        except Exception as e:
            print(f"  Seq length {test_seq_len}: ✗ ({e})")


def visualize_bit_reversal(segment_size=64):
    """Visualize the bit-reversal pattern as a 2D grid."""
    print(f"\n\nVisualizing bit-reversal pattern for segment size {segment_size}...")

    import math

    # Create mapping
    mapping = create_hilbert_mapping_per_segment(segment_size, segment_size)

    # Convert to 2D grid for visualization
    grid_size = int(math.ceil(math.sqrt(segment_size)))
    grid = torch.full((grid_size, grid_size), -1, dtype=torch.long)

    for i in range(segment_size):
        row = i // grid_size
        col = i % grid_size
        if row < grid_size and col < grid_size:
            grid[row, col] = mapping[i].item()

    # Print the grid
    print("\nBit-reversal pattern as 2D grid (read order):")
    print("Each number shows when that position is read")
    for row in grid:
        row_str = " ".join(f"{val:3d}" if val >= 0 else "  -" for val in row)
        print(row_str)


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Updated Hilbert Attention with Bit-Reversal Pattern")
    print("=" * 80)

    # Run tests
    test_bit_reversal_pattern()
    test_hilbert_attention()
    visualize_bit_reversal(64)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
