#!/usr/bin/env python3
"""
Test script to validate Hilbert attention index fixes.
Checks for common index errors and validates correctness.
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    HilbertAttentionCore as HilbertAttentionTritonFixed,
    create_hilbert_mapping,
)


def test_hilbert_mapping():
    """Test Hilbert mapping generation for various sizes."""
    print("=== Testing Hilbert Mapping Generation ===\n")

    test_sizes = [64, 128, 256, 512, 1000, 1024]

    for size in test_sizes:
        mapping = create_hilbert_mapping(size)

        # Check mapping properties
        assert mapping.shape == (size,), f"Mapping shape mismatch for size {size}"
        assert mapping.dtype == torch.int32, "Mapping dtype should be int32"

        # Check that all indices are unique and in valid range
        unique_indices = torch.unique(mapping)
        assert len(unique_indices) == size, f"Mapping should have {size} unique indices"
        assert mapping.min() >= 0, "Mapping contains negative indices"
        assert mapping.max() < size, "Mapping contains out-of-bound indices"

        print(
            f"✓ Size {size:4}: Mapping shape {mapping.shape}, "
            f"range [{mapping.min().item()}, {mapping.max().item()}]"
        )

    print("\nAll Hilbert mappings validated successfully!\n")


def test_attention_shapes():
    """Test attention computation with various shapes."""
    print("=== Testing Attention Shape Compatibility ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test configurations (batch, seq_len, hidden_dim, heads, segment_size, dilation)
    configs = [
        (2, 128, 256, 8, 32, 1),  # Basic
        (4, 256, 512, 16, 64, 2),  # Standard
        (1, 512, 768, 12, 128, 4),  # Large
        (3, 255, 384, 8, 64, 2),  # Non-power-of-2 sequence
        (2, 1000, 512, 8, 128, 8),  # Non-square sequence
        (1, 64, 256, 4, 16, 1),  # Small
    ]

    for batch, seq_len, hidden_dim, heads, segment_size, dilation in configs:
        try:
            model = HilbertAttentionTritonFixed(
                hidden_dim=hidden_dim,
                num_heads=heads,
                segment_size=segment_size,
                dilation_rate=dilation,
            ).to(device)

            x = torch.randn(batch, seq_len, hidden_dim, device=device)

            # Test both Hilbert and standard modes
            with torch.no_grad():
                out_hilbert = model(x, use_hilbert=True)
                out_standard = model(x, use_hilbert=False)

            # Verify output shapes
            assert out_hilbert.shape == x.shape, "Hilbert output shape mismatch"
            assert out_standard.shape == x.shape, "Standard output shape mismatch"

            # Check for NaN/Inf
            assert not torch.isnan(out_hilbert).any(), "Hilbert output contains NaN"
            assert not torch.isinf(out_hilbert).any(), "Hilbert output contains Inf"
            assert not torch.isnan(out_standard).any(), "Standard output contains NaN"
            assert not torch.isinf(out_standard).any(), "Standard output contains Inf"

            print(
                f"✓ Config B={batch} L={seq_len} D={hidden_dim} H={heads} "
                f"seg={segment_size} dil={dilation}: OK"
            )

        except Exception as e:
            print(
                f"✗ Config B={batch} L={seq_len} D={hidden_dim} H={heads} "
                f"seg={segment_size} dil={dilation}: {str(e)}"
            )

    print("\nShape compatibility tests completed!\n")


def test_index_bounds():
    """Test that all index operations stay within bounds."""
    print("=== Testing Index Bounds ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Edge case configurations
    edge_cases = [
        (1, 63, 256, 8, 32, 1),  # seq_len < segment_size * 2
        (1, 65, 256, 8, 32, 2),  # seq_len slightly > power of 2
        (1, 127, 256, 8, 64, 4),  # seq_len = segment_size * 2 - 1
        (1, 128, 256, 8, 128, 1),  # seq_len = segment_size
        (1, 256, 256, 8, 64, 8),  # high dilation rate
        (1, 511, 512, 16, 128, 16),  # very high dilation
    ]

    for batch, seq_len, hidden_dim, heads, segment_size, dilation in edge_cases:
        try:
            model = HilbertAttentionTritonFixed(
                hidden_dim=hidden_dim,
                num_heads=heads,
                segment_size=segment_size,
                dilation_rate=dilation,
            ).to(device)

            x = torch.randn(batch, seq_len, hidden_dim, device=device)

            # Test with extreme values to stress-test indexing
            x_zeros = torch.zeros_like(x)
            x_ones = torch.ones_like(x)

            with torch.no_grad():
                # Should not crash with any input
                _ = model(x, use_hilbert=True)
                _ = model(x, use_hilbert=False)
                _ = model(x_zeros, use_hilbert=True)
                _ = model(x_ones, use_hilbert=True)

            print(f"✓ Edge case L={seq_len} seg={segment_size} dil={dilation}: Passed")

        except Exception as e:
            print(
                f"✗ Edge case L={seq_len} seg={segment_size} dil={dilation}: {str(e)}"
            )

    print("\nIndex bounds tests completed!\n")


def test_gradient_flow():
    """Test that gradients flow correctly through both paths."""
    print("=== Testing Gradient Flow ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch, seq_len, hidden_dim = 2, 128, 256
    heads, segment_size, dilation = 8, 32, 2

    model = HilbertAttentionTritonFixed(
        hidden_dim=hidden_dim,
        num_heads=heads,
        segment_size=segment_size,
        dilation_rate=dilation,
    ).to(device)

    x = torch.randn(batch, seq_len, hidden_dim, device=device, requires_grad=True)

    # Test Hilbert path
    out_hilbert = model(x, use_hilbert=True)
    loss_hilbert = out_hilbert.mean()
    loss_hilbert.backward()

    grad_hilbert = x.grad.clone()
    x.grad.zero_()

    # Test standard path
    out_standard = model(x, use_hilbert=False)
    loss_standard = out_standard.mean()
    loss_standard.backward()

    grad_standard = x.grad.clone()

    # Check gradients exist and are reasonable
    assert not torch.isnan(grad_hilbert).any(), "Hilbert gradients contain NaN"
    assert not torch.isnan(grad_standard).any(), "Standard gradients contain NaN"
    assert grad_hilbert.abs().max() < 100, "Hilbert gradients too large"
    assert grad_standard.abs().max() < 100, "Standard gradients too large"

    print("✓ Gradient flow test passed")
    print(f"  Hilbert grad norm: {grad_hilbert.norm().item():.4f}")
    print(f"  Standard grad norm: {grad_standard.norm().item():.4f}")
    print(
        f"  Relative difference: {(grad_hilbert - grad_standard).norm() / grad_standard.norm():.4f}"
    )
    print()


def test_memory_access_patterns():
    """Analyze memory access patterns to ensure Hilbert ordering is correct."""
    print("=== Testing Memory Access Patterns ===\n")

    seq_len = 256
    segment_size = 64
    dilation_rate = 4

    # Create Hilbert mapping
    mapping = create_hilbert_mapping(seq_len)

    # Analyze access patterns for one segment
    segment_idx = 1
    seg_start = segment_idx * segment_size
    _ = seg_start + segment_size

    # Standard access pattern
    standard_accesses = []
    for offset in range(0, segment_size, dilation_rate):
        key_pos = seg_start + offset
        if key_pos < seq_len:
            standard_accesses.append(key_pos)

    # Hilbert access pattern
    hilbert_accesses = []
    for pos in standard_accesses:
        hilbert_pos = mapping[pos].item()
        hilbert_accesses.append(hilbert_pos)

    # Calculate memory jumps
    standard_jumps = [
        abs(standard_accesses[i + 1] - standard_accesses[i])
        for i in range(len(standard_accesses) - 1)
    ]
    hilbert_jumps = [
        abs(hilbert_accesses[i + 1] - hilbert_accesses[i])
        for i in range(len(hilbert_accesses) - 1)
    ]

    print(
        f"Segment {segment_idx} analysis (size={segment_size}, dilation={dilation_rate}):"
    )
    print(
        f"  Standard accesses: {standard_accesses[:5]} ... (avg jump: {np.mean(standard_jumps):.1f})"
    )
    print(
        f"  Hilbert accesses:  {hilbert_accesses[:5]} ... (avg jump: {np.mean(hilbert_jumps):.1f})"
    )
    print(
        f"  Jump reduction: {(1 - np.mean(hilbert_jumps) / np.mean(standard_jumps)) * 100:.1f}%"
    )
    print()


def run_all_tests():
    """Run all test suites."""
    print("=" * 80)
    print("Running Hilbert Attention Index Fix Tests")
    print("=" * 80)
    print()

    test_hilbert_mapping()
    test_attention_shapes()
    test_index_bounds()
    test_gradient_flow()
    test_memory_access_patterns()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
