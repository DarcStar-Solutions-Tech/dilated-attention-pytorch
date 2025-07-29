#!/usr/bin/env python3
"""Test correctness of Hilbert attention."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def test_correctness():
    """Verify Hilbert attention produces correct results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple test case
    batch_size = 1
    seq_len = 64
    hidden_dim = 64
    num_heads = 4

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=32,
        dilation_rate=1,  # Start with standard attention
        dropout=0.0,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test standard vs Hilbert for non-dilated
    with torch.no_grad():
        out_standard = module(x, use_hilbert=False)
        out_hilbert = module(x, use_hilbert=True)

    diff = torch.norm(out_standard - out_hilbert) / torch.norm(out_standard)
    print("Standard attention (dilation=1):")
    print(f"  Relative difference: {diff:.6f}")
    print("  Should be non-zero due to reordering")

    # Test with dilation
    module.dilation_rate = 4
    with torch.no_grad():
        out_standard_dilated = module(x, use_hilbert=False)
        out_hilbert_dilated = module(x, use_hilbert=True)

    diff_dilated = torch.norm(out_standard_dilated - out_hilbert_dilated) / torch.norm(
        out_standard_dilated
    )
    print("\nDilated attention (dilation=4):")
    print(f"  Relative difference: {diff_dilated:.6f}")

    # Test if Hilbert mapping is valid
    mapping = module._create_hilbert_mapping(64)
    print("\nHilbert mapping validation:")
    print(f"  Min: {mapping.min().item()}, Max: {mapping.max().item()}")
    print(f"  Unique values: {len(torch.unique(mapping))}")
    print(f"  Is permutation: {len(torch.unique(mapping)) == 64}")

    # Show first few mappings
    print("\nFirst 16 position mappings:")
    for i in range(16):
        print(f"  Position {i} -> Hilbert position {mapping[i].item()}")


if __name__ == "__main__":
    test_correctness()
