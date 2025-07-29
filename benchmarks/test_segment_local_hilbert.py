#!/usr/bin/env python3
"""Test segment-local Hilbert mapping."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def test_segment_local_mapping():
    """Analyze segment-local Hilbert mapping."""

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768, num_heads=12, segment_size=64, dilation_rate=4, dropout=0.0
    )

    # Test with a small sequence
    seq_len = 256

    # Get the mapping
    device = "cpu"  # Use CPU for analysis
    mapping = module._get_hilbert_mapping(seq_len, torch.device(device))

    print(
        f"Segment-local Hilbert mapping for seq_len={seq_len}, segment_size=64, dilation_rate=4:"
    )

    # Analyze first segment
    print("\nFirst segment (0-64):")
    sparse_positions = list(range(0, 64, 4))  # Every 4th position
    print(f"  Sparse positions: {sparse_positions}")

    mapped_positions = []
    for pos in sparse_positions:
        mapped_positions.append(mapping[pos].item())
    print(f"  Mapped positions: {mapped_positions}")

    # Calculate jumps
    if len(mapped_positions) > 1:
        jumps = []
        for i in range(1, len(mapped_positions)):
            jump = abs(mapped_positions[i] - mapped_positions[i - 1])
            jumps.append(jump)
        avg_jump = sum(jumps) / len(jumps)
        print(f"  Average jump: {avg_jump:.1f}")
        print(f"  Max jump: {max(jumps)}")

    # Compare to sequential
    seq_jumps = []
    for i in range(1, len(sparse_positions)):
        jump = sparse_positions[i] - sparse_positions[i - 1]
        seq_jumps.append(jump)
    print(f"  Sequential jump: {seq_jumps[0] if seq_jumps else 0}")

    # Test correctness - verify it's a permutation
    print("\nMapping validation:")
    unique_values = torch.unique(mapping)
    print(f"  Unique values: {len(unique_values)}")
    print(f"  Is valid permutation: {len(unique_values) == seq_len}")


if __name__ == "__main__":
    test_segment_local_mapping()
