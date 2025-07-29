#!/usr/bin/env python3
"""Test Hilbert with 256 sequence length."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def main():
    """Test with seq_len > 64 to see actual Hilbert mapping."""

    # Create mapping for 256
    mapping = UnifiedHilbertAttention._create_hilbert_mapping(256)

    print("Hilbert mapping for seq_len=256:")
    print("First 32 mappings:")
    for i in range(32):
        print(f"  {i} -> {mapping[i].item()}")

    # Check if it's actually doing Hilbert curve
    print("\nChecking Hilbert properties:")

    # In a proper Hilbert curve, adjacent positions in Hilbert space
    # should be spatially close in 2D grid
    grid_size = 16  # 256 = 16x16

    # Find positions 0,1,2,3 in Hilbert order
    hilbert_positions = []
    for h_pos in range(4):
        # Find which original position maps to this Hilbert position
        orig_pos = torch.where(mapping == h_pos)[0][0].item()
        x = orig_pos % grid_size
        y = orig_pos // grid_size
        hilbert_positions.append((x, y))
        print(f"  Hilbert pos {h_pos}: original pos {orig_pos} -> grid ({x}, {y})")

    # Check spatial locality
    for i in range(1, 4):
        dx = abs(hilbert_positions[i][0] - hilbert_positions[i - 1][0])
        dy = abs(hilbert_positions[i][1] - hilbert_positions[i - 1][1])
        dist = dx + dy
        print(f"  Distance {i - 1}->{i}: {dist} (dx={dx}, dy={dy})")


if __name__ == "__main__":
    main()
