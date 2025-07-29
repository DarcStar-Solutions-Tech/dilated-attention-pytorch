#!/usr/bin/env python3
"""Debug what the Hilbert kernel is actually doing."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def analyze_kernel_access_pattern():
    """Analyze how the kernel accesses memory."""

    # Simple case for analysis
    seq_len = 16
    segment_size = 8
    dilation_rate = 2

    # Create Hilbert mapping
    hilbert_map = UnifiedHilbertAttention._create_hilbert_mapping(seq_len)
    print(f"Hilbert mapping for seq_len={seq_len}:")
    print(f"  {hilbert_map.tolist()}")

    # Simulate what the kernel does
    print("\nKernel access pattern:")

    # For each segment
    for seg_idx in range(2):  # 2 segments
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size

        print(f"\nSegment {seg_idx} (positions {seg_start}-{seg_end}):")

        # The kernel iterates through all positions
        for pos in range(seg_start, seg_end):
            # Check dilation
            if (pos - seg_start) % dilation_rate == 0:
                # This position passes the dilation check
                h_idx = hilbert_map[pos].item()
                print(f"  Position {pos} -> Hilbert index {h_idx} (USED)")
            else:
                print(f"  Position {pos} -> skipped by dilation")

    # Show the actual access order
    print("\nActual memory access order for sparse attention:")
    access_order = []
    for seg_idx in range(2):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size
        for pos in range(seg_start, seg_end):
            if (pos - seg_start) % dilation_rate == 0:
                h_idx = hilbert_map[pos].item()
                access_order.append(h_idx)
    print(f"  {access_order}")

    # Compare to ideal sparse Hilbert
    print("\nIdeal sparse Hilbert access (what we want):")
    # For sparse, we should first get sparse positions, then apply Hilbert
    sparse_positions = [0, 2, 4, 6, 8, 10, 12, 14]  # Every 2nd position
    # Create Hilbert for just these positions
    sparse_hilbert = UnifiedHilbertAttention._create_hilbert_mapping(
        len(sparse_positions)
    )
    ideal_access = []
    for i in range(len(sparse_positions)):
        ideal_access.append(sparse_positions[sparse_hilbert[i].item()])
    print(f"  {ideal_access}")


def test_small_example():
    """Test with a small example to see the issue."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Very small example
    batch_size = 1
    seq_len = 256
    hidden_dim = 64
    num_heads = 4
    segment_size = 64
    dilation_rate = 4

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        dropout=0.0,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Get outputs
    with torch.no_grad():
        out_standard = module(x, use_hilbert=False)
        out_hilbert = module(x, use_hilbert=True)

    # Check difference
    diff = torch.norm(out_hilbert - out_standard) / torch.norm(out_standard)
    print(f"\nRelative difference: {diff:.6f}")

    # Analyze memory access pattern for one segment
    hilbert_map = module._create_hilbert_mapping(256)

    print("\nMemory access analysis for first segment:")
    access_positions = []
    for pos in range(0, segment_size, dilation_rate):
        h_idx = hilbert_map[pos].item()
        access_positions.append(h_idx)

    print(f"  Sparse positions: {list(range(0, segment_size, dilation_rate))}")
    print(f"  Hilbert indices: {access_positions}")

    # Calculate jump distances
    if len(access_positions) > 1:
        jumps = [
            abs(access_positions[i + 1] - access_positions[i])
            for i in range(len(access_positions) - 1)
        ]
        avg_jump = sum(jumps) / len(jumps)
        print(f"  Average jump distance: {avg_jump:.1f}")
        print(f"  Max jump: {max(jumps)}")


if __name__ == "__main__":
    analyze_kernel_access_pattern()
    print("\n" + "=" * 60 + "\n")
    test_small_example()
