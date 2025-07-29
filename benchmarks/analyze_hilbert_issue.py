#!/usr/bin/env python3
"""Analyze why Hilbert performance is poor."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def analyze_access_patterns():
    """Analyze memory access patterns with and without Hilbert."""

    # Create a simple example
    seq_len = 256
    segment_size = 64
    dilation_rate = 4

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        dropout=0.0,
    )

    # Get Hilbert mapping
    hilbert_map = module._create_hilbert_mapping(seq_len)

    print(f"Sequence length: {seq_len}")
    print(f"Segment size: {segment_size}")
    print(f"Dilation rate: {dilation_rate}")
    print()

    # Analyze each segment
    num_segments = (seq_len + segment_size - 1) // segment_size

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)
        seg_len = seg_end - seg_start

        # Calculate sparse positions
        num_sparse = (seg_len + dilation_rate - 1) // dilation_rate
        sparse_indices = torch.arange(
            seg_start,
            min(seg_start + num_sparse * dilation_rate, seq_len),
            dilation_rate,
        )

        # Get Hilbert-reordered positions
        hilbert_positions = hilbert_map[sparse_indices]

        print(f"\nSegment {seg_idx} (positions {seg_start}-{seg_end}):")
        print(f"  Sparse indices: {sparse_indices.tolist()}")
        print(f"  Hilbert positions: {hilbert_positions.tolist()}")

        # Calculate locality score (how close are consecutive accesses)
        if len(hilbert_positions) > 1:
            diffs = torch.diff(hilbert_positions).abs()
            avg_jump = diffs.float().mean().item()
            max_jump = diffs.max().item()
            print(f"  Average jump distance: {avg_jump:.1f}")
            print(f"  Max jump distance: {max_jump}")

            # Compare to sequential access
            seq_diffs = torch.diff(sparse_indices).abs()
            seq_avg = seq_diffs.float().mean().item()
            print(f"  Sequential avg jump: {seq_avg:.1f}")
            print(f"  Locality degradation: {avg_jump / seq_avg:.2f}x")


def test_per_segment_hilbert():
    """Test if per-segment Hilbert would be better."""

    segment_size = 64
    dilation_rate = 4

    print("\nComparing global vs per-segment Hilbert:")
    print("=" * 50)

    # Global Hilbert
    global_map = UnifiedHilbertAttention._create_hilbert_mapping(256)
    sparse_indices = torch.arange(0, segment_size, dilation_rate)
    global_positions = global_map[sparse_indices]

    print(f"Global Hilbert positions: {global_positions.tolist()}")

    # Per-segment Hilbert
    segment_map = UnifiedHilbertAttention._create_hilbert_mapping(segment_size)
    segment_positions = segment_map[sparse_indices]

    print(f"Per-segment Hilbert positions: {segment_positions.tolist()}")

    # Compare locality
    global_jumps = torch.diff(global_positions).abs().float().mean()
    segment_jumps = torch.diff(segment_positions).abs().float().mean()

    print("\nAverage jump distance:")
    print(f"  Global Hilbert: {global_jumps:.1f}")
    print(f"  Per-segment Hilbert: {segment_jumps:.1f}")
    print(f"  Improvement: {global_jumps / segment_jumps:.2f}x")


if __name__ == "__main__":
    analyze_access_patterns()
    test_per_segment_hilbert()
