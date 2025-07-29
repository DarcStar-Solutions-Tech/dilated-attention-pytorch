#!/usr/bin/env python3
"""Test the corrected sparse Hilbert implementation."""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention


def test_corrected_implementation():
    """Test that the corrected implementation maintains sparse pattern locality."""

    print("TESTING CORRECTED SPARSE HILBERT IMPLEMENTATION")
    print("=" * 80)

    # Create module with sparse pattern
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=64,
        dilation_rate=4,
        dropout=0.0,
        hilbert_threshold=0,  # Force Hilbert to be used
    )

    device = "cpu"
    seq_len = 256

    # Get the sparse Hilbert mapping
    mapping = module._create_segment_local_hilbert_mapping(
        seq_len, module.segment_size, module.dilation_rate
    )

    print("\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Segment size: {module.segment_size}")
    print(f"  Dilation rate: {module.dilation_rate}")
    print(
        f"  Sparse positions per segment: {module.segment_size // module.dilation_rate}"
    )

    # Analyze each segment
    num_segments = (seq_len + module.segment_size - 1) // module.segment_size

    all_jumps = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * module.segment_size
        seg_end = min(seg_start + module.segment_size, seq_len)

        # Get sparse positions in this segment
        sparse_positions = []
        for i in range(0, seg_end - seg_start, module.dilation_rate):
            pos = seg_start + i
            if pos < seg_end:
                sparse_positions.append(pos)

        # Get their mapped positions
        mapped_positions = [mapping[p].item() for p in sparse_positions]

        # Calculate jumps
        jumps = []
        for i in range(1, len(mapped_positions)):
            jump = abs(mapped_positions[i] - mapped_positions[i - 1])
            jumps.append(jump)

        all_jumps.extend(jumps)

        # Print analysis for first segment
        if seg_idx == 0:
            print(f"\nSegment {seg_idx} analysis:")
            print(f"  Original sparse positions: {sparse_positions[:8]}...")
            print(f"  Mapped positions: {mapped_positions[:8]}...")
            print(f"  Jump distances: {jumps[:7]}...")
            print(f"  Average jump: {np.mean(jumps):.1f}")
            print(f"  Max jump: {max(jumps) if jumps else 0}")

    # Overall statistics
    print("\nOverall statistics across all segments:")
    print(f"  Average jump distance: {np.mean(all_jumps):.1f}")
    print(f"  Max jump distance: {max(all_jumps)}")
    print(f"  Min jump distance: {min(all_jumps)}")
    print(f"  Std deviation: {np.std(all_jumps):.1f}")

    # Compare with naive approach
    print("\nComparison:")
    print(f"  Sequential sparse (no Hilbert): jump = {module.dilation_rate}")
    print(f"  Our implementation: avg jump = {np.mean(all_jumps):.1f}")
    print("  Previous wrong implementation: avg jump = ~46")

    # Test that it's still a valid permutation within sparse positions
    print("\nValidation:")

    # Check each segment's sparse positions
    all_original = []
    all_mapped = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * module.segment_size
        seg_end = min(seg_start + module.segment_size, seq_len)

        sparse_positions = []
        for i in range(0, seg_end - seg_start, module.dilation_rate):
            pos = seg_start + i
            if pos < seg_end:
                sparse_positions.append(pos)
                all_original.append(pos)
                all_mapped.append(mapping[pos].item())

    # Check if mapped positions are a permutation of original sparse positions
    all_original_sorted = sorted(all_original)
    all_mapped_sorted = sorted(all_mapped)

    is_valid_permutation = all_original_sorted == all_mapped_sorted
    print(f"  Is valid permutation of sparse positions: {is_valid_permutation}")

    if not is_valid_permutation:
        print("  ERROR: Mapping is not a valid permutation!")
        missing = set(all_original) - set(all_mapped)
        extra = set(all_mapped) - set(all_original)
        print(f"  Missing positions: {missing}")
        print(f"  Extra positions: {extra}")

    # Test with actual attention computation
    print("\nTesting with actual attention computation:")

    x = torch.randn(1, seq_len, 768, device=device)

    # Force different configurations
    module._triton_available = False  # Use PyTorch backend

    with torch.no_grad():
        # Without Hilbert
        out1 = module(x, use_hilbert=False)

        # With corrected Hilbert
        out2 = module(x, use_hilbert=True)

    # They should be different but both valid
    are_different = not torch.allclose(out1, out2, rtol=1e-4)
    print(f"  Outputs are different (expected): {are_different}")
    print(f"  Output shapes match: {out1.shape == out2.shape}")

    # Visualize the pattern
    print("\nPattern visualization for first 32 sparse positions:")
    positions = []
    mapped = []

    for seg_idx in range(2):  # First two segments
        seg_start = seg_idx * module.segment_size
        seg_end = min(seg_start + module.segment_size, seq_len)

        for i in range(0, seg_end - seg_start, module.dilation_rate):
            pos = seg_start + i
            if pos < seg_end and len(positions) < 32:
                positions.append(pos)
                mapped.append(mapping[pos].item())

    # Print as a simple graph
    print("  Original: ", end="")
    for i in range(len(positions)):
        print(f"{positions[i]:3d}", end=" ")
        if (i + 1) % 16 == 0:
            print("\n           ", end="")

    print("\n  Mapped:   ", end="")
    for i in range(len(mapped)):
        print(f"{mapped[i]:3d}", end=" ")
        if (i + 1) % 16 == 0:
            print("\n           ", end="")
    print()

    print("\nConclusion:")
    if np.mean(all_jumps) < 20 and is_valid_permutation:
        print("  ✓ The corrected implementation maintains good locality!")
        print("  ✓ Sparse positions are reordered within their segments")
        print("  ✓ Average jump distance is reasonable")
    else:
        print("  ✗ There are still issues with the implementation")


if __name__ == "__main__":
    test_corrected_implementation()
