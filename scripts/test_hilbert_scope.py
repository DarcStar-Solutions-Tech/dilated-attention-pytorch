#!/usr/bin/env python3
"""
Test to determine if Hilbert reordering is applied to the entire sequence
or just the dilated attention segments.
"""

import torch

# Import the kernel
from dilated_attention_pytorch.kernels import HilbertAttentionCore


def analyze_hilbert_scope():
    """Analyze the scope of Hilbert reordering."""

    # Configuration
    hidden_dim = 768
    num_heads = 12
    batch_size = 1
    seq_len = 1024
    segment_size = 256
    dilation_rate = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module
    module = HilbertAttentionCore(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    # Check the Hilbert mapping size
    hilbert_map = module.get_hilbert_mapping(seq_len, torch.device(device))
    print(f"Sequence length: {seq_len}")
    print(f"Hilbert map size: {hilbert_map.shape[0]}")
    print(f"Hilbert map dtype: {hilbert_map.dtype}")

    # Analyze the mapping
    print("\nHilbert mapping analysis:")
    print(f"Min index: {hilbert_map.min().item()}")
    print(f"Max index: {hilbert_map.max().item()}")
    print(f"Unique values: {torch.unique(hilbert_map).shape[0]}")

    # Check if mapping covers the entire sequence
    expected_indices = torch.arange(seq_len, device=device)
    sorted_hilbert = torch.sort(hilbert_map)[0]
    covers_all = torch.equal(sorted_hilbert, expected_indices)
    print(f"Covers entire sequence: {covers_all}")

    # Analyze segment-wise behavior
    print(f"\nSegment analysis (segment_size={segment_size}):")
    for seg_idx in range(seq_len // segment_size):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size

        # Get Hilbert indices for this segment
        segment_hilbert = hilbert_map[seg_start:seg_end]

        # Check which original positions this segment maps to
        min_mapped = segment_hilbert.min().item()
        max_mapped = segment_hilbert.max().item()

        print(
            f"  Segment {seg_idx} [{seg_start}:{seg_end}] maps to indices [{min_mapped}:{max_mapped + 1}]"
        )

        # Check if segment is self-contained
        is_local = (min_mapped >= seg_start) and (max_mapped < seg_end)
        print(f"    Is locally contained: {is_local}")

    # Test with actual data to see memory access patterns
    print("\nTesting actual attention computation:")
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Create a special input where each position has a unique identifier
    for i in range(seq_len):
        x[0, i, 0] = i  # Position identifier in first element

    # Run forward pass
    with torch.no_grad():
        output_with_hilbert = module(x, use_hilbert=True)
        output_without_hilbert = module(x, use_hilbert=False)

    # Check if outputs differ
    diff = (output_with_hilbert - output_without_hilbert).abs().max().item()
    print(f"\nMax difference between Hilbert and non-Hilbert: {diff:.6f}")

    # Analyze attention pattern
    print("\nDilated attention pattern analysis:")
    print(f"Dilation rate: {dilation_rate}")
    print(f"Positions accessed per segment: {segment_size // dilation_rate}")
    print(
        f"Total positions accessed: {(seq_len // segment_size) * (segment_size // dilation_rate)}"
    )
    print(f"Sparsity: {1 - 1 / dilation_rate:.1%}")


def test_hilbert_kernel_internals():
    """Test to understand how Hilbert reordering interacts with dilated attention."""

    print("\n" + "=" * 60)
    print("HILBERT KERNEL INTERNALS")
    print("=" * 60)

    # Small example for clarity
    seq_len = 16
    segment_size = 8
    dilation_rate = 2

    # Create simple Hilbert mapping
    hilbert_map = torch.tensor(
        [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14], dtype=torch.int32
    )

    print(f"Sequence positions: {list(range(seq_len))}")
    print(f"Hilbert mapping:    {hilbert_map.tolist()}")

    # Simulate what happens in the kernel
    print("\nDilated attention access pattern:")

    for seg_idx in range(seq_len // segment_size):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size

        print(f"\nSegment {seg_idx} (positions {seg_start}-{seg_end - 1}):")

        # For each query position in segment
        for q_pos in range(seg_start, seg_end):
            # Which keys does this query attend to?
            attended_keys = []
            for k_pos in range(seg_start, seg_end):
                if (k_pos - seg_start) % dilation_rate == 0:
                    # This key position is attended to
                    # Get the Hilbert-reordered position
                    hilbert_pos = hilbert_map[k_pos].item()
                    attended_keys.append((k_pos, hilbert_pos))

            print(f"  Query {q_pos} attends to keys: {attended_keys}")


if __name__ == "__main__":
    analyze_hilbert_scope()
    test_hilbert_kernel_internals()
