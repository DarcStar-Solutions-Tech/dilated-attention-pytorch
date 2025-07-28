"""
Debug script to understand the correct dilation approach for Ring Attention.

This script demonstrates the difference between:
1. Local dilation (within segment) - CORRECT
2. Global dilation (across sequence) - INCORRECT
"""

import torch


def demonstrate_dilation_approaches():
    """Show the difference between local and global dilation."""
    seq_len = 2048
    segment_length = 512
    dilation_rate = 2

    print("=== Dilation Approaches for Ring Attention ===")
    print(f"Sequence length: {seq_len}")
    print(f"Segment length: {segment_length}")
    print(f"Dilation rate: {dilation_rate}")
    print()

    # Create segments
    num_segments = seq_len // segment_length
    print(f"Number of segments: {num_segments}")
    print()

    # Approach 1: Local Dilation (CORRECT)
    print("APPROACH 1: Local Dilation (within segment)")
    print("-" * 50)

    for seg_idx in range(num_segments):
        start_idx = seg_idx * segment_length
        end_idx = start_idx + segment_length

        # Generate dilated indices within the segment
        local_dilated_indices = torch.arange(0, segment_length, dilation_rate)

        # Map to global sequence positions
        global_positions = start_idx + local_dilated_indices

        print(f"Segment {seg_idx}: positions {start_idx}-{end_idx}")
        print(
            f"  Local dilated indices: {local_dilated_indices[:5].tolist()}...{local_dilated_indices[-5:].tolist()}"
        )
        print(
            f"  Global positions: {global_positions[:5].tolist()}...{global_positions[-5:].tolist()}"
        )
        print(f"  Number of dilated positions: {len(local_dilated_indices)}")
        print()

    # Approach 2: Global Dilation (INCORRECT - my buggy implementation)
    print("\nAPPROACH 2: Global Dilation (INCORRECT)")
    print("-" * 50)

    for seg_idx in range(num_segments):
        start_idx = seg_idx * segment_length
        end_idx = start_idx + segment_length

        # Incorrectly trying to dilate from segment start across full sequence
        global_dilated_indices = []
        for i in range(segment_length):
            idx = start_idx + i * dilation_rate
            if idx < seq_len:
                global_dilated_indices.append(idx)

        print(f"Segment {seg_idx}: positions {start_idx}-{end_idx}")
        if global_dilated_indices:
            print(
                f"  Global dilated indices: {global_dilated_indices[:5]}...{global_dilated_indices[-5:]}"
            )
            print(f"  Number of dilated positions: {len(global_dilated_indices)}")
            print(f"  Max index: {max(global_dilated_indices)} (exceeds segment!)")
        print()

    # Show the correct pattern visually
    print("\n=== Visual Representation ===")
    print("CORRECT (Local Dilation):")
    print("Segment 0: [0, 2, 4, ..., 510] -> 256 positions within [0, 512)")
    print("Segment 1: [512, 514, 516, ..., 1022] -> 256 positions within [512, 1024)")
    print(
        "Segment 2: [1024, 1026, 1028, ..., 1534] -> 256 positions within [1024, 1536)"
    )
    print(
        "Segment 3: [1536, 1538, 1540, ..., 2046] -> 256 positions within [1536, 2048)"
    )
    print()
    print("INCORRECT (Global Dilation):")
    print("Segment 0: [0, 2, 4, ..., 1022] -> extends beyond segment!")
    print("Segment 1: [512, 514, 516, ..., 1534] -> extends beyond segment!")


if __name__ == "__main__":
    demonstrate_dilation_approaches()
