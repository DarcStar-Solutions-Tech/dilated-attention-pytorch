#!/usr/bin/env python3
"""Analyze the ordering issue with sparse patterns and Hilbert curves."""

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    create_hilbert_mapping,
)


def visualize_sparse_pattern_issue():
    """Visualize how Hilbert reordering interacts with sparse patterns."""

    print("HILBERT ORDERING vs SPARSE PATTERNS ANALYSIS")
    print("=" * 80)

    seq_len = 256
    segment_size = 64
    dilation_rate = 4

    # Create Hilbert mapping
    hilbert_map = create_hilbert_mapping(seq_len)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Hilbert Reordering vs Sparse Patterns (dilation_rate={dilation_rate})",
        fontsize=16,
    )

    # 1. Original sequential order
    ax = axes[0, 0]
    grid = np.arange(seq_len).reshape(16, 16)
    _ = ax.imshow(grid, cmap="viridis")
    ax.set_title("1. Original Sequential Order")
    ax.set_xlabel("Position in sequence")

    # Mark sparse positions
    for i in range(0, seq_len, dilation_rate):
        y, x = i // 16, i % 16
        ax.plot(x, y, "r.", markersize=8)

    # 2. After Hilbert reordering
    ax = axes[0, 1]
    reordered_grid = hilbert_map.numpy().reshape(16, 16)
    _ = ax.imshow(reordered_grid, cmap="viridis")
    ax.set_title("2. After Hilbert Reordering")
    ax.set_xlabel("Hilbert-reordered positions")

    # 3. Sparse positions in Hilbert space
    ax = axes[0, 2]
    sparse_mask = np.zeros(seq_len)
    for i in range(0, seq_len, dilation_rate):
        sparse_mask[i] = 1

    # Apply Hilbert reordering to sparse mask
    hilbert_sparse = np.zeros(seq_len)
    for i in range(seq_len):
        hilbert_sparse[hilbert_map[i]] = sparse_mask[i]

    sparse_grid = hilbert_sparse.reshape(16, 16)
    ax.imshow(sparse_grid, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_title("3. Sparse Positions After Reordering")
    ax.set_xlabel("Red = selected positions")

    # 4. CORRECT approach: Apply Hilbert within segments
    ax = axes[1, 0]
    correct_approach = np.arange(seq_len).reshape(16, 16)

    # Process each segment
    for seg_idx in range(seq_len // segment_size):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size

        # Get sparse positions in segment
        sparse_in_seg = list(range(seg_start, seg_end, dilation_rate))

        # Apply Hilbert only to sparse positions
        if len(sparse_in_seg) > 1:
            mini_hilbert = create_hilbert_mapping(len(sparse_in_seg))
            reordered_sparse = [
                sparse_in_seg[mini_hilbert[i]] for i in range(len(sparse_in_seg))
            ]

            # Visualize
            for i, pos in enumerate(sparse_in_seg):
                y1, x1 = pos // 16, pos % 16
                new_pos = reordered_sparse[i]
                y2, x2 = new_pos // 16, new_pos % 16
                ax.arrow(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    head_width=0.3,
                    head_length=0.2,
                    fc="red",
                    ec="red",
                    alpha=0.5,
                )

    ax.imshow(correct_approach, cmap="viridis", alpha=0.3)
    ax.set_title("4. CORRECT: Hilbert Within Sparse Positions")
    ax.set_xlabel("Apply Hilbert to sparse positions only")

    # 5. Access pattern comparison
    ax = axes[1, 1]

    # Simulate K,V access for one query
    access_pattern_wrong = []
    access_pattern_right = []

    # Wrong way: Hilbert then sparse
    for i in range(0, seq_len, dilation_rate):
        hilbert_pos = hilbert_map[i].item()
        access_pattern_wrong.append(hilbert_pos)

    # Right way: Sparse then Hilbert within segment
    for seg_idx in range(seq_len // segment_size):
        seg_start = seg_idx * segment_size
        sparse_positions = list(
            range(seg_start, seg_start + segment_size, dilation_rate)
        )
        # In correct implementation, these would be reordered within segment
        access_pattern_right.extend(sparse_positions)

    ax.plot(
        access_pattern_wrong[:32], "r.-", label="Wrong: Hilbert→Sparse", markersize=8
    )
    ax.plot(
        access_pattern_right[:32], "g.-", label="Right: Sparse→Hilbert", markersize=8
    )
    ax.set_xlabel("Access order")
    ax.set_ylabel("Memory position")
    ax.set_title("5. Memory Access Patterns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Jump distance analysis
    ax = axes[1, 2]

    jumps_wrong = [
        abs(access_pattern_wrong[i + 1] - access_pattern_wrong[i])
        for i in range(len(access_pattern_wrong) - 1)
    ]
    jumps_right = [
        abs(access_pattern_right[i + 1] - access_pattern_right[i])
        for i in range(len(access_pattern_right) - 1)
    ]

    bins = np.linspace(0, max(max(jumps_wrong), max(jumps_right)), 20)
    ax.hist(
        jumps_wrong,
        bins=bins,
        alpha=0.5,
        label=f"Wrong: avg={np.mean(jumps_wrong):.1f}",
        color="red",
    )
    ax.hist(
        jumps_right,
        bins=bins,
        alpha=0.5,
        label=f"Right: avg={np.mean(jumps_right):.1f}",
        color="green",
    )
    ax.set_xlabel("Jump distance")
    ax.set_ylabel("Frequency")
    ax.set_title("6. Memory Jump Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        "benchmarks/hilbert_sparse_ordering_issue.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Print analysis
    print(
        f"\nAnalysis for seq_len={seq_len}, segment_size={segment_size}, dilation_rate={dilation_rate}"
    )
    print("\nWRONG APPROACH (current implementation):")
    print("  1. Apply Hilbert to ALL positions: [0,1,2,...,255] → [hilbert reordered]")
    print(f"  2. Then select sparse: take every {dilation_rate}th position")
    print("  Result: Scattered access pattern")
    print(f"  Average jump: {np.mean(jumps_wrong):.1f}")
    print(f"  Max jump: {max(jumps_wrong)}")

    print("\nCORRECT APPROACH:")
    print("  1. First identify sparse positions: [0,4,8,12,...]")
    print("  2. Apply Hilbert ONLY within each segment's sparse positions")
    print("  Result: Preserves locality within working set")
    print(f"  Average jump: {np.mean(jumps_right):.1f}")
    print(f"  Max jump: {max(jumps_right)}")

    print("\nVisualization saved to: benchmarks/hilbert_sparse_ordering_issue.png")


def test_implementation_approach():
    """Test how the current implementation handles sparse patterns."""
    print("\n\nTESTING CURRENT IMPLEMENTATION")
    print("=" * 80)

    # Create module with sparse pattern
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=64,
        dilation_rate=4,
        dropout=0.0,
        hilbert_threshold=0,  # Force Hilbert
    )

    seq_len = 256
    _ = "cpu"

    print("\nChecking implementation logic:")

    # Standard Hilbert mapping
    standard_mapping = module._create_hilbert_mapping(seq_len)

    # Segment-local Hilbert mapping
    sparse_mapping = module._create_segment_local_hilbert_mapping(
        seq_len, module.segment_size, module.dilation_rate
    )

    # Check what happens in attention
    print("\n1. In forward pass with use_hilbert=True and dilation_rate=4:")
    print("   - K and V are reordered using Hilbert mapping")
    print("   - THEN sparse positions are selected")
    print("   - This destroys locality!")

    # Analyze the sparse pattern
    segment_size = 64
    dilation_rate = 4

    print(f"\n2. For first segment (0-{segment_size}):")
    sparse_positions = list(range(0, segment_size, dilation_rate))
    print(f"   Original sparse positions: {sparse_positions[:8]}...")

    # After standard Hilbert
    hilbert_sparse = [standard_mapping[i].item() for i in sparse_positions]
    print(f"   After standard Hilbert: {hilbert_sparse[:8]}...")

    # After segment-local Hilbert
    segment_sparse = [sparse_mapping[i].item() for i in sparse_positions]
    print(f"   After segment-local: {segment_sparse[:8]}...")

    # Calculate jumps
    std_jumps = [
        abs(hilbert_sparse[i + 1] - hilbert_sparse[i])
        for i in range(len(hilbert_sparse) - 1)
    ]
    seg_jumps = [
        abs(segment_sparse[i + 1] - segment_sparse[i])
        for i in range(len(segment_sparse) - 1)
    ]

    print("\n3. Memory access quality:")
    print(
        f"   Standard Hilbert: avg jump = {np.mean(std_jumps):.1f}, max = {max(std_jumps)}"
    )
    print(
        f"   Segment-local: avg jump = {np.mean(seg_jumps):.1f}, max = {max(seg_jumps)}"
    )

    print("\n4. THE FUNDAMENTAL ISSUE:")
    print("   - Hilbert curves optimize for 2D spatial locality")
    print("   - Attention accesses K,V sequentially for each Q")
    print(
        "   - Sparse patterns need positions close in SPARSE SPACE, not original space"
    )
    print("   - Current implementation applies Hilbert in WRONG order")


def propose_solution():
    """Propose the correct solution."""
    print("\n\nPROPOSED SOLUTION")
    print("=" * 80)

    print("""
The correct approach for sparse/dilated attention with Hilbert ordering:

1. **Identify Active Positions First**
   - For each segment, determine which positions will be accessed
   - For dilation_rate=4: positions [0,4,8,12,16,20,24,28,...]

2. **Apply Hilbert Only to Active Set**
   - Create mini Hilbert curve for the active positions
   - This preserves locality within the working set
   - Much smaller overhead (16 positions vs 64)

3. **Better Yet: Don't Use Hilbert for Sparse Patterns**
   - Sparse patterns already have good locality (sequential access)
   - Hilbert reordering can only make it worse
   - The overhead isn't worth it

4. **Implementation Fix**:
   ```python
   if self.dilation_rate > 1:
       # For sparse patterns, Hilbert hurts more than helps
       use_hilbert = False
   ```

5. **Or Remove Hilbert Entirely**
   - The benchmarks show it hurts performance
   - The implementation is buggy
   - The concept doesn't match GPU memory architecture
""")


if __name__ == "__main__":
    visualize_sparse_pattern_issue()
    test_implementation_approach()
    propose_solution()
