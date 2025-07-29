#!/usr/bin/env python3
"""Analyze the Hilbert mapping functions to understand their behavior."""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    create_hilbert_mapping,
    _hilbert_index_to_xy,
)


def analyze_hilbert_mappings():
    """Analyze different Hilbert mapping strategies."""
    print("HILBERT MAPPING ANALYSIS")
    print("=" * 80)

    # Test different sequence lengths
    seq_lengths = [64, 128, 256, 512, 1024, 4096]

    for seq_len in seq_lengths:
        print(f"\n\nSequence Length: {seq_len}")
        print("-" * 60)

        # Get the mapping
        mapping = create_hilbert_mapping(seq_len)

        # Determine which strategy was used
        if seq_len <= 64:
            strategy = "Identity (no reordering)"
        elif seq_len <= 512:
            strategy = "True Hilbert curve"
        else:
            strategy = "Snake pattern"

        print(f"Strategy used: {strategy}")
        print(f"Mapping shape: {mapping.shape}")
        print(f"Mapping dtype: {mapping.dtype}")

        # Analyze the mapping quality
        # 1. Check if it's a valid permutation
        unique_values = torch.unique(mapping)
        is_valid_permutation = len(unique_values) == seq_len and torch.all(
            unique_values == torch.arange(seq_len)
        )
        print(f"Valid permutation: {is_valid_permutation}")

        if not is_valid_permutation:
            print("WARNING: Invalid permutation detected!")
            missing = set(range(seq_len)) - set(mapping.tolist())
            duplicates = [x for x in mapping.tolist() if mapping.tolist().count(x) > 1]
            print(f"  Missing values: {missing}")
            print(f"  Duplicate values: {set(duplicates)}")

        # 2. Analyze locality - average jump distance
        jumps = []
        for i in range(1, seq_len):
            # Find where position i-1 and i map to
            prev_pos = mapping[i - 1].item()
            curr_pos = mapping[i].item()
            jump = abs(curr_pos - prev_pos)
            jumps.append(jump)

        avg_jump = np.mean(jumps)
        max_jump = max(jumps)
        std_jump = np.std(jumps)

        print("\nLocality analysis:")
        print(f"  Average jump: {avg_jump:.2f}")
        print(f"  Max jump: {max_jump}")
        print(f"  Std deviation: {std_jump:.2f}")
        print("  Sequential would be: 1.0")
        print(f"  Random would be: ~{seq_len / 3:.1f}")

        # 3. Check cache line efficiency
        cache_line_elements = 32  # Assuming 128 bytes / 4 bytes per float
        cache_misses = sum(1 for j in jumps if j > cache_line_elements)
        cache_miss_rate = cache_misses / len(jumps) * 100

        print("\nCache efficiency:")
        print(f"  Estimated cache miss rate: {cache_miss_rate:.1f}%")

        # 4. Visualize small mappings
        if seq_len <= 256:
            visualize_mapping(mapping, seq_len, strategy)


def visualize_mapping(mapping, seq_len, strategy):
    """Visualize the mapping pattern."""
    # Create a 2D representation
    grid_size = int(math.ceil(math.sqrt(seq_len)))
    grid = np.full((grid_size, grid_size), -1, dtype=int)

    # Fill the grid
    for i in range(seq_len):
        x = i % grid_size
        y = i // grid_size
        grid[y, x] = mapping[i].item()

    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Mapped Position")
    plt.title(f"{strategy} - Sequence Length {seq_len}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Add grid lines
    for i in range(grid_size + 1):
        plt.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
        plt.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"benchmarks/hilbert_mapping_{seq_len}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Visualization saved: benchmarks/hilbert_mapping_{seq_len}.png")


def test_specific_issues():
    """Test for specific issues in the implementation."""
    print("\n\nTESTING SPECIFIC ISSUES")
    print("=" * 80)

    # Issue 1: Check if the mapping is actually improving locality
    print("\n1. Testing locality improvement:")
    seq_len = 1024

    # Identity mapping (baseline)
    identity = torch.arange(seq_len)

    # Hilbert mapping
    hilbert = create_hilbert_mapping(seq_len)

    # Simulate cache access pattern
    def simulate_cache_hits(mapping, cache_size=64):
        """Simulate cache hits for a given access pattern."""
        cache = set()
        hits = 0
        misses = 0

        for i in range(seq_len):
            pos = mapping[i].item()

            # Check if in cache
            if pos in cache:
                hits += 1
            else:
                misses += 1
                # Add to cache
                cache.add(pos)
                # Evict if cache full (simple FIFO)
                if len(cache) > cache_size:
                    cache.pop()

        return hits, misses

    identity_hits, identity_misses = simulate_cache_hits(identity)
    hilbert_hits, hilbert_misses = simulate_cache_hits(hilbert)

    print(f"  Identity mapping: {identity_hits} hits, {identity_misses} misses")
    print(f"  Hilbert mapping: {hilbert_hits} hits, {hilbert_misses} misses")
    if identity_hits > 0:
        print(
            f"  Improvement: {(hilbert_hits - identity_hits) / identity_hits * 100:+.1f}%"
        )
    else:
        print("  Improvement: N/A (no cache hits with identity mapping)")

    # Issue 2: Check the actual Hilbert curve implementation
    print("\n2. Verifying Hilbert curve implementation:")

    # Test small Hilbert curve
    n = 4  # 4x4 grid
    expected_order = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (2, 1),
        (2, 2),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (1, 2),
        (0, 2),
        (0, 3),
    ]

    print("  Testing 4x4 Hilbert curve:")
    for i, (ex, ey) in enumerate(expected_order):
        x, y = _hilbert_index_to_xy(i, n)
        correct = x == ex and y == ey
        print(
            f"    Index {i}: expected ({ex},{ey}), got ({x},{y}) - {'✓' if correct else '✗'}"
        )

    # Issue 3: Performance of mapping generation
    print("\n3. Mapping generation performance:")

    import time

    for seq_len in [1024, 4096, 16384]:
        start = time.time()
        for _ in range(100):
            _ = create_hilbert_mapping(seq_len)
        gen_time = (time.time() - start) / 100 * 1000
        print(f"  Seq {seq_len}: {gen_time:.3f}ms per generation")

    # Issue 4: Memory access pattern simulation
    print("\n4. Memory access pattern analysis:")

    seq_len = 4096
    hilbert_map = create_hilbert_mapping(seq_len)

    # Simulate attention computation access pattern
    print(f"  Simulating attention access for seq_len={seq_len}")

    # In attention, for each query position, we access all key positions
    # Count unique cache lines accessed
    cache_line_size = 32  # elements

    total_accesses = 0
    unique_cache_lines = set()

    # Sample some query positions
    for q_idx in range(0, seq_len, seq_len // 10):  # Sample 10 query positions
        for k_idx in range(seq_len):
            mapped_k = hilbert_map[k_idx].item()
            cache_line = mapped_k // cache_line_size
            unique_cache_lines.add(cache_line)
            total_accesses += 1

    print(f"  Total accesses: {total_accesses}")
    print(f"  Unique cache lines touched: {len(unique_cache_lines)}")
    print(f"  Cache line reuse: {total_accesses / len(unique_cache_lines):.2f}x")


def analyze_sparse_pattern_hilbert():
    """Analyze Hilbert mapping for sparse patterns."""
    print("\n\nSPARSE PATTERN HILBERT ANALYSIS")
    print("=" * 80)

    from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention

    # Create module with sparse pattern
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,  # 75% sparse
        dropout=0.0,
        hilbert_threshold=0,  # Force Hilbert
    )

    seq_len = 512
    _ = "cpu"

    # Get sparse Hilbert mapping
    mapping = module._create_segment_local_hilbert_mapping(
        seq_len, module.segment_size, module.dilation_rate
    )

    print(f"Sparse pattern analysis (dilation_rate={module.dilation_rate}):")
    print(f"  Sequence length: {seq_len}")
    print(f"  Segment size: {module.segment_size}")
    print(
        f"  Active positions per segment: {module.segment_size // module.dilation_rate}"
    )

    # Check which positions are actually reordered
    identity = torch.arange(seq_len)
    changed_positions = torch.sum(mapping != identity).item()
    print(
        f"  Positions changed: {changed_positions} ({changed_positions / seq_len * 100:.1f}%)"
    )

    # Analyze per segment
    num_segments = (seq_len + module.segment_size - 1) // module.segment_size

    for seg_idx in range(min(3, num_segments)):  # First 3 segments
        seg_start = seg_idx * module.segment_size
        seg_end = min(seg_start + module.segment_size, seq_len)

        print(f"\n  Segment {seg_idx} ({seg_start}-{seg_end}):")

        # Get active positions
        active_positions = []
        for i in range(seg_start, seg_end, module.dilation_rate):
            active_positions.append(i)

        print(f"    Active positions: {active_positions[:10]}...")

        # Check their mapping
        mapped_positions = [mapping[pos].item() for pos in active_positions[:10]]
        print(f"    Mapped to: {mapped_positions}")


if __name__ == "__main__":
    analyze_hilbert_mappings()
    test_specific_issues()
    analyze_sparse_pattern_hilbert()
