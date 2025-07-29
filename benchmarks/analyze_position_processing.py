#!/usr/bin/env python3
"""Analyze position processing inefficiency in sparse attention."""


def analyze_sparse_positions():
    """Analyze how many positions are actually used vs processed."""

    print("Position Processing Analysis for Sparse Attention")
    print("=" * 60)

    # Test different configurations
    configs = [
        (4096, 128, 1, "Dense (baseline)"),
        (4096, 128, 2, "50% sparse"),
        (4096, 128, 4, "75% sparse"),
        (4096, 128, 8, "87.5% sparse"),
        (8192, 256, 4, "Large segment, 75% sparse"),
    ]

    for seq_len, segment_size, dilation_rate, desc in configs:
        print(f"\n{desc}:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Segment size: {segment_size}")
        print(f"  Dilation rate: {dilation_rate}")

        # Calculate statistics
        num_segments = seq_len // segment_size
        positions_per_segment = segment_size // dilation_rate
        total_active = num_segments * positions_per_segment

        # Current approach: process all positions
        positions_processed = seq_len

        # Optimal approach: process only active positions
        positions_needed = total_active

        # Waste ratio
        waste_ratio = (
            positions_processed / positions_needed if positions_needed > 0 else 1
        )

        print(f"  Segments: {num_segments}")
        print(f"  Active positions per segment: {positions_per_segment}")
        print(f"  Total active positions: {total_active}")
        print(f"  Current approach processes: {positions_processed}")
        print(f"  Optimal would process: {positions_needed}")
        print(f"  Waste factor: {waste_ratio:.1f}x")

        # Memory access pattern
        if dilation_rate > 1:
            print(f"  Memory pattern: Every {dilation_rate}th position")
            # Show example positions for first segment
            example_positions = list(range(0, min(segment_size, 16), dilation_rate))
            print(f"  Example (first segment): {example_positions}")


def simulate_kernel_iterations():
    """Simulate how many iterations the kernel makes."""

    print("\n\nKernel Iteration Analysis")
    print("=" * 60)

    BLOCK_N = 64  # Typical block size

    for seq_len, segment_size, dilation_rate in [(4096, 128, 4), (8192, 256, 8)]:
        print(
            f"\nConfig: seq={seq_len}, segment={segment_size}, dilation={dilation_rate}"
        )

        # Current approach
        current_iterations = seq_len // BLOCK_N

        # Optimal approach for sparse
        active_per_segment = segment_size // dilation_rate
        active_blocks_per_segment = (active_per_segment + BLOCK_N - 1) // BLOCK_N
        num_segments = seq_len // segment_size
        optimal_iterations = num_segments * active_blocks_per_segment

        print(f"  Current kernel iterations: {current_iterations}")
        print(f"  Optimal kernel iterations: {optimal_iterations}")
        print(f"  Reduction factor: {current_iterations / optimal_iterations:.1f}x")

        # Show memory access pattern
        print(
            f"  Current: Check all {seq_len} positions, keep {seq_len // dilation_rate}"
        )
        print(f"  Optimal: Only check {seq_len // dilation_rate} positions")


if __name__ == "__main__":
    analyze_sparse_positions()
    simulate_kernel_iterations()
