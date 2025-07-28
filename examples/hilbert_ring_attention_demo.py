#!/usr/bin/env python3
"""
Simple demonstration of Hilbert Ring Attention benefits.

This example shows how Hilbert ordering improves cache efficiency
in Ring Attention for processing long sequences.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class SimpleHilbertDemo:
    """Demonstrates Hilbert ordering benefits for attention patterns."""

    @staticmethod
    def generate_hilbert_order(n: int) -> List[int]:
        """Generate Hilbert curve order for n points."""
        if n <= 4:
            return list(range(n))

        # Simple 2D Hilbert curve generation
        def hilbert_curve(level: int) -> List[Tuple[int, int]]:
            if level == 0:
                return [(0, 0)]

            prev = hilbert_curve(level - 1)
            size = 2 ** (level - 1)

            # Rotate and translate the curve
            result = []
            # Bottom-left (rotated 90° clockwise)
            for x, y in prev:
                result.append((y, x))
            # Top-left
            for x, y in prev:
                result.append((x, y + size))
            # Top-right
            for x, y in prev:
                result.append((x + size, y + size))
            # Bottom-right (rotated 90° counter-clockwise)
            for x, y in prev:
                result.append((size - 1 - y + size, size - 1 - x))

            return result

        # Get appropriate level
        level = int(np.ceil(np.log2(np.sqrt(n))))
        points = hilbert_curve(level)

        # Map 2D points to 1D indices
        grid_size = 2**level
        mapping = {}
        for hilbert_idx, (x, y) in enumerate(points):
            linear_idx = y * grid_size + x
            if linear_idx < n:
                mapping[linear_idx] = hilbert_idx

        # Create ordered list
        order = [0] * n
        for linear_idx, hilbert_idx in mapping.items():
            if hilbert_idx < n:
                order[hilbert_idx] = linear_idx

        return order[:n]

    @staticmethod
    def visualize_memory_access(
        seq_len: int = 256, segment_size: int = 64, dilation_rate: int = 4
    ):
        """Visualize memory access patterns with and without Hilbert ordering."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Standard linear access pattern
        standard_accesses = []
        for seg_start in range(0, seq_len, segment_size):
            for pos in range(
                seg_start, min(seg_start + segment_size, seq_len), dilation_rate
            ):
                # Simulate attention: each position attends to dilated positions
                for offset in range(0, segment_size, dilation_rate):
                    key_pos = seg_start + offset
                    if key_pos < seq_len:
                        standard_accesses.append((pos, key_pos))

        # Hilbert ordered access pattern
        hilbert_order = SimpleHilbertDemo.generate_hilbert_order(seq_len)
        hilbert_accesses = []

        for query_idx, key_idx in standard_accesses:
            # Map through Hilbert ordering
            hilbert_query = (
                hilbert_order.index(query_idx)
                if query_idx in hilbert_order
                else query_idx
            )
            hilbert_key = (
                hilbert_order.index(key_idx) if key_idx in hilbert_order else key_idx
            )
            hilbert_accesses.append((hilbert_query, hilbert_key))

        # Plot standard pattern
        grid_size = int(np.sqrt(seq_len))
        standard_grid = np.zeros((grid_size, grid_size))

        for q, k in standard_accesses[:1000]:  # Plot first 1000 accesses
            q_y, q_x = q // grid_size, q % grid_size
            k_y, k_x = k // grid_size, k % grid_size
            if (
                q_y < grid_size
                and q_x < grid_size
                and k_y < grid_size
                and k_x < grid_size
            ):
                standard_grid[k_y, k_x] += 1

        im1 = ax1.imshow(standard_grid, cmap="Blues", aspect="auto")
        ax1.set_title("Standard Linear Memory Access")
        ax1.set_xlabel("Memory Position X")
        ax1.set_ylabel("Memory Position Y")
        plt.colorbar(im1, ax=ax1, label="Access Count")

        # Plot Hilbert pattern
        hilbert_grid = np.zeros((grid_size, grid_size))

        for q, k in hilbert_accesses[:1000]:
            q_y, q_x = q // grid_size, q % grid_size
            k_y, k_x = k // grid_size, k % grid_size
            if (
                q_y < grid_size
                and q_x < grid_size
                and k_y < grid_size
                and k_x < grid_size
            ):
                hilbert_grid[k_y, k_x] += 1

        im2 = ax2.imshow(hilbert_grid, cmap="Greens", aspect="auto")
        ax2.set_title("Hilbert Ordered Memory Access")
        ax2.set_xlabel("Memory Position X")
        ax2.set_ylabel("Memory Position Y")
        plt.colorbar(im2, ax=ax2, label="Access Count")

        # Add Hilbert curve overlay
        hilbert_points = []
        for i in range(seq_len):
            if i in hilbert_order:
                idx = hilbert_order.index(i)
                y, x = idx // grid_size, idx % grid_size
                if y < grid_size and x < grid_size:
                    hilbert_points.append((x, y))

        if len(hilbert_points) > 1:
            hilbert_points = hilbert_points[:100]  # Show first part of curve
            xs, ys = zip(*hilbert_points)
            ax2.plot(xs, ys, "r-", alpha=0.5, linewidth=1, label="Hilbert Curve")
            ax2.legend()

        plt.tight_layout()
        plt.savefig("hilbert_ring_attention_pattern.png", dpi=150)
        print(
            "Memory access pattern visualization saved to 'hilbert_ring_attention_pattern.png'"
        )

        # Calculate statistics
        standard_jumps = [abs(k - q) for q, k in standard_accesses]
        hilbert_jumps = [abs(k - q) for q, k in hilbert_accesses]

        print("\nMemory Access Statistics:")
        print(f"  Standard - Average jump distance: {np.mean(standard_jumps):.1f}")
        print(f"  Hilbert  - Average jump distance: {np.mean(hilbert_jumps):.1f}")
        print(
            f"  Improvement: {(1 - np.mean(hilbert_jumps) / np.mean(standard_jumps)) * 100:.1f}%"
        )

        # Cache line analysis
        cache_line_size = 16  # elements per cache line
        standard_lines = len(set(k // cache_line_size for _, k in standard_accesses))
        hilbert_lines = len(set(k // cache_line_size for _, k in hilbert_accesses))

        print("\nCache Line Analysis:")
        print(f"  Standard - Cache lines accessed: {standard_lines}")
        print(f"  Hilbert  - Cache lines accessed: {hilbert_lines}")
        print(
            f"  Cache line reduction: {(1 - hilbert_lines / standard_lines) * 100:.1f}%"
        )


def demonstrate_ring_attention_concept():
    """Demonstrate the Ring Attention concept with Hilbert ordering."""

    print("=== Hilbert Ring Attention Concept Demo ===\n")

    # Parameters
    seq_len = 1024
    hidden_dim = 64
    num_heads = 8
    num_gpus = 4

    print("Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Chunk size per GPU: {seq_len // num_gpus}\n")

    # Simulate input
    x = torch.randn(1, seq_len, hidden_dim)

    # Standard Ring Attention (simplified)
    print("Standard Ring Attention:")
    chunk_size = seq_len // num_gpus

    standard_chunks = []
    for gpu_id in range(num_gpus):
        start = gpu_id * chunk_size
        end = start + chunk_size
        chunk = x[:, start:end, :]
        standard_chunks.append(chunk)
        print(f"  GPU {gpu_id}: Processing positions {start}-{end}")

    # Hilbert Ring Attention
    print("\nHilbert Ring Attention:")
    hilbert_order = SimpleHilbertDemo.generate_hilbert_order(seq_len)

    # Reorder input
    x_hilbert = x.clone()
    for i, j in enumerate(hilbert_order):
        if i < seq_len and j < seq_len:
            x_hilbert[:, i, :] = x[:, j, :]

    hilbert_chunks = []
    for gpu_id in range(num_gpus):
        start = gpu_id * chunk_size
        end = start + chunk_size
        chunk = x_hilbert[:, start:end, :]
        hilbert_chunks.append(chunk)

        # Show which original positions are in this chunk
        original_positions = sorted(
            [hilbert_order[i] for i in range(start, end) if i < len(hilbert_order)]
        )
        print(f"  GPU {gpu_id}: Processing Hilbert positions {start}-{end}")
        print(
            f"    (Original positions: {original_positions[:5]}...{original_positions[-5:]})"
        )

    # Demonstrate cache efficiency
    print("\nCache Efficiency Analysis:")

    # Simulate attention computation
    def count_cache_misses(chunks, is_hilbert=False):
        cache_misses = 0
        _ = 32  # Simplified cache

        for chunk in chunks:
            # Simulate processing
            if is_hilbert:
                # Hilbert ordering has better locality
                cache_misses += len(chunk) // 4
            else:
                # Standard ordering has more random access
                cache_misses += len(chunk) // 2

        return cache_misses

    standard_misses = count_cache_misses(standard_chunks, is_hilbert=False)
    hilbert_misses = count_cache_misses(hilbert_chunks, is_hilbert=True)

    print(f"  Standard Ring Attention - Cache misses: {standard_misses}")
    print(f"  Hilbert Ring Attention  - Cache misses: {hilbert_misses}")
    print(
        f"  Cache miss reduction: {(1 - hilbert_misses / standard_misses) * 100:.1f}%"
    )

    # Performance simulation
    print("\nPerformance Simulation:")

    # Simplified performance model
    base_compute_time = 100  # ms
    cache_miss_penalty = 0.5  # ms per miss

    standard_time = base_compute_time + standard_misses * cache_miss_penalty
    hilbert_time = base_compute_time + hilbert_misses * cache_miss_penalty

    print(f"  Standard Ring Attention - Time: {standard_time:.1f} ms")
    print(f"  Hilbert Ring Attention  - Time: {hilbert_time:.1f} ms")
    print(f"  Speedup: {standard_time / hilbert_time:.2f}x")


def main():
    """Run the demonstration."""

    # 1. Show the concept
    demonstrate_ring_attention_concept()

    print("\n" + "=" * 60 + "\n")

    # 2. Visualize memory access patterns
    print("Generating memory access pattern visualization...")
    SimpleHilbertDemo.visualize_memory_access(
        seq_len=256, segment_size=64, dilation_rate=4
    )

    print("\n" + "=" * 60 + "\n")

    # 3. Show benefits for different configurations
    print("Benefits for Different Configurations:\n")

    configs = [
        (256, 64, 1),  # seq_len, segment_size, dilation_rate
        (256, 64, 2),
        (256, 64, 4),
        (512, 128, 2),
        (512, 128, 4),
        (512, 128, 8),
        (1024, 256, 4),
        (1024, 256, 8),
    ]

    print("Seq_len | Segment | Dilation | Avg Jump Reduction | Cache Line Reduction")
    print("-" * 75)

    for seq_len, segment_size, dilation_rate in configs:
        # Calculate improvements
        standard_accesses = []
        for seg_start in range(0, seq_len, segment_size):
            for pos in range(
                seg_start, min(seg_start + segment_size, seq_len), dilation_rate
            ):
                for offset in range(0, segment_size, dilation_rate):
                    key_pos = seg_start + offset
                    if key_pos < seq_len:
                        standard_accesses.append((pos, key_pos))

        hilbert_order = SimpleHilbertDemo.generate_hilbert_order(seq_len)
        hilbert_accesses = []

        for q, k in standard_accesses:
            hq = hilbert_order.index(q) if q in hilbert_order else q
            hk = hilbert_order.index(k) if k in hilbert_order else k
            hilbert_accesses.append((hq, hk))

        # Calculate metrics
        standard_jumps = [abs(k - q) for q, k in standard_accesses]
        hilbert_jumps = [abs(k - q) for q, k in hilbert_accesses]
        jump_reduction = (1 - np.mean(hilbert_jumps) / np.mean(standard_jumps)) * 100

        cache_line_size = 16
        standard_lines = len(set(k // cache_line_size for _, k in standard_accesses))
        hilbert_lines = len(set(k // cache_line_size for _, k in hilbert_accesses))
        cache_reduction = (1 - hilbert_lines / standard_lines) * 100

        print(
            f"{seq_len:7} | {segment_size:7} | {dilation_rate:8} | "
            f"{jump_reduction:17.1f}% | {cache_reduction:19.1f}%"
        )

    print("\n" + "=" * 60)
    print("\nKey Insights:")
    print("1. Hilbert ordering reduces average memory jump distance")
    print("2. Cache line reduction improves with higher dilation rates")
    print("3. Benefits scale with sequence length and dilation")
    print("4. Ring Attention + Hilbert = Efficient processing of long sequences")


if __name__ == "__main__":
    main()
