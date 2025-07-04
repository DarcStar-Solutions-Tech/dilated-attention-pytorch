#!/usr/bin/env python3
"""
Simple demonstration of Hilbert curve benefits for dilated attention.
Uses a basic implementation to show the concept without complex dependencies.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict


def generate_simple_hilbert_order(n: int) -> List[int]:
    """Generate a simple Hilbert-like ordering for n points."""
    # For simplicity, just create a pattern that groups nearby elements
    # Real Hilbert curve would be more complex
    order = []
    block_size = int(np.sqrt(n))

    for block_start in range(0, n, block_size):
        block = list(range(block_start, min(block_start + block_size, n)))
        # Alternate direction for each block (snake pattern)
        if (block_start // block_size) % 2 == 1:
            block.reverse()
        order.extend(block)

    return order[:n]


def simple_dilated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_size: int,
    dilation_rate: int,
    ordering: List[int] = None,
) -> torch.Tensor:
    """Simple dilated attention implementation."""

    batch_size, seq_len, hidden_dim = query.shape

    # Apply ordering if provided
    if ordering is not None:
        perm = torch.tensor(ordering, device=query.device)
        query = query[:, perm]
        key = key[:, perm]
        value = value[:, perm]

    output = torch.zeros_like(query)

    # Process each segment
    for seg_start in range(0, seq_len, segment_size):
        seg_end = min(seg_start + segment_size, seq_len)

        # Get queries for this segment
        q_seg = query[:, seg_start:seg_end]

        # Get dilated keys and values
        key_positions = list(range(seg_start, seg_end, dilation_rate))
        if key_positions:
            k_seg = key[:, key_positions]
            v_seg = value[:, key_positions]

            # Compute attention
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) / np.sqrt(hidden_dim)
            attn_weights = F.softmax(scores, dim=-1)
            output[:, seg_start:seg_end] = torch.matmul(attn_weights, v_seg)

    # Reverse ordering if applied
    if ordering is not None:
        inverse_perm = torch.argsort(perm)
        output = output[:, inverse_perm]

    return output


def measure_memory_access_pattern(
    seq_len: int, segment_size: int, dilation_rate: int, ordering: List[int] = None
) -> Dict[str, float]:
    """Measure characteristics of memory access pattern."""

    accesses = []

    # Simulate access pattern
    for seg_start in range(0, seq_len, segment_size):
        seg_end = min(seg_start + segment_size, seq_len)

        # Query positions
        for q_pos in range(seg_start, seg_end):
            # Key positions (dilated)
            for k_pos in range(seg_start, seg_end, dilation_rate):
                if ordering:
                    # Map through ordering
                    q_actual = ordering[q_pos] if q_pos < len(ordering) else q_pos
                    k_actual = ordering[k_pos] if k_pos < len(ordering) else k_pos
                    accesses.append((q_actual, k_actual))
                else:
                    accesses.append((q_pos, k_pos))

    # Calculate average memory distance
    distances = [abs(k - q) for q, k in accesses]
    avg_distance = np.mean(distances) if distances else 0

    # Calculate cache line hits (assuming 16 elements per cache line)
    cache_line_size = 16
    cache_lines_accessed = len(set((k // cache_line_size) for _, k in accesses))

    return {
        "avg_distance": avg_distance,
        "total_accesses": len(accesses),
        "cache_lines": cache_lines_accessed,
        "cache_efficiency": len(accesses) / max(cache_lines_accessed, 1),
    }


def benchmark_attention_patterns():
    """Benchmark different attention patterns."""

    print("=== Simple Hilbert Dilated Attention Demonstration ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Test configurations
    configs = [
        (512, 128, 1),  # seq_len, segment_size, dilation_rate
        (512, 128, 2),
        (512, 128, 4),
        (512, 128, 8),
        (1024, 256, 1),
        (1024, 256, 2),
        (1024, 256, 4),
        (1024, 256, 8),
    ]

    results = []

    print(
        "Configuration        | Standard (ms) | Hilbert (ms) | Speedup | Cache Improvement"
    )
    print("-" * 80)

    for seq_len, segment_size, dilation_rate in configs:
        # Create test data
        batch_size = 4
        hidden_dim = 256

        q = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        k = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        v = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Generate Hilbert ordering
        hilbert_order = generate_simple_hilbert_order(seq_len)

        # Warmup
        for _ in range(10):
            _ = simple_dilated_attention(q, k, v, segment_size, dilation_rate)
            _ = simple_dilated_attention(
                q, k, v, segment_size, dilation_rate, hilbert_order
            )

        # Benchmark standard
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            _ = simple_dilated_attention(q, k, v, segment_size, dilation_rate)
        standard_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark Hilbert
        start = time.perf_counter()
        for _ in range(iterations):
            _ = simple_dilated_attention(
                q, k, v, segment_size, dilation_rate, hilbert_order
            )
        hilbert_time = (time.perf_counter() - start) / iterations * 1000

        # Analyze patterns
        standard_pattern = measure_memory_access_pattern(
            seq_len, segment_size, dilation_rate
        )
        hilbert_pattern = measure_memory_access_pattern(
            seq_len, segment_size, dilation_rate, hilbert_order
        )

        speedup = standard_time / hilbert_time
        cache_improvement = (
            standard_pattern["cache_lines"] - hilbert_pattern["cache_lines"]
        ) / standard_pattern["cache_lines"]

        results.append(
            {
                "config": (seq_len, segment_size, dilation_rate),
                "standard_time": standard_time,
                "hilbert_time": hilbert_time,
                "speedup": speedup,
                "cache_improvement": cache_improvement,
                "standard_pattern": standard_pattern,
                "hilbert_pattern": hilbert_pattern,
            }
        )

        print(
            f"L={seq_len:4} seg={segment_size:3} d={dilation_rate} | "
            f"{standard_time:13.2f} | {hilbert_time:12.2f} | "
            f"{speedup:7.2f} | {cache_improvement * 100:16.1f}%"
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    speedups = [r["speedup"] for r in results]
    cache_improvements = [r["cache_improvement"] for r in results]

    print("\nPerformance:")
    print(f"  Average speedup: {np.mean(speedups):.2f}x")
    print(f"  Best speedup: {max(speedups):.2f}x")
    print(
        f"  Speedup > 1: {sum(1 for s in speedups if s > 1)}/{len(speedups)} configurations"
    )

    print("\nCache Efficiency:")
    print(f"  Average improvement: {np.mean(cache_improvements) * 100:.1f}%")
    print(f"  Best improvement: {max(cache_improvements) * 100:.1f}%")

    # Visualize one example
    visualize_access_patterns(512, 128, 4)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
    This simple demonstration shows:
    
    1. Hilbert-like orderings can improve cache efficiency for dilated attention
    2. Benefits depend on sequence length, segment size, and dilation rate
    3. Actual speedups vary based on hardware and implementation
    4. The concept is sound but requires careful optimization
    
    In practice, a full CUDA/Triton implementation with proper Hilbert curves
    and optimized memory access would show more consistent benefits.
    """)


def visualize_access_patterns(seq_len: int, segment_size: int, dilation_rate: int):
    """Visualize memory access patterns."""

    print("\nGenerating visualization...")

    # Create simple 2D representation
    grid_size = int(np.sqrt(seq_len))

    # Standard pattern
    standard_grid = np.zeros((grid_size, grid_size))
    hilbert_order = generate_simple_hilbert_order(seq_len)
    hilbert_grid = np.zeros((grid_size, grid_size))

    # Mark accessed positions for one segment
    seg_start = segment_size
    for offset in range(0, min(segment_size, seq_len - seg_start), dilation_rate):
        pos = seg_start + offset
        # Standard
        x, y = pos % grid_size, pos // grid_size
        if x < grid_size and y < grid_size:
            standard_grid[y, x] = 1

        # Hilbert
        hilbert_pos = hilbert_order[pos] if pos < len(hilbert_order) else pos
        x, y = hilbert_pos % grid_size, hilbert_pos // grid_size
        if x < grid_size and y < grid_size:
            hilbert_grid[y, x] = 1

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(standard_grid, cmap="Blues", interpolation="nearest")
    ax1.set_title("Standard Linear Access")
    ax1.set_xlabel("Memory Position X")
    ax1.set_ylabel("Memory Position Y")

    ax2.imshow(hilbert_grid, cmap="Greens", interpolation="nearest")
    ax2.set_title("Hilbert-like Access")
    ax2.set_xlabel("Memory Position X")
    ax2.set_ylabel("Memory Position Y")

    plt.tight_layout()
    plt.savefig("hilbert_simple_demo.png", dpi=150)
    print("Visualization saved to 'hilbert_simple_demo.png'")


if __name__ == "__main__":
    benchmark_attention_patterns()
