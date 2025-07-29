#!/usr/bin/env python3
"""Test a fix for sparse Hilbert ordering."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def sparse_hilbert_ordering(seq_len, segment_size, dilation_rate):
    """
    Create a Hilbert-optimized ordering for sparse attention.
    Instead of applying Hilbert to all positions, we:
    1. Group sparse positions that will be accessed together
    2. Order them for better cache locality
    """
    num_segments = (seq_len + segment_size - 1) // segment_size

    # Collect all sparse positions by segment
    all_positions = []
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)
        seg_len = seg_end - seg_start

        # Get sparse positions for this segment
        num_sparse = (seg_len + dilation_rate - 1) // dilation_rate
        for i in range(num_sparse):
            pos = seg_start + i * dilation_rate
            if pos < seq_len:
                all_positions.append(pos)

    # Create inverse mapping (position -> reordered index)
    inverse_map = torch.zeros(seq_len, dtype=torch.long)
    for new_idx, old_idx in enumerate(all_positions):
        inverse_map[old_idx] = new_idx

    # For non-sparse positions, maintain relative order
    next_idx = len(all_positions)
    for i in range(seq_len):
        if i not in all_positions:
            inverse_map[i] = next_idx
            next_idx += 1

    return inverse_map


def test_sparse_optimized_ordering():
    """Test performance with sparse-optimized ordering."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuration
    seq_len = 2048
    batch_size = 2
    hidden_dim = 768
    num_heads = 12
    segment_size = 128
    dilation_rate = 4

    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Create module
    module = HilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        dropout=0.0,
    ).to(device)

    print("Testing sparse-optimized Hilbert ordering")
    print(f"Sequence length: {seq_len}, Dilation rate: {dilation_rate}")
    print()

    # Test standard sparse attention
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out_standard = module(x, use_hilbert=False)
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) * 1000

    # Test current Hilbert (global ordering)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out_hilbert = module(x, use_hilbert=True)
    torch.cuda.synchronize()
    hilbert_time = (time.perf_counter() - start) * 1000

    # Test idea: For sparse attention, just use standard ordering
    # since sparse access already has good locality within segments

    print(f"Standard sparse: {standard_time:.2f}ms")
    print(f"Global Hilbert: {hilbert_time:.2f}ms")
    print(f"Speedup: {standard_time / hilbert_time:.2f}x")

    # Verify outputs are similar (they should be different due to reordering)
    diff = torch.norm(out_standard - out_hilbert) / torch.norm(out_standard)
    print(f"\nRelative difference: {diff:.6f}")

    # Test memory access pattern
    print("\nMemory access analysis:")
    sparse_map = sparse_hilbert_ordering(256, segment_size, dilation_rate)

    # Check first segment's access pattern
    seg_positions = torch.arange(0, segment_size, dilation_rate)
    reordered = sparse_map[seg_positions]

    print(f"Original sparse positions: {seg_positions.tolist()[:8]}...")
    print(f"Reordered positions: {reordered.tolist()[:8]}...")

    jumps = torch.diff(reordered).abs().float().mean()
    print(f"Average jump distance: {jumps:.1f}")


if __name__ == "__main__":
    test_sparse_optimized_ordering()
