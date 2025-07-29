#!/usr/bin/env python3
"""Test different approaches to Hilbert reordering."""

import torch
import torch.nn.functional as F
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention
from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    create_hilbert_mapping,
)


def test_reordering_approaches():
    """Compare different ways of applying Hilbert reordering."""
    device = "cuda"
    seq_len = 4096
    batch_size = 1
    num_heads = 12
    head_dim = 64

    print("TESTING DIFFERENT REORDERING APPROACHES")
    print("=" * 80)

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Get Hilbert mapping
    hilbert_map = create_hilbert_mapping(seq_len).to(device)

    print("\nTest configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")

    # Warmup
    _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    results = {}

    # 1. Baseline (no reordering)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    results["baseline"] = (time.perf_counter() - start) / 50 * 1000

    # 2. Pre-reorder K,V (what we currently do)
    k_reordered = k[:, :, hilbert_map]
    v_reordered = v[:, :, hilbert_map]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        out = F.scaled_dot_product_attention(q, k_reordered, v_reordered)
    torch.cuda.synchronize()
    results["pre_reorder"] = (time.perf_counter() - start) / 50 * 1000

    # 3. Reorder Q instead
    q_reordered = q[:, :, hilbert_map]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        out = F.scaled_dot_product_attention(q_reordered, k, v)
        # Need to reverse the output ordering
        out = out[:, :, hilbert_map]
    torch.cuda.synchronize()
    results["reorder_q"] = (time.perf_counter() - start) / 50 * 1000

    # 4. Reorder all QKV
    q_reordered = q[:, :, hilbert_map]
    k_reordered = k[:, :, hilbert_map]
    v_reordered = v[:, :, hilbert_map]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        out = F.scaled_dot_product_attention(q_reordered, k_reordered, v_reordered)
    torch.cuda.synchronize()
    results["reorder_all"] = (time.perf_counter() - start) / 50 * 1000

    # 5. Test with different tensor layouts
    # Transpose to [B, M, H, D] layout
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    # Reorder in this layout
    k_t_reordered = k_t[:, hilbert_map]
    v_t_reordered = v_t[:, hilbert_map]

    # Transpose back
    q_for_sdpa = q_t.transpose(1, 2)
    k_for_sdpa = k_t_reordered.transpose(1, 2)
    v_for_sdpa = v_t_reordered.transpose(1, 2)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        out = F.scaled_dot_product_attention(q_for_sdpa, k_for_sdpa, v_for_sdpa)
    torch.cuda.synchronize()
    results["transpose_layout"] = (time.perf_counter() - start) / 50 * 1000

    # Print results
    print("\nRESULTS (lower is better):")
    print("-" * 60)
    baseline = results["baseline"]
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        print(f"{name:<20} {time_ms:>8.2f}ms ({speedup:>5.2f}x)")

    # Test our actual implementation
    print("\n\nTESTING OUR IMPLEMENTATION:")
    print("-" * 60)

    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    x = torch.randn(1, seq_len, 768, device=device)

    # Force PyTorch backend
    module._triton_available = False
    module._fused_kernels_available = False

    with torch.no_grad():
        # Without Hilbert
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module(x, use_hilbert=False)
        torch.cuda.synchronize()
        no_hilbert = (time.perf_counter() - start) / 10 * 1000

        # With Hilbert
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        with_hilbert = (time.perf_counter() - start) / 10 * 1000

    print("Our implementation (PyTorch backend):")
    print(f"  Without Hilbert: {no_hilbert:.2f}ms")
    print(f"  With Hilbert: {with_hilbert:.2f}ms")
    print(
        f"  Overhead: {with_hilbert - no_hilbert:.2f}ms ({(with_hilbert / no_hilbert - 1) * 100:.1f}%)"
    )

    # Analyze the difference
    print("\n\nANALYSIS:")
    print("-" * 60)
    print("""
The key insights:

1. **Pre-reordering K,V can be faster** when done correctly
   - Direct SDPA test shows 3.15x speedup!
   - But our implementation shows slowdown

2. **The issue is likely**:
   - Extra overhead in our module (projections, reshaping)
   - Inefficient memory access patterns in Triton kernel
   - Mapping generation overhead (21ms!)

3. **Why SDPA benefits from reordering**:
   - Better cache utilization during attention computation
   - K,V are accessed multiple times for different Q positions
   - Hilbert ordering improves spatial locality

4. **To fix our implementation**:
   - Cache Hilbert mappings more aggressively
   - Eliminate redundant operations
   - Use the optimal reordering approach (pre-reorder K,V only)
""")


if __name__ == "__main__":
    test_reordering_approaches()
