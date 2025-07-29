#!/usr/bin/env python3
"""Diagnose why Hilbert performance is poor."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def benchmark_reordering_overhead():
    """Measure the overhead of tensor reordering."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test parameters
    batch_size = 2
    num_heads = 12
    seq_len = 2048
    head_dim = 64

    # Create tensors
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Create a random permutation (simulating Hilbert)
    perm = torch.randperm(seq_len, device=device)

    # Warmup
    for _ in range(10):
        _ = k[:, :, perm]

    # Benchmark reordering
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = k[:, :, perm]
        _ = v[:, :, perm]
    torch.cuda.synchronize()
    reorder_time = (time.perf_counter() - start) / 100 * 1000

    print(f"Reordering overhead for seq_len={seq_len}: {reorder_time:.2f}ms")

    # Test if memory access pattern matters
    print("\nTesting memory access patterns:")

    # Sequential access
    seq_indices = torch.arange(0, seq_len, 4, device=device)  # Every 4th element
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = k[:, :, seq_indices]
    torch.cuda.synchronize()
    seq_time = (time.perf_counter() - start) / 100 * 1000

    # Random access
    rand_indices = torch.randperm(seq_len, device=device)[: len(seq_indices)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = k[:, :, rand_indices]
    torch.cuda.synchronize()
    rand_time = (time.perf_counter() - start) / 100 * 1000

    print(f"  Sequential sparse access: {seq_time:.2f}ms")
    print(f"  Random sparse access: {rand_time:.2f}ms")
    print(f"  Random slowdown: {rand_time / seq_time:.2f}x")


def test_hilbert_benefit_threshold():
    """Find where Hilbert becomes beneficial."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nTesting Hilbert benefit at different sequence lengths:")
    print("=" * 60)

    from dilated_attention_pytorch.kernels import UnifiedHilbertAttention

    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        # Create module
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=1,  # Test with standard attention first
            dropout=0.0,
        ).to(device)

        # Create input
        x = torch.randn(1, seq_len, 768, device=device)

        # Warmup
        with torch.no_grad():
            _ = module(x, use_hilbert=False)
            _ = module(x, use_hilbert=True)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x, use_hilbert=False)
        torch.cuda.synchronize()
        std_time = (time.perf_counter() - start) * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        hilbert_time = (time.perf_counter() - start) * 1000

        speedup = std_time / hilbert_time
        print(
            f"  Seq={seq_len}: Standard={std_time:.1f}ms, Hilbert={hilbert_time:.1f}ms, Speedup={speedup:.2f}x"
        )

        if seq_len >= 8192:
            break  # Avoid OOM


if __name__ == "__main__":
    benchmark_reordering_overhead()
    test_hilbert_benefit_threshold()
