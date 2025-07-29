#!/usr/bin/env python3
"""Debug Hilbert performance issues."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def profile_attention():
    """Profile different parts of attention."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768, num_heads=12, segment_size=128, dilation_rate=1, dropout=0.0
    ).to(device)

    # Test input
    batch_size, seq_len = 2, 1024
    x = torch.randn(batch_size, seq_len, 768, device=device)

    print(f"Configuration: B={batch_size}, L={seq_len}, Segment={module.segment_size}")
    print(f"Triton available: {module._triton_available}")
    print(f"Cache size: {module._hilbert_cache.max_size}")
    print()

    # Check if Hilbert mapping is being cached
    print("Testing cache behavior:")

    # First call - should populate cache
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = module(x, use_hilbert=True)
    torch.cuda.synchronize()
    first_call = (time.perf_counter() - start) * 1000
    print(f"First call (cold cache): {first_call:.2f}ms")

    # Second call - should use cache
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = module(x, use_hilbert=True)
    torch.cuda.synchronize()
    second_call = (time.perf_counter() - start) * 1000
    print(f"Second call (warm cache): {second_call:.2f}ms")
    print(f"Cache hit improvement: {first_call / second_call:.2f}x")

    # Check different segment sizes
    print("\nTesting different segment sizes:")
    for segment_size in [64, 128, 256]:
        module.segment_size = segment_size

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) * 1000
        print(f"Segment={segment_size}: {time_ms:.2f}ms")

    # Test sparse vs dense attention
    print("\nComparing sparse patterns:")
    module.segment_size = 128

    # Dense attention (dilation_rate=1)
    module.dilation_rate = 1
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = module(x, use_hilbert=True)
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) * 1000

    # Sparse attention (dilation_rate=4)
    module.dilation_rate = 4
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = module(x, use_hilbert=True)
    torch.cuda.synchronize()
    sparse_time = (time.perf_counter() - start) * 1000

    print(f"Dense (dilation=1): {dense_time:.2f}ms")
    print(f"Sparse (dilation=4): {sparse_time:.2f}ms")
    print(f"Sparse speedup: {dense_time / sparse_time:.2f}x")


if __name__ == "__main__":
    profile_attention()
