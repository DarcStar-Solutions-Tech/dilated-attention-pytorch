#!/usr/bin/env python3
"""
Test script to demonstrate HilbertAttentionCore integration with existing classes.
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)
from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core import (
    RingDilatedAttentionHilbertCore,
)
from src.dilated_attention_pytorch.utils.hilbert_attention_mixin import (
    HilbertAttentionMixin,
)


class SimpleAttentionWithHilbert(nn.Module, HilbertAttentionMixin):
    """Example of adding Hilbert optimization to a simple attention class."""

    def __init__(self, dim: int, heads: int, use_hilbert_core: bool = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # Standard projections
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        # Setup Hilbert attention
        self.setup_hilbert_attention(
            hidden_dim=dim,
            num_heads=heads,
            segment_size=128,
            use_hilbert_core=use_hilbert_core,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Get Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Use Hilbert-optimized attention
        out = self.compute_hilbert_attention(q, k, v, use_hilbert_ordering=True)

        # Reshape and project
        out = out.reshape(B, N, C)
        return self.out_proj(out)


def benchmark_implementation(model, x, name, num_iterations=10):
    """Benchmark forward and backward passes."""
    device = x.device

    # Warmup
    for _ in range(3):
        x.grad = None
        out = model(x)
        loss = out.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time forward pass
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()
    forward_time = (end - start) / num_iterations * 1000

    # Time backward pass
    x.requires_grad_(True)
    start = time.perf_counter()
    for _ in range(num_iterations):
        x.grad = None
        out = model(x)
        loss = out.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()
    backward_time = (end - start) / num_iterations * 1000

    # Memory usage
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    )

    return forward_time, backward_time, peak_memory


def test_integration():
    """Test different integration approaches."""
    print("Testing HilbertAttentionCore Integration")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 1024
    dim = 512
    heads = 8

    # Test input - 3D for ring attention classes
    x_3d = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

    # Test input - 4D for Q,K,V interface
    head_dim = dim // heads
    q = torch.randn(batch_size, seq_len, heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, heads, head_dim, device=device)

    print("\nTest Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Hidden Dimension: {dim}")
    print(f"  Number of Heads: {heads}")

    # Test 1: Original implementation
    print("\n" + "=" * 60)
    print("Test 1: Original RingDilatedAttentionHilbertOptimizedFixed")
    try:
        model_original = RingDilatedAttentionHilbertOptimizedFixed(
            dim=dim,
            heads=heads,
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
        ).to(device)

        # Test forward pass
        with torch.no_grad():
            out = model_original(q, k, v)
        print(f"✓ Forward pass successful: {out.shape}")

        # Benchmark
        fwd, bwd, mem = benchmark_implementation(model_original, x_3d, "Original")
        print(
            f"✓ Performance: {fwd:.2f}ms forward, {bwd:.2f}ms backward, {mem:.2f}MB peak"
        )

    except Exception as e:
        print(f"✗ Failed: {str(e)}")

    # Test 2: New HilbertCore integration
    print("\n" + "=" * 60)
    print("Test 2: RingDilatedAttentionHilbertCore (New)")
    try:
        model_core = RingDilatedAttentionHilbertCore(
            dim=dim,
            heads=heads,
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            use_custom_backward=True,
        ).to(device)

        # Test forward pass
        with torch.no_grad():
            out = model_core(q, k, v)
        print(f"✓ Forward pass successful: {out.shape}")

        # Benchmark
        fwd, bwd, mem = benchmark_implementation(model_core, x_3d, "HilbertCore")
        print(
            f"✓ Performance: {fwd:.2f}ms forward, {bwd:.2f}ms backward, {mem:.2f}MB peak"
        )

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Test 3: Simple attention with mixin (ordering only)
    print("\n" + "=" * 60)
    print("Test 3: SimpleAttentionWithHilbert (Mixin - Ordering Only)")
    try:
        model_mixin_order = SimpleAttentionWithHilbert(
            dim=dim,
            heads=heads,
            use_hilbert_core=False,  # Just ordering
        ).to(device)

        # Test forward pass
        with torch.no_grad():
            out = model_mixin_order(x_3d)
        print(f"✓ Forward pass successful: {out.shape}")

        # Benchmark
        fwd, bwd, mem = benchmark_implementation(
            model_mixin_order, x_3d, "Mixin Ordering"
        )
        print(
            f"✓ Performance: {fwd:.2f}ms forward, {bwd:.2f}ms backward, {mem:.2f}MB peak"
        )

    except Exception as e:
        print(f"✗ Failed: {str(e)}")

    # Test 4: Simple attention with full HilbertCore
    print("\n" + "=" * 60)
    print("Test 4: SimpleAttentionWithHilbert (Mixin - Full HilbertCore)")
    try:
        model_mixin_core = SimpleAttentionWithHilbert(
            dim=dim,
            heads=heads,
            use_hilbert_core=True,  # Full Triton implementation
        ).to(device)

        # Test forward pass
        with torch.no_grad():
            out = model_mixin_core(x_3d)
        print(f"✓ Forward pass successful: {out.shape}")

        # Benchmark
        fwd, bwd, mem = benchmark_implementation(
            model_mixin_core, x_3d, "Mixin HilbertCore"
        )
        print(
            f"✓ Performance: {fwd:.2f}ms forward, {bwd:.2f}ms backward, {mem:.2f}MB peak"
        )

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Integration Testing Complete!")


if __name__ == "__main__":
    test_integration()
