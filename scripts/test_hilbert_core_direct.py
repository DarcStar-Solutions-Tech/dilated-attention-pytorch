#!/usr/bin/env python3
"""
Direct test of HilbertAttentionCore with proper input format.
Tests the actual HilbertAttentionCore module directly.
"""

import torch
import torch.nn as nn
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dilated_attention_pytorch.kernels.hilbert_attention_core import (
    HilbertAttentionCore,
)
from src.dilated_attention_pytorch.utils.hilbert_attention_mixin import (
    HilbertAttentionMixin,
)


class StandardAttentionWithHilbert(nn.Module, HilbertAttentionMixin):
    """Standard attention with Hilbert ordering for comparison."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Setup Hilbert (ordering only)
        self.setup_hilbert_attention(
            hidden_dim=dim,
            num_heads=heads,
            use_hilbert_core=False,
        )

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        B, N, C = x.shape

        # Get Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, N, H, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if use_hilbert:
            # Apply Hilbert ordering
            indices = self.get_hilbert_indices(N, x.device)
            q = q[:, indices]
            k = k[:, indices]
            v = v[:, indices]

        # Compute attention
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous()  # [B, N, H, D]

        if use_hilbert:
            # Apply inverse ordering
            inverse_indices = torch.argsort(indices)
            out = out[:, inverse_indices]

        out = out.reshape(B, N, C)
        return self.out_proj(out)


def test_hilbert_core_directly():
    """Test HilbertAttentionCore directly with proper input format."""
    print("\n" + "=" * 80)
    print("Direct HilbertAttentionCore Test (fp32)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Test configuration
    batch_size = 2
    seq_len = 1024
    dim = 512
    heads = 8
    segment_size = 128

    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {dim}")
    print(f"  Heads: {heads}")
    print(f"  Segment size: {segment_size}")
    print("  Precision: fp32")

    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    x.requires_grad_(True)

    # Test 1: HilbertAttentionCore
    print("\n" + "-" * 60)
    print("1. HilbertAttentionCore (Triton kernels)")

    model_core = (
        HilbertAttentionCore(
            hidden_dim=dim,
            num_heads=heads,
            segment_size=segment_size,
            dilation_rate=1,
            dropout=0.0,
            use_custom_backward=True,
        )
        .to(device)
        .to(torch.float32)
    )

    # Initialize weights to match
    torch.manual_seed(42)
    model_core.qkv_proj.weight.data.normal_(0, 0.02)
    model_core.out_proj.weight.data.normal_(0, 0.02)

    # Forward pass
    torch.cuda.synchronize()
    start = time.perf_counter()
    out_core = model_core(x, use_hilbert=True)
    torch.cuda.synchronize()
    fwd_time_core = (time.perf_counter() - start) * 1000

    print(f"  Output shape: {out_core.shape}")
    print(f"  Forward time: {fwd_time_core:.2f}ms")

    # Backward pass
    x.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = out_core.sum()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time_core = (time.perf_counter() - start) * 1000

    grad_core = x.grad.clone()
    print(f"  Backward time: {bwd_time_core:.2f}ms")
    print(f"  Gradient norm: {grad_core.norm().item():.6f}")

    # Memory
    peak_mem_core = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak memory: {peak_mem_core:.2f}MB")

    # Test 2: Standard attention with Hilbert ordering
    print("\n" + "-" * 60)
    print("2. Standard Attention with Hilbert Ordering")

    model_standard = (
        StandardAttentionWithHilbert(
            dim=dim,
            heads=heads,
        )
        .to(device)
        .to(torch.float32)
    )

    # Copy weights to match
    model_standard.qkv.weight.data = model_core.qkv_proj.weight.data.clone()
    model_standard.out_proj.weight.data = model_core.out_proj.weight.data.clone()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    x.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    out_standard = model_standard(x, use_hilbert=True)
    torch.cuda.synchronize()
    fwd_time_standard = (time.perf_counter() - start) * 1000

    print(f"  Output shape: {out_standard.shape}")
    print(f"  Forward time: {fwd_time_standard:.2f}ms")

    # Backward pass
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = out_standard.sum()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time_standard = (time.perf_counter() - start) * 1000

    grad_standard = x.grad.clone()
    print(f"  Backward time: {bwd_time_standard:.2f}ms")
    print(f"  Gradient norm: {grad_standard.norm().item():.6f}")

    # Memory
    peak_mem_standard = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak memory: {peak_mem_standard:.2f}MB")

    # Compare outputs
    print("\n" + "-" * 60)
    print("Comparison:")

    output_diff = torch.abs(out_core - out_standard)
    print(f"  Output max diff: {output_diff.max().item():.6e}")
    print(f"  Output mean diff: {output_diff.mean().item():.6e}")
    print(
        f"  Output match: {'✓' if torch.allclose(out_core, out_standard, rtol=1e-3, atol=1e-3) else '✗'}"
    )

    grad_diff = torch.abs(grad_core - grad_standard)
    print(f"  Gradient max diff: {grad_diff.max().item():.6e}")
    print(f"  Gradient mean diff: {grad_diff.mean().item():.6e}")
    print(
        f"  Gradient match: {'✓' if torch.allclose(grad_core, grad_standard, rtol=1e-3, atol=1e-3) else '✗'}"
    )

    print("\nPerformance:")
    print(f"  Forward speedup: {fwd_time_standard / fwd_time_core:.2f}x")
    print(f"  Backward speedup: {bwd_time_standard / bwd_time_core:.2f}x")
    print(f"  Memory reduction: {(1 - peak_mem_core / peak_mem_standard) * 100:.1f}%")

    # Test without Hilbert ordering
    print("\n" + "-" * 60)
    print("3. Testing without Hilbert ordering")

    x.grad = None
    out_core_no_hilbert = model_core(x, use_hilbert=False)
    out_standard_no_hilbert = model_standard(x, use_hilbert=False)

    no_hilbert_diff = torch.abs(out_core_no_hilbert - out_standard_no_hilbert)
    print(f"  Output max diff: {no_hilbert_diff.max().item():.6e}")
    print(f"  Output mean diff: {no_hilbert_diff.mean().item():.6e}")
    print(
        f"  Match: {'✓' if torch.allclose(out_core_no_hilbert, out_standard_no_hilbert, rtol=1e-3, atol=1e-3) else '✗'}"
    )


def test_multi_gpu_functionality():
    """Test basic multi-GPU functionality."""
    if torch.cuda.device_count() < 2:
        print("\nSkipping multi-GPU test (need at least 2 GPUs)")
        return

    print("\n" + "=" * 80)
    print("Multi-GPU Functionality Test")
    print("=" * 80)

    # Create models on different GPUs
    dim = 256
    heads = 4
    batch_size = 2
    seq_len = 512

    model_gpu0 = HilbertAttentionCore(
        hidden_dim=dim,
        num_heads=heads,
        segment_size=128,
    ).to("cuda:0")

    model_gpu1 = HilbertAttentionCore(
        hidden_dim=dim,
        num_heads=heads,
        segment_size=128,
    ).to("cuda:1")

    # Copy weights
    model_gpu1.load_state_dict(model_gpu0.state_dict())

    # Test on different GPUs
    x0 = torch.randn(batch_size, seq_len, dim, device="cuda:0", dtype=torch.float32)
    x1 = x0.to("cuda:1")

    out0 = model_gpu0(x0)
    out1 = model_gpu1(x1)

    # Compare outputs
    out1_cpu = out1.to("cuda:0")
    diff = torch.abs(out0 - out1_cpu).max().item()

    print(f"  GPU 0 output shape: {out0.shape}")
    print(f"  GPU 1 output shape: {out1.shape}")
    print(f"  Max difference: {diff:.6e}")
    print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")


def main():
    """Run all tests."""
    print("HilbertAttentionCore Direct Testing")
    print("Verifying Triton kernel implementation")

    # Direct comparison test
    test_hilbert_core_directly()

    # Multi-GPU test
    test_multi_gpu_functionality()

    print("\n" + "=" * 80)
    print("Testing Complete!")


if __name__ == "__main__":
    main()
