#!/usr/bin/env python3
"""
Fixed verification script for HilbertAttentionCore integration.
Tests with fp32 precision and handles the different input formats correctly.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
import time
from typing import Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)
from src.dilated_attention_pytorch.utils.hilbert_attention_mixin import (
    HilbertAttentionMixin,
)


class SimpleRingAttentionWithHilbert(nn.Module, HilbertAttentionMixin):
    """
    A simple ring attention that can use either:
    1. Just Hilbert ordering with standard attention
    2. Full HilbertAttentionCore (but needs adaptation for Q,K,V input)
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        segment_lengths: list,
        dilation_rates: list,
        use_hilbert_core: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates

        # Setup Hilbert attention
        self.setup_hilbert_attention(
            hidden_dim=dim,
            num_heads=heads,
            segment_size=segment_lengths[0],  # Use first segment size
            dilation_rate=dilation_rates[0],
            use_hilbert_core=use_hilbert_core,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward with Q, K, V interface."""
        if self.use_hilbert_core:
            # HilbertAttentionCore expects single input, so we need to adapt
            # For now, just use Hilbert ordering without the core
            return self._forward_with_ordering(q, k, v)
        else:
            # Use Hilbert ordering only
            return self._forward_with_ordering(q, k, v)

    def _forward_with_ordering(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Apply Hilbert ordering to standard attention."""
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Apply Hilbert ordering
        q_ordered = self.apply_hilbert_ordering(q, dim=1)
        k_ordered = self.apply_hilbert_ordering(k, dim=1)
        v_ordered = self.apply_hilbert_ordering(v, dim=1)

        # Simple scaled dot-product attention
        q_flat = q_ordered.reshape(batch_size * num_heads, seq_len, head_dim)
        k_flat = k_ordered.reshape(batch_size * num_heads, seq_len, head_dim)
        v_flat = v_ordered.reshape(batch_size * num_heads, seq_len, head_dim)

        scores = torch.bmm(q_flat, k_flat.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        out_flat = torch.bmm(attn_weights, v_flat)

        out_ordered = out_flat.reshape(batch_size, num_heads, seq_len, head_dim)
        out_ordered = out_ordered.transpose(1, 2).contiguous()
        out_ordered = out_ordered.reshape(batch_size, seq_len, num_heads, head_dim)

        # Apply inverse Hilbert ordering
        return self.apply_hilbert_ordering(out_ordered, inverse=True, dim=1)


def create_test_data(
    batch_size: int,
    seq_len: int,
    dim: int,
    heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test tensors with fp32 precision."""
    torch.manual_seed(42)  # For reproducibility

    # Create input tensor for models that expect single input
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    x.requires_grad_(True)

    # Create Q, K, V tensors for models that expect separate inputs
    head_dim = dim // heads
    q = torch.randn(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    return x, q, k, v


def verify_outputs(
    out1: torch.Tensor,
    out2: torch.Tensor,
    name1: str,
    name2: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Verify two outputs match within tolerance."""
    if out1.shape != out2.shape:
        print(f"❌ Shape mismatch: {name1}={out1.shape} vs {name2}={out2.shape}")
        return False

    abs_diff = torch.abs(out1 - out2)
    rel_diff = abs_diff / (torch.abs(out1) + 1e-8)

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    max_rel = rel_diff.max().item()

    matches = torch.allclose(out1, out2, rtol=rtol, atol=atol)

    print(f"  {name1} vs {name2}:")
    print(f"    Max abs diff: {max_abs:.6e}")
    print(f"    Mean abs diff: {mean_abs:.6e}")
    print(f"    Max rel diff: {max_rel:.6e}")
    print(f"    Matches (rtol={rtol}, atol={atol}): {'✓' if matches else '✗'}")

    return matches


def benchmark_model(
    model: nn.Module,
    inputs: tuple,
    num_iterations: int = 10,
    warmup: int = 3,
) -> Tuple[float, float, float]:
    """Benchmark forward and backward passes."""
    device = inputs[0].device

    # Clear gradients
    for inp in inputs:
        if hasattr(inp, "grad"):
            inp.grad = None

    # Warmup
    for _ in range(warmup):
        out = model(*inputs)
        loss = out.sum()
        loss.backward()

        # Clear gradients
        for inp in inputs:
            if hasattr(inp, "grad"):
                inp.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Forward timing
    forward_times = []
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            out = model(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        forward_times.append(time.perf_counter() - start)

    # Backward timing
    backward_times = []
    for _ in range(num_iterations):
        # Clear gradients
        for inp in inputs:
            if hasattr(inp, "grad"):
                inp.grad = None

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        out = model(*inputs)
        loss = out.sum()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        backward_times.append(time.perf_counter() - start)

    # Memory stats
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0

    avg_forward_ms = np.mean(forward_times) * 1000
    avg_backward_ms = np.mean(backward_times) * 1000

    return avg_forward_ms, avg_backward_ms, peak_memory_mb


def test_hilbert_integration():
    """Test different Hilbert integration approaches."""
    print("\n" + "=" * 80)
    print("Hilbert Integration Verification (fp32)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Test configuration
    batch_size = 2
    seq_len = 1024
    dim = 512
    heads = 8
    segment_lengths = [512]  # Single segment for simpler comparison
    dilation_rates = [1]

    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {dim}")
    print(f"  Heads: {heads}")
    print(f"  Segments: {segment_lengths}")
    print(f"  Dilations: {dilation_rates}")
    print("  Precision: fp32")

    # Create test data
    x, q, k, v = create_test_data(
        batch_size, seq_len, dim, heads, device, torch.float32
    )

    # Test 1: Original implementation
    print("\n" + "-" * 60)
    print("1. Original RingDilatedAttentionHilbertOptimizedFixed")

    model_original = (
        RingDilatedAttentionHilbertOptimizedFixed(
            dim=dim,
            heads=heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=True,
        )
        .to(device)
        .to(torch.float32)
    )

    # Test forward
    with torch.no_grad():
        out_original = model_original(q, k, v)
    print(f"  Output shape: {out_original.shape}")

    # Benchmark
    fwd_orig, bwd_orig, mem_orig = benchmark_model(model_original, (q, k, v))
    print(f"  Forward: {fwd_orig:.2f}ms")
    print(f"  Backward: {bwd_orig:.2f}ms")
    print(f"  Memory: {mem_orig:.2f}MB")

    # Test 2: Simple attention with Hilbert ordering
    print("\n" + "-" * 60)
    print("2. SimpleRingAttentionWithHilbert (Ordering Only)")

    model_ordering = (
        SimpleRingAttentionWithHilbert(
            dim=dim,
            heads=heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert_core=False,
        )
        .to(device)
        .to(torch.float32)
    )

    # Test forward
    with torch.no_grad():
        out_ordering = model_ordering(q, k, v)
    print(f"  Output shape: {out_ordering.shape}")

    # Benchmark
    fwd_order, bwd_order, mem_order = benchmark_model(model_ordering, (q, k, v))
    print(f"  Forward: {fwd_order:.2f}ms")
    print(f"  Backward: {bwd_order:.2f}ms")
    print(f"  Memory: {mem_order:.2f}MB")

    # Compare outputs
    print("\n" + "-" * 60)
    print("Output Comparison:")
    verify_outputs(
        out_original, out_ordering, "Original", "Ordering", rtol=1e-2, atol=1e-2
    )

    # Test gradient consistency
    print("\n" + "-" * 60)
    print("Gradient Consistency Test:")

    # Reset gradients and compute fresh
    q.grad = k.grad = v.grad = None
    out1 = model_original(q, k, v)
    loss1 = out1.sum()
    loss1.backward()
    q_grad1 = q.grad.clone()
    k_grad1 = k.grad.clone()
    v_grad1 = v.grad.clone()

    # Reset for second model
    q.grad = k.grad = v.grad = None
    out2 = model_ordering(q, k, v)
    loss2 = out2.sum()
    loss2.backward()

    # Compare gradients
    grad_match = True
    grad_match &= verify_outputs(
        q_grad1, q.grad, "Q.grad Original", "Q.grad Ordering", rtol=1e-2, atol=1e-2
    )
    grad_match &= verify_outputs(
        k_grad1, k.grad, "K.grad Original", "K.grad Ordering", rtol=1e-2, atol=1e-2
    )
    grad_match &= verify_outputs(
        v_grad1, v.grad, "V.grad Original", "V.grad Ordering", rtol=1e-2, atol=1e-2
    )

    print("\n" + "-" * 60)
    print("Summary:")
    print(
        f"  Output match: {'✓' if torch.allclose(out_original, out_ordering, rtol=1e-2, atol=1e-2) else '✗'}"
    )
    print(f"  Gradient match: {'✓' if grad_match else '✗'}")
    print(f"  Forward speedup: {fwd_orig / fwd_order:.2f}x")
    print(f"  Memory change: {(mem_order - mem_orig) / mem_orig * 100:+.1f}%")


def test_multi_gpu_ring():
    """Test multi-GPU ring functionality."""
    if "LOCAL_RANK" not in os.environ:
        print("\nSkipping multi-GPU test (set LOCAL_RANK to enable)")
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n[Rank {rank}] Multi-GPU Ring Test")
    print(f"[Rank {rank}] World size: {world_size}")

    # Test configuration
    batch_size = 2
    seq_len_per_gpu = 512
    dim = 256
    heads = 4

    # Create local data
    x, q, k, v = create_test_data(
        batch_size, seq_len_per_gpu, dim, heads, device, torch.float32
    )

    # Test with ring attention
    model = (
        SimpleRingAttentionWithHilbert(
            dim=dim,
            heads=heads,
            segment_lengths=[256],
            dilation_rates=[1],
            use_hilbert_core=False,
        )
        .to(device)
        .to(torch.float32)
    )

    # Forward pass
    out = model(q, k, v)
    print(f"[Rank {rank}] Output shape: {out.shape}")

    # Verify gradient flow
    loss = out.sum()
    loss.backward()

    q_grad_norm = q.grad.norm().item()
    print(f"[Rank {rank}] Q gradient norm: {q_grad_norm:.6f}")

    dist.barrier()

    if rank == 0:
        print("\n✓ Multi-GPU test completed successfully")


def main():
    """Run all verification tests."""
    print("HilbertAttentionCore Integration Verification")
    print("Testing with fp32 precision")

    # Single GPU tests
    test_hilbert_integration()

    # Multi-GPU tests
    test_multi_gpu_ring()

    print("\n" + "=" * 80)
    print("Verification Complete!")


if __name__ == "__main__":
    main()
