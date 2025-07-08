#!/usr/bin/env python3
"""
Verify HilbertAttentionCore integration with multi-GPU and fp32 precision.
Tests ring-based implementations without using all-gather for metrics.
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
from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core import (
    RingDilatedAttentionHilbertCore,
)


def setup_distributed():
    """Initialize distributed training environment."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")

        print(f"[Rank {rank}] Initialized on GPU {local_rank}")
        return rank, world_size, device
    else:
        # Single GPU fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device


def create_test_tensors(
    batch_size: int,
    seq_len: int,
    dim: int,
    heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test Q, K, V tensors with fp32 precision."""
    head_dim = dim // heads

    # Create tensors with fp32
    q = torch.randn(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, heads, head_dim, device=device, dtype=dtype)

    # Make them require gradients for backward pass testing
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    return q, k, v


def verify_outputs_match(
    out1: torch.Tensor,
    out2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    name1: str = "Output1",
    name2: str = "Output2",
) -> bool:
    """Verify that two outputs match within tolerance."""
    if out1.shape != out2.shape:
        print(f"❌ Shape mismatch: {name1}={out1.shape} vs {name2}={out2.shape}")
        return False

    # Compute differences
    abs_diff = torch.abs(out1 - out2)
    rel_diff = abs_diff / (torch.abs(out1) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Check if within tolerance
    matches = torch.allclose(out1, out2, rtol=rtol, atol=atol)

    print(f"  Comparison {name1} vs {name2}:")
    print(f"    Max absolute diff: {max_abs_diff:.6e}")
    print(f"    Max relative diff: {max_rel_diff:.6e}")
    print(f"    Mean absolute diff: {mean_abs_diff:.6e}")
    print(f"    Within tolerance (rtol={rtol}, atol={atol}): {'✓' if matches else '✗'}")

    return matches


def measure_memory_and_time(
    model: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_iterations: int = 5,
    warmup: int = 2,
) -> Tuple[float, float, float, torch.Tensor]:
    """Measure forward/backward time and memory without all-gather."""
    device = q.device

    # Warmup
    for _ in range(warmup):
        q.grad = k.grad = v.grad = None
        out = model(q, k, v)
        loss = out.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    # Time forward pass
    forward_times = []
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            out = model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        forward_times.append(time.perf_counter() - start)

    # Time backward pass
    backward_times = []
    for _ in range(num_iterations):
        q.grad = k.grad = v.grad = None

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        out = model(q, k, v)
        loss = out.sum()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        backward_times.append(time.perf_counter() - start)

    # Get memory stats
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        _ = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_used = peak_memory - start_memory
    else:
        memory_used = 0

    avg_forward = np.mean(forward_times) * 1000  # ms
    avg_backward = np.mean(backward_times) * 1000  # ms

    return avg_forward, avg_backward, memory_used, out


def test_single_gpu_verification():
    """Test implementations on single GPU with fp32."""
    print("\n" + "=" * 80)
    print("Single GPU Verification (fp32)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test configuration
    batch_size = 2
    seq_len = 2048
    dim = 512
    heads = 8
    segment_lengths = [512, 1024]
    dilation_rates = [1, 2]

    print("\nConfiguration:")
    print(f"  Device: {device}")
    print("  Precision: fp32")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {dim}")
    print(f"  Heads: {heads}")
    print(f"  Segments: {segment_lengths}")
    print(f"  Dilations: {dilation_rates}")

    # Create test tensors
    q, k, v = create_test_tensors(
        batch_size, seq_len, dim, heads, device, torch.float32
    )

    # Test 1: Original implementation
    print("\n" + "-" * 60)
    print("Testing Original RingDilatedAttentionHilbertOptimizedFixed")

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

    fwd_orig, bwd_orig, mem_orig, out_orig = measure_memory_and_time(
        model_original, q, k, v
    )

    print(f"  Forward: {fwd_orig:.2f}ms")
    print(f"  Backward: {bwd_orig:.2f}ms")
    print(f"  Memory: {mem_orig:.2f}MB")

    # Test 2: HilbertCore implementation
    print("\n" + "-" * 60)
    print("Testing RingDilatedAttentionHilbertCore")

    model_core = (
        RingDilatedAttentionHilbertCore(
            dim=dim,
            heads=heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=True,
            use_custom_backward=True,
        )
        .to(device)
        .to(torch.float32)
    )

    fwd_core, bwd_core, mem_core, out_core = measure_memory_and_time(
        model_core, q, k, v
    )

    print(f"  Forward: {fwd_core:.2f}ms")
    print(f"  Backward: {bwd_core:.2f}ms")
    print(f"  Memory: {mem_core:.2f}MB")

    # Verify outputs match
    print("\n" + "-" * 60)
    print("Verifying Output Equivalence:")
    verify_outputs_match(out_orig, out_core, name1="Original", name2="HilbertCore")

    # Performance comparison
    print("\n" + "-" * 60)
    print("Performance Comparison:")
    print(f"  Forward speedup: {fwd_orig / fwd_core:.2f}x")
    print(f"  Backward speedup: {bwd_orig / bwd_core:.2f}x")
    print(f"  Memory reduction: {(1 - mem_core / mem_orig) * 100:.1f}%")


def test_multi_gpu_ring_verification():
    """Test ring implementations on multi-GPU without all-gather."""
    rank, world_size, device = setup_distributed()

    if world_size == 1:
        print("\nSkipping multi-GPU test (only 1 GPU available)")
        return

    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Multi-GPU Ring Verification (fp32)")
    print(f"[Rank {rank}] " + "=" * 60)

    # Test configuration
    batch_size = 2
    seq_len_per_gpu = 1024
    total_seq_len = seq_len_per_gpu * world_size
    dim = 512
    heads = 8
    segment_lengths = [512, 1024]
    dilation_rates = [1, 2]

    if rank == 0:
        print("\nConfiguration:")
        print(f"  World size: {world_size}")
        print("  Precision: fp32")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length per GPU: {seq_len_per_gpu}")
        print(f"  Total sequence length: {total_seq_len}")
        print(f"  Hidden dim: {dim}")
        print(f"  Heads: {heads}")

    # Create local tensors (each GPU has its portion)
    q_local, k_local, v_local = create_test_tensors(
        batch_size, seq_len_per_gpu, dim, heads, device, torch.float32
    )

    # Test with ring_size > 1
    print(
        f"\n[Rank {rank}] Testing RingDilatedAttentionHilbertCore with ring_size={world_size}"
    )

    model_ring = (
        RingDilatedAttentionHilbertCore(
            dim=dim,
            heads=heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=world_size,
            use_hilbert=True,
            use_custom_backward=True,
        )
        .to(device)
        .to(torch.float32)
    )

    # Measure performance (local metrics only, no all-gather)
    fwd_time, bwd_time, mem_used, out_local = measure_memory_and_time(
        model_ring, q_local, k_local, v_local
    )

    print(f"[Rank {rank}] Local Metrics:")
    print(f"[Rank {rank}]   Forward: {fwd_time:.2f}ms")
    print(f"[Rank {rank}]   Backward: {bwd_time:.2f}ms")
    print(f"[Rank {rank}]   Memory: {mem_used:.2f}MB")
    print(f"[Rank {rank}]   Output shape: {out_local.shape}")

    # Verify gradient flow
    if q_local.grad is not None:
        grad_norm = q_local.grad.norm().item()
        print(f"[Rank {rank}]   Query gradient norm: {grad_norm:.6f}")

    dist.barrier()


def test_gradient_consistency():
    """Test gradient consistency between implementations."""
    print("\n" + "=" * 80)
    print("Gradient Consistency Test (fp32)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small test case for precise comparison
    batch_size = 1
    seq_len = 512
    dim = 256
    heads = 4
    segment_lengths = [256]
    dilation_rates = [1]

    # Create identical inputs
    torch.manual_seed(42)
    q_orig = torch.randn(
        batch_size,
        seq_len,
        heads,
        dim // heads,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    k_orig = torch.randn_like(q_orig)
    v_orig = torch.randn_like(q_orig)

    # Clone for second model
    q_core = q_orig.clone().detach().requires_grad_(True)
    k_core = k_orig.clone().detach().requires_grad_(True)
    v_core = v_orig.clone().detach().requires_grad_(True)

    # Create models
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

    model_core = (
        RingDilatedAttentionHilbertCore(
            dim=dim,
            heads=heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=True,
            use_custom_backward=True,
        )
        .to(device)
        .to(torch.float32)
    )

    # Forward and backward for original
    out_orig = model_original(q_orig, k_orig, v_orig)
    loss_orig = out_orig.sum()
    loss_orig.backward()

    # Forward and backward for core
    out_core = model_core(q_core, k_core, v_core)
    loss_core = out_core.sum()
    loss_core.backward()

    print("\nGradient Comparison:")
    print("-" * 40)

    # Compare gradients
    grad_match_q = verify_outputs_match(
        q_orig.grad,
        q_core.grad,
        rtol=1e-3,
        atol=1e-3,
        name1="Original Q.grad",
        name2="Core Q.grad",
    )

    grad_match_k = verify_outputs_match(
        k_orig.grad,
        k_core.grad,
        rtol=1e-3,
        atol=1e-3,
        name1="Original K.grad",
        name2="Core K.grad",
    )

    grad_match_v = verify_outputs_match(
        v_orig.grad,
        v_core.grad,
        rtol=1e-3,
        atol=1e-3,
        name1="Original V.grad",
        name2="Core V.grad",
    )

    if grad_match_q and grad_match_k and grad_match_v:
        print("\n✅ All gradients match within tolerance!")
    else:
        print("\n❌ Gradient mismatch detected!")


def main():
    """Run all verification tests."""
    print("HilbertAttentionCore Integration Verification")
    print("=" * 80)
    print("Testing with fp32 precision and ring-based metrics")

    # Test 1: Single GPU verification
    test_single_gpu_verification()

    # Test 2: Gradient consistency
    test_gradient_consistency()

    # Test 3: Multi-GPU ring verification (if available)
    if "LOCAL_RANK" in os.environ or torch.cuda.device_count() > 1:
        test_multi_gpu_ring_verification()
    else:
        print("\nNote: Set LOCAL_RANK or use torchrun for multi-GPU testing")

    print("\n" + "=" * 80)
    print("Verification Complete!")


if __name__ == "__main__":
    main()
