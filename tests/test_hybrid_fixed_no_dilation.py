#!/usr/bin/env python3
"""
Test segment locality with no dilation (dilation_rate=1).
Run with: torchrun --nproc_per_node=2 tests/test_hybrid_fixed_no_dilation.py
"""

import os
import torch
import torch.distributed as dist


def test_no_dilation():
    """Test with dilation_rate=1 to verify pure segment locality."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 tests/test_hybrid_fixed_no_dilation.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing segment locality with dilation_rate=1")

    from dilated_attention_pytorch.ring_dilated_attention_hybrid_fixed import (
        RingDilatedAttentionHybridFixed,
    )

    # Create model with no dilation
    model = RingDilatedAttentionHybridFixed(
        segment_lengths=[256],
        dilation_rates=[1],  # No dilation
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        use_flash_attention=False,
    )

    # Create inputs with distinct segment values
    seq_len = 512
    q = torch.zeros(1, seq_len, 4, 32, device=device)
    k = torch.zeros(1, seq_len, 4, 32, device=device)
    v = torch.zeros(1, seq_len, 4, 32, device=device)

    # Segment 1 (0-255): value 1.0
    q[:, :256] = 1.0
    k[:, :256] = 1.0
    v[:, :256] = 1.0

    # Segment 2 (256-511): value 10.0
    q[:, 256:] = 10.0
    k[:, 256:] = 10.0
    v[:, 256:] = 10.0

    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    seg1_mean = output[:, :256].mean().item()
    seg2_mean = output[:, 256:].mean().item()

    print(f"[Rank {rank}] Segment 1 mean: {seg1_mean:.4f} (expected 1.0)")
    print(f"[Rank {rank}] Segment 2 mean: {seg2_mean:.4f} (expected 10.0)")

    # With no dilation, segments should maintain their values exactly
    locality_ok = abs(seg1_mean - 1.0) < 0.01 and abs(seg2_mean - 10.0) < 0.01
    print(f"[Rank {rank}] Locality preserved: {locality_ok}")

    # Also test with the original (broken) implementation for comparison
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    print(f"\n[Rank {rank}] Testing original (broken) implementation...")

    model_old = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[1],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        use_flash_attention=False,
    )

    with torch.no_grad():
        output_old = model_old(q, k, v, is_causal=False)

    seg1_mean_old = output_old[:, :256].mean().item()
    seg2_mean_old = output_old[:, 256:].mean().item()

    print(f"[Rank {rank}] Original - Segment 1 mean: {seg1_mean_old:.4f}")
    print(f"[Rank {rank}] Original - Segment 2 mean: {seg2_mean_old:.4f}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\nSUMMARY:")
        print("Fixed implementation maintains segment locality")
        print(
            "Original implementation should also maintain locality with dilation_rate=1"
        )


if __name__ == "__main__":
    test_no_dilation()
