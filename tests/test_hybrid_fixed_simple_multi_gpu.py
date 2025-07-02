#!/usr/bin/env python3
"""
Simple test for the fixed hybrid implementation on multiple GPUs.
Run with: torchrun --nproc_per_node=2 tests/test_hybrid_fixed_simple_multi_gpu.py
"""

import os
import torch
import torch.distributed as dist
import traceback


def test_fixed_implementation():
    """Simple test without all_gather_object."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 tests/test_hybrid_fixed_simple_multi_gpu.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Starting test with world_size={world_size}")

    try:
        # Import the fixed implementation
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_fixed import (
            RingDilatedAttentionHybridFixed,
        )

        print(f"[Rank {rank}] Successfully imported fixed implementation")

        # Create model
        model = RingDilatedAttentionHybridFixed(
            segment_lengths=[256],
            dilation_rates=[2],
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            use_flash_attention=False,
        )
        print(f"[Rank {rank}] Model created successfully")

        # Test 1: Basic forward pass
        seq_len = 512
        batch_size = 1
        num_heads = 4
        head_dim = 32

        # Create test inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 0.1
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 0.1
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 0.1

        print(f"[Rank {rank}] Running forward pass...")
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        print(f"[Rank {rank}] Forward pass complete!")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(
            f"[Rank {rank}] Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}"
        )
        print(f"[Rank {rank}] Has NaN: {torch.isnan(output).any().item()}")
        print(f"[Rank {rank}] Has Inf: {torch.isinf(output).any().item()}")

        # Test 2: Segment locality test
        print(f"\n[Rank {rank}] Testing segment locality...")

        # Create inputs with distinct segment values
        q_seg = torch.zeros(1, 512, 4, 32, device=device)
        k_seg = torch.zeros(1, 512, 4, 32, device=device)
        v_seg = torch.zeros(1, 512, 4, 32, device=device)

        # Segment 1 (0-255): value 1.0
        q_seg[:, :256] = 1.0
        k_seg[:, :256] = 1.0
        v_seg[:, :256] = 1.0

        # Segment 2 (256-511): value 10.0
        q_seg[:, 256:] = 10.0
        k_seg[:, 256:] = 10.0
        v_seg[:, 256:] = 10.0

        with torch.no_grad():
            output_seg = model(q_seg, k_seg, v_seg, is_causal=False)

        seg1_mean = output_seg[:, :256].mean().item()
        seg2_mean = output_seg[:, 256:].mean().item()

        print(f"[Rank {rank}] Segment 1 mean: {seg1_mean:.4f} (expected ~1.0)")
        print(f"[Rank {rank}] Segment 2 mean: {seg2_mean:.4f} (expected ~10.0)")

        locality_ok = abs(seg1_mean - 1.0) < 0.1 and abs(seg2_mean - 10.0) < 0.1
        print(f"[Rank {rank}] Locality preserved: {locality_ok}")

        # Synchronize before cleanup
        dist.barrier()

        print(f"\n[Rank {rank}] All tests completed successfully!")

    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        traceback.print_exc()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_fixed_implementation()
