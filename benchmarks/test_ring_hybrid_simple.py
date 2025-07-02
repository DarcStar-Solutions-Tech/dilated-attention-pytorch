#!/usr/bin/env python3
"""
Simple test for Hybrid Ring Attention to debug multi-GPU issues.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_hybrid_simple.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def test_simple():
    """Simple test to debug."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_hybrid_simple.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized")

    # Very simple test
    try:
        print(f"[Rank {rank}] Creating model...")

        model = RingDilatedAttentionHybrid(
            segment_lengths=[128],
            dilation_rates=[1],
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
            enable_memory_pool=False,  # Disable for simplicity
            use_flash_attention=False,  # Disable for simplicity
        )

        print(f"[Rank {rank}] Model created")

        # Small inputs
        seq_len = 256
        q = torch.ones(1, seq_len, 4, 32, device=device) * 0.1
        k = torch.ones(1, seq_len, 4, 32, device=device) * 0.1
        v = torch.ones(1, seq_len, 4, 32, device=device) * 0.1

        print(f"[Rank {rank}] Calling forward...")

        output = model(q, k, v, is_causal=False)

        print(f"[Rank {rank}] Forward completed!")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(f"[Rank {rank}] Output mean: {output.mean().item():.6f}")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    print(f"[Rank {rank}] Cleaning up...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    test_simple()
