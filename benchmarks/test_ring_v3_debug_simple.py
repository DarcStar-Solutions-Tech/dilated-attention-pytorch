#!/usr/bin/env python3
"""
Very simple debug test for Ring V3 multi-GPU.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_debug_simple.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def debug_simple():
    """Minimal test to debug multi-GPU issue."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_debug_simple.py"
        )
        return

    # Initialize distributed
    print(f"[PID {os.getpid()}] Initializing...")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized, world_size={world_size}")

    # Very simple test
    print(f"[Rank {rank}] Creating small model...")

    model = RingDilatedAttentionV3(
        segment_lengths=[64],  # Very small
        dilation_rates=[1],
        use_bucketed=False,  # No bucketing
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    print(f"[Rank {rank}] Model created")

    # Tiny inputs
    seq_len = 128  # Must be divisible by world_size
    q = torch.ones(1, seq_len, 2, 16, device=device) * 0.1
    k = torch.ones(1, seq_len, 2, 16, device=device) * 0.1
    v = torch.ones(1, seq_len, 2, 16, device=device) * 0.1

    print(f"[Rank {rank}] Inputs created, calling forward...")

    try:
        import time

        start = time.time()

        output = model(q, k, v, is_causal=False)

        elapsed = time.time() - start
        print(f"[Rank {rank}] Forward completed in {elapsed:.3f}s")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(f"[Rank {rank}] Output mean: {output.mean().item():.6f}")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    print(f"[Rank {rank}] Destroying process group...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    debug_simple()
