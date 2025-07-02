#!/usr/bin/env python3
"""
Test if V2 Collective is hanging.
Run with: torchrun --nproc_per_node=2 benchmarks/test_v2_collective_hang.py
"""

import os
import torch
import torch.distributed as dist
import time


def test_v2_hang():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_v2_collective_hang.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Starting test...")

    # Import V2
    print(f"[Rank {rank}] Importing V2 Collective...")
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )

    # Create model
    print(f"[Rank {rank}] Creating V2 model...")
    model = RingDilatedAttentionV2Collective(
        segment_lengths=[256],
        dilation_rates=[1],
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
    )
    print(f"[Rank {rank}] Model created")

    # Create minimal inputs
    seq_len = 512
    q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
    k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
    v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

    print(f"[Rank {rank}] Starting forward pass...")
    start = time.time()

    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"[Rank {rank}] Forward pass completed in {elapsed:.3f}s")
    print(f"[Rank {rank}] Output shape: {output.shape}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    test_v2_hang()
