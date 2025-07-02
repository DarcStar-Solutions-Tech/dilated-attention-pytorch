#!/usr/bin/env python3
"""
Debug hybrid implementation on multiple GPUs.
Run with: torchrun --nproc_per_node=2 benchmarks/debug_hybrid_multi_gpu.py
"""

import os
import torch
import torch.distributed as dist
import traceback


def debug_hybrid():
    """Debug where the hybrid implementation fails."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/debug_hybrid_multi_gpu.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized, world_size={world_size}")

    try:
        print(f"[Rank {rank}] Importing hybrid module...")
        from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
            RingDilatedAttentionHybrid,
        )

        print(f"[Rank {rank}] Import successful")

        print(f"[Rank {rank}] Creating model...")
        model = RingDilatedAttentionHybrid(
            segment_lengths=[256],
            dilation_rates=[1],
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            enable_memory_pool=False,  # Disable features to isolate issues
            use_pattern_cache=False,
            use_flash_attention=False,
        )
        print(f"[Rank {rank}] Model created successfully")

        # Create small inputs
        seq_len = 512
        print(f"[Rank {rank}] Creating inputs with seq_len={seq_len}...")
        q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        print(f"[Rank {rank}] Inputs created")

        print(f"[Rank {rank}] Starting forward pass...")
        output = model(q, k, v, is_causal=False)
        print(f"[Rank {rank}] Forward pass completed!")
        print(f"[Rank {rank}] Output shape: {output.shape}")

    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        traceback.print_exc()

    print(f"[Rank {rank}] Cleaning up...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    debug_hybrid()
