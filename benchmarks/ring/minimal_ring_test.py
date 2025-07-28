#!/usr/bin/env python3
"""Minimal ring attention test for multi-GPU."""

import torch
import torch.distributed as dist
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dilated_attention_pytorch import StandardRingAttention, RingAttentionConfig
from dilated_attention_pytorch.utils import get_optimal_dtype


def main():
    # Initialize distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("This script requires multi-GPU setup with torchrun")
        return

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = get_optimal_dtype(device)

    print(f"[Rank {rank}] Initialized - Device: {device}, World size: {world_size}")

    try:
        # Create a simple ring attention model
        config = RingAttentionConfig(
            segment_lengths=[512], dilation_rates=[1], dropout=0.0, ring_size=world_size
        )

        model = StandardRingAttention(config, device=device, dtype=dtype)
        model.eval()

        print(f"[Rank {rank}] Created StandardRingAttention model")

        # Create small test tensors
        batch_size = 1
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        print(f"[Rank {rank}] Created input tensors: {q.shape}")

        # Synchronize before forward
        dist.barrier()

        # Forward pass
        print(f"[Rank {rank}] Starting forward pass...")
        with torch.no_grad():
            output = model(q, k, v)

        print(f"[Rank {rank}] Forward pass complete! Output shape: {output.shape}")

        # Final sync
        dist.barrier()

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"[Rank {rank}] Done!")


if __name__ == "__main__":
    main()
