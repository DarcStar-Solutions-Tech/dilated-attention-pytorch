#!/usr/bin/env python3
"""
Debug Ring Dilated Attention SDPA initialization issue.

Launch with:
torchrun --nproc_per_node=2 test_ring_sdpa_debug.py
"""

import torch
import torch.distributed as dist
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_initialization():
    """Test RingDilatedAttentionSDPA initialization."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")
    print(f"Rank {rank}: Device set to {device}")

    try:
        # Import AFTER distributed init
        from dilated_attention_pytorch.ring_dilated_attention_sdpa import (
            RingDilatedAttentionSDPA,
        )

        print(f"Rank {rank}: Creating model...")

        # Create model with explicit device
        model = RingDilatedAttentionSDPA(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048],
            dilation_rates=[1],
            dropout=0.0,
            device=device,
            dtype=torch.float32,  # Explicit FP32
        )

        print(f"Rank {rank}: Model created successfully")
        print(f"Rank {rank}: Model device: {next(model.parameters()).device}")
        print(f"Rank {rank}: Model dtype: {next(model.parameters()).dtype}")

        # Test forward pass with small input
        x = torch.randn(1, 1024, 768, device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(x, already_split=True)
            torch.cuda.synchronize()

        print(f"Rank {rank}: Forward pass successful, output shape: {output.shape}")

        # Cleanup
        del model, x, output
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Rank {rank}: Error - {e}")
        import traceback

        traceback.print_exc()

    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"Rank {rank}: Test completed")


if __name__ == "__main__":
    test_initialization()
