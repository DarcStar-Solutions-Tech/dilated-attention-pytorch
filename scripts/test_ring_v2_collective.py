#!/usr/bin/env python3
"""
Quick test script to verify RingDilatedAttentionV2Collective works correctly
in both single and multi-GPU settings.
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def test_ring_attention():
    """Test basic functionality of RingDilatedAttentionV2Collective."""

    # Setup distributed if available
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {rank}] Testing on {device}, world_size={world_size}")

    # Test parameters
    batch_size = 2
    seq_length = 8192
    embed_dim = 768
    num_heads = 12
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]
    ring_size = min(2, world_size)  # Use at most 2 for this test

    try:
        # Create model
        model = RingDilatedAttentionV2Collective(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=ring_size,
            use_flash_attention=True,
            device=device,
            dtype=torch.float16,
        ).to(device)

        print(f"[Rank {rank}] Model created successfully")

        # Create input
        head_dim = embed_dim // num_heads
        query = torch.randn(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        key = value = query

        print(f"[Rank {rank}] Input shape: {query.shape}")

        # Forward pass
        output = model(query, key, value)
        print(f"[Rank {rank}] Output shape: {output.shape}")

        # Backward pass
        loss = output.sum()
        loss.backward()
        print(f"[Rank {rank}] Backward pass completed")

        # Check gradients
        if query.grad is not None:
            grad_norm = query.grad.norm().item()
            print(f"[Rank {rank}] Gradient norm: {grad_norm:.4f}")

        # Memory stats
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(
                f"[Rank {rank}] Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

        print(f"[Rank {rank}] ✓ Test passed!")

    except Exception as e:
        print(f"[Rank {rank}] ✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        if is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    test_ring_attention()
