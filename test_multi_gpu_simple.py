#!/usr/bin/env python3
"""
Simple test to verify multi-GPU functionality.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dilated_attention_pytorch.ring_dilated_attention_simple_triton import (
    RingDilatedAttentionSimpleTriton,
)


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def test_gpu(rank, world_size):
    """Test on a single GPU in distributed setting."""
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing on {device}")

    # Simple test
    seq_len = 4096
    batch_size = 1
    num_heads = 4
    head_dim = 32

    # Create model
    model = RingDilatedAttentionSimpleTriton(
        segment_lengths=[1024],
        dilation_rates=[1],  # No dilation for simplicity
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        use_hilbert=False,
    )

    # Create input
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"[Rank {rank}] Input shape: {q.shape}")

    # Forward pass
    try:
        output = model(q, k, v, is_causal=False)
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(f"[Rank {rank}] Success!")
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")

    cleanup()


def main():
    world_size = 2

    if torch.cuda.device_count() < world_size:
        print(f"Need at least {world_size} GPUs")
        return

    print(f"Testing with {world_size} GPUs")
    print("=" * 50)

    # Test single GPU first
    print("\nSingle GPU test:")
    device = torch.device("cuda:0")
    model = RingDilatedAttentionSimpleTriton(
        segment_lengths=[1024],
        dilation_rates=[1],
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=torch.float32,
        use_hilbert=False,
    )

    q = torch.randn(1, 4096, 4, 32, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = model(q, k, v, is_causal=False)
    print(f"Single GPU output shape: {output.shape}")
    print("Single GPU test passed!")

    # Multi-GPU test
    print("\nMulti-GPU test:")
    mp.spawn(test_gpu, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
