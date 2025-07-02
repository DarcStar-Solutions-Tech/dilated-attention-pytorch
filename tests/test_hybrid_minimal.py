#!/usr/bin/env python3
"""
Minimal test to debug hybrid implementation.
Run with: torchrun --nproc_per_node=2 tests/test_hybrid_minimal.py
"""

import os
import torch
import torch.distributed as dist


def main():
    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 tests/test_hybrid_minimal.py")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Starting test...")

    # Import model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    print(f"[Rank {rank}] Creating model...")

    # Create simple model
    model = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[1],  # No dilation for simplicity
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        enable_memory_pool=False,
        use_flash_attention=False,
    )

    print(f"[Rank {rank}] Model created")

    # Small inputs
    batch_size = 1
    seq_len = 512
    num_heads = 4
    head_dim = 32

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    print(f"[Rank {rank}] Inputs created, running forward...")

    try:
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        print(f"[Rank {rank}] Forward pass completed!")
        print(f"[Rank {rank}] Output shape: {output.shape}")

    except Exception as e:
        print(f"[Rank {rank}] Error: {str(e)}")
        import traceback

        traceback.print_exc()

    dist.barrier()
    print(f"[Rank {rank}] Test completed")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
