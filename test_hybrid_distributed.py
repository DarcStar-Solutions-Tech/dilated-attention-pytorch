#!/usr/bin/env python3
"""
Simple distributed test of Hybrid Ring Attention.
"""

import os
import torch
import torch.distributed as dist
import time


def init_distributed():
    """Initialize distributed if needed."""
    if not dist.is_initialized() and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def test_simple():
    """Simple test of hybrid ring attention."""
    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {rank}] Starting test on {device}, world_size={world_size}")

    try:
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
            RingDilatedAttentionHybridHilbert,
        )

        # Simple parameters
        batch_size = 1
        seq_len = 4096 * world_size  # Total sequence length
        num_heads = 8
        hidden_dim = 512
        head_dim = hidden_dim // num_heads

        print(f"[Rank {rank}] Testing Hilbert version...")

        model = RingDilatedAttentionHybridHilbert(
            segment_lengths=[1024],
            dilation_rates=[1],
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            enable_memory_pool=False,  # Disable for simplicity
            use_pattern_cache=False,
            use_hilbert=True,
        )

        print(f"[Rank {rank}] Creating tensors (seq_len={seq_len})...")

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        print(f"[Rank {rank}] Running forward pass...")

        if world_size > 1:
            dist.barrier()

        start = time.time()
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        if world_size > 1:
            dist.barrier()

        elapsed = (time.time() - start) * 1000

        print(f"[Rank {rank}] ✓ Forward pass complete in {elapsed:.1f} ms")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(f"[Rank {rank}] Output mean: {output.mean().item():.4f}")

    except Exception as e:
        print(f"[Rank {rank}] ✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if world_size > 1:
            dist.barrier()
            if rank == 0:
                print("\nCleaning up...")
            dist.destroy_process_group()


if __name__ == "__main__":
    test_simple()
