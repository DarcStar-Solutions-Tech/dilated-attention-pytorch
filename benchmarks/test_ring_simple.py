#!/usr/bin/env python3
"""
Simple test of ring attention to isolate the issue.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed as RingDilatedAttentionHilbertOptimized,
)


def test_ring_worker(rank, world_size):
    """Test ring attention on a single worker."""
    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Starting test")

    try:
        # Create model
        model = RingDilatedAttentionHilbertOptimized(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            dropout=0.0,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Small test tensor
        seq_len = 8192
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        print(f"[GPU {rank}] Running forward pass...")

        # Time the forward pass
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v)

        torch.cuda.synchronize()
        end = time.time()

        print(f"[GPU {rank}] Forward pass completed in {end - start:.3f}s")
        print(f"[GPU {rank}] Output shape: {output.shape}")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        dist.destroy_process_group()


def main():
    """Run simple ring attention test."""
    num_gpus = torch.cuda.device_count()
    print(f"Testing with {num_gpus} GPUs")

    if num_gpus < 2:
        print("Need at least 2 GPUs for ring attention")
        return

    # Test with 2 GPUs
    world_size = 2
    mp.spawn(test_ring_worker, args=(world_size,), nprocs=world_size, join=True)

    print("Test completed")


if __name__ == "__main__":
    main()
