#!/usr/bin/env python3
"""
Test hybrid ring dilated attention on multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def test_gpu(rank, world_size, seq_len, batch_size, num_heads, head_dim):
    """Test on a single GPU in distributed setting."""
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Create model
    model = RingDilatedAttentionHybridOptimizedV2(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
    )

    # Create input
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Synchronize
    dist.barrier()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    # Time
    dist.barrier()
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            output = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    dist.barrier()

    forward_time = (time.time() - start) * 200  # ms per iteration

    # Calculate throughput
    total_tokens = batch_size * seq_len * world_size
    throughput = (total_tokens / forward_time) * 1000

    if rank == 0:
        print("\nHybrid Optimized V2 Results:")
        print(f"GPUs: {world_size}")
        print(f"Total tokens: {total_tokens}")
        print(f"Forward time: {forward_time:.2f} ms")
        print(f"Throughput: {throughput:,.0f} tokens/sec")
        print(f"Output shape: {output.shape}")

    cleanup()


def main():
    world_size = 2
    seq_len = 4096
    batch_size = 1
    num_heads = 8
    head_dim = 64

    if torch.cuda.device_count() < world_size:
        print(f"Need at least {world_size} GPUs")
        return

    print(f"Testing Hybrid Optimized V2 with {world_size} GPUs")
    print(f"Sequence length per GPU: {seq_len}")
    print(f"Total sequence length: {seq_len * world_size}")
    print("Using FP32 (correct for Pascal)")

    # Test single GPU first
    print("\nSingle GPU baseline:")
    device = torch.device("cuda:0")

    model = RingDilatedAttentionHybridOptimizedV2(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=torch.float32,
    )

    q = torch.randn(
        batch_size, seq_len * world_size, num_heads, head_dim, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()

    # Time
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    forward_time = (time.time() - start) * 200  # ms per iteration
    throughput = (batch_size * seq_len * world_size / forward_time) * 1000

    print(f"Single GPU forward time: {forward_time:.2f} ms")
    print(f"Single GPU throughput: {throughput:,.0f} tokens/sec")

    # Multi-GPU test
    print("\nMulti-GPU test:")
    mp.spawn(
        test_gpu,
        args=(world_size, seq_len, batch_size, num_heads, head_dim),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
