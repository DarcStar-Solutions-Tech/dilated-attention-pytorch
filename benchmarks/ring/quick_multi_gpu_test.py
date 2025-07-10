#!/usr/bin/env python3
"""
Quick multi-GPU test for ring attention memory scaling.
"""

import torch
import torch.distributed as dist
import os

from dilated_attention_pytorch import StandardRingAttention, RingAttentionConfig
from dilated_attention_pytorch.utils import get_optimal_dtype


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device properly
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n[Rank {rank}] Setup:")
    print(f"  World size: {world_size}")
    print(f"  Device: {device}")

    # Small test parameters for quick execution
    batch_size = 1
    seq_len = 1024  # Small sequence
    num_heads = 4
    head_dim = 32

    # Select optimal dtype
    dtype = get_optimal_dtype(device)
    print(f"  Dtype: {dtype}")

    # Create model
    config = RingAttentionConfig(
        segment_lengths=[256, 512],  # Small segments
        dilation_rates=[1, 2],
        dropout=0.0,
    )

    model = StandardRingAttention(config, device=device, dtype=dtype)

    # Check model state
    print(f"\n[Rank {rank}] Model state:")
    print(f"  is_distributed: {model.is_distributed}")
    print(f"  world_size: {model.world_size}")
    print(f"  rank: {model.rank}")

    # Create small inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Monitor memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated(device) / (1024**2)

    print(f"\n[Rank {rank}] Running forward pass...")
    print(f"  Memory before: {mem_before:.2f} MB")

    # Forward pass
    output = model(q, k, v)

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    mem_used = peak_mem - mem_before

    print(f"\n[Rank {rank}] Results:")
    print(f"  Output shape: {output.shape}")
    print(f"  Memory used: {mem_used:.2f} MB")

    # Calculate what we expect for ring attention
    local_seq = seq_len // world_size
    expected_mem_ratio = 1.0 / world_size
    print(f"  Expected local sequence: {local_seq}")
    print(f"  Expected memory ratio: {expected_mem_ratio:.2f}x")

    # Check if we're getting O(n/k) scaling
    full_attention_estimate = (seq_len * seq_len * num_heads * 2) / (
        1024**2
    )  # rough estimate
    print(f"  Full attention estimate: {full_attention_estimate:.2f} MB")
    print(
        f"  Actual/Expected ratio: {mem_used / (full_attention_estimate * expected_mem_ratio):.2f}x"
    )

    dist.barrier()
    print(f"\n[Rank {rank}] ✓ Test complete!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
