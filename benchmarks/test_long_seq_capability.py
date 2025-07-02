#!/usr/bin/env python3
"""
Quick test of long sequence capability with multi-GPU ring attention.
"""

import torch
import torch.distributed as dist
import os

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    return rank, world_size


def test_sequence(seq_len, world_size, rank):
    """Test if we can handle a specific sequence length."""
    if seq_len % world_size != 0:
        seq_len = (seq_len // world_size) * world_size

    device = torch.device(f"cuda:{rank}")

    # Create model
    model = RingDilatedAttentionHybrid(
        segment_lengths=[4096, 8192],
        dilation_rates=[1, 2],
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    # Create small inputs to test
    batch_size = 1
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Clear memory
    torch.cuda.reset_peak_memory_stats(device)

    # Forward pass
    with torch.no_grad():
        _ = model(q, k, v, is_causal=False)

    # Get memory
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    return peak_mb


def main():
    rank, world_size = setup_distributed()

    if rank == 0:
        print(f"\n=== Testing Long Sequence Capability with {world_size} GPUs ===\n")

    # Test sequences based on GPU count
    if world_size == 1:
        test_seqs = [16384, 32768, 65536]
    elif world_size == 2:
        test_seqs = [32768, 65536, 131072, 262144]
    else:
        test_seqs = [65536, 131072, 262144, 524288]

    for seq_len in test_seqs:
        try:
            if world_size > 1:
                dist.barrier()

            peak_mb = test_sequence(seq_len, world_size, rank)

            if rank == 0:
                per_gpu = seq_len // world_size
                print(f"✓ {seq_len:,} tokens ({per_gpu:,}/GPU): {peak_mb:.0f} MB/GPU")

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"✗ {seq_len:,} tokens: OOM")
            break
        except Exception as e:
            if rank == 0:
                print(f"✗ {seq_len:,} tokens: {type(e).__name__}")
            break

    if rank == 0:
        print("\nBased on 8GB GPUs, theoretical maximum sequences:")
        print("  1 GPU: ~100K tokens")
        print("  2 GPUs: ~200K tokens")
        print("  4 GPUs: ~400K tokens")
        print("  8 GPUs: ~800K tokens")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
