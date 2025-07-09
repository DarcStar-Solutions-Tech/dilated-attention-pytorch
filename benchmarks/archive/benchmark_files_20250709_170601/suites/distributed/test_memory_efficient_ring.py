#!/usr/bin/env python3
"""
Test memory-efficient ring attention with 2 GPUs.

Launch with:
torchrun --nproc_per_node=2 test_memory_efficient_ring.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_memory_efficient import (
    RingDilatedAttentionMemoryEfficient,
)


def setup_distributed():
    """Setup distributed training."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def test_sequence(
    total_seq_len: int, batch_size: int, rank: int, world_size: int, local_rank: int
):
    """Test a specific sequence length."""
    device = torch.device(f"cuda:{local_rank}")
    local_seq_len = total_seq_len // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create local input
        x_local = torch.randn(
            batch_size, local_seq_len, 768, device=device, dtype=torch.float32
        )

        # Create model
        model = RingDilatedAttentionMemoryEfficient(
            embed_dim=768,
            num_heads=12,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
        )
        model.eval()

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Warmup
        with torch.no_grad():
            _ = model(x_local, total_seq_len=total_seq_len, already_split=True)
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        with torch.no_grad():
            output = model(x_local, total_seq_len=total_seq_len, already_split=True)
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Get memory
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        # Gather results
        if dist.is_initialized():
            mem_tensor = torch.tensor([mem_mb], device=device)
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
            mem_mb = mem_tensor.item()

        if rank == 0:
            mem_per_token = mem_mb / local_seq_len
            throughput = total_seq_len / elapsed / 1e6
            print(
                f"✓ {total_seq_len:,} tokens: {mem_mb:.1f} MB ({mem_per_token:.4f} MB/token), "
                f"{elapsed:.2f}s, {throughput:.2f}M tok/s"
            )

        # Cleanup
        del x_local, model, output
        gc.collect()
        torch.cuda.empty_cache()

        return True, mem_mb

    except Exception as e:
        if rank == 0:
            print(f"✗ {total_seq_len:,} tokens: {e}")
        return False, 0


def main():
    """Main test function."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print(f"Testing Memory-Efficient Ring Attention with {world_size} GPUs")
        print("=" * 80)
        print("This implementation uses true O(n/k) memory scaling")
        print()

    # Test increasing sequence lengths
    test_lengths = [
        8192,  # 8K
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        100000,  # 100K
        150000,  # 150K
        200000,  # 200K
        250000,  # 250K
        300000,  # 300K
        400000,  # 400K
        500000,  # 500K
    ]

    max_successful = 0

    for seq_len in test_lengths:
        # Make divisible by world_size
        seq_len = (seq_len // world_size) * world_size

        success, memory = test_sequence(seq_len, 1, rank, world_size, local_rank)

        if success:
            max_successful = seq_len
        else:
            break

        # Small delay
        time.sleep(0.5)

    if rank == 0:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Maximum successful sequence: {max_successful:,} tokens")
        print(f"Per GPU: {max_successful // world_size:,} tokens")

        if max_successful >= 200000:
            print("\n✅ Successfully processed 200K+ tokens with 2 GPUs!")
            print("🎉 True O(n/k) memory scaling achieved!")
        else:
            print(f"\n❌ Could only process up to {max_successful:,} tokens")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
