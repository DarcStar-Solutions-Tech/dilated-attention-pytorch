#!/usr/bin/env python3
"""
Test the fixed ring attention implementation with 2 GPUs.

Launch with:
torchrun --nproc_per_node=2 test_2gpu_fixed_v2.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed_v2 import (
    RingDilatedAttentionHilbertOptimizedFixedV2,
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


def test_sequence(total_seq_len: int, rank: int, world_size: int, local_rank: int):
    """Test a specific sequence length."""
    device = torch.device(f"cuda:{local_rank}")
    local_seq_len = total_seq_len // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create local input - ensure contiguous
        x_local = torch.randn(
            1, local_seq_len, 768, device=device, dtype=torch.float32
        ).contiguous()

        if rank == 0:
            print(f"\nTesting {total_seq_len:,} tokens ({local_seq_len:,} per GPU)")
            print(
                f"Input shape: {x_local.shape}, contiguous: {x_local.is_contiguous()}"
            )

        # Create model with fixed implementation
        model = RingDilatedAttentionHilbertOptimizedFixedV2(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            dropout=0.0,
            use_hilbert=False,  # Disable Hilbert initially
            device=device,
            dtype=torch.float32,
            memory_efficient=True,
        )

        model = model.to(device)
        model.eval()

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Warmup
        with torch.no_grad():
            _ = model(x_local, total_seq_len=total_seq_len, already_split=True)
            torch.cuda.synchronize()

        # Measure
        if dist.is_initialized():
            dist.barrier()

        start_time = time.time()
        with torch.no_grad():
            output = model(x_local, total_seq_len=total_seq_len, already_split=True)
            torch.cuda.synchronize()
        end_time = time.time()

        # Get memory
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        # Gather results
        if dist.is_initialized():
            mem_tensor = torch.tensor([mem_mb], device=device)
            time_tensor = torch.tensor([end_time - start_time], device=device)

            dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)

            mem_mb = mem_tensor.item()
            elapsed = time_tensor.item()
        else:
            elapsed = end_time - start_time

        if rank == 0:
            print("✓ SUCCESS!")
            print(f"  Memory: {mem_mb:.1f} MB per GPU")
            print(f"  Time: {elapsed:.3f} seconds")
            print(f"  Throughput: {total_seq_len / elapsed / 1e6:.3f}M tokens/sec")
            print(f"  Output shape: {output.shape}")

        del x_local, model, output
        gc.collect()
        torch.cuda.empty_cache()

        return True, mem_mb

    except Exception as e:
        if rank == 0:
            print(f"✗ ERROR: {e}")
            import traceback

            traceback.print_exc()
        return False, 0


def main():
    """Main test function."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print(f"Testing Fixed Ring Attention V2 with {world_size} GPUs")
        print("=" * 80)

    # Test increasing sequence lengths
    test_lengths = [
        8192,  # 8K
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        100000,  # 100K
        150000,  # 150K
        200000,  # 200K
    ]

    max_successful = 0

    for seq_len in test_lengths:
        # Make divisible by world_size
        seq_len = (seq_len // world_size) * world_size

        if dist.is_initialized():
            dist.barrier()

        success, memory = test_sequence(seq_len, rank, world_size, local_rank)

        if success:
            max_successful = seq_len
        else:
            if rank == 0:
                print(f"\nStopping at {seq_len:,} tokens due to error")
            break

        # Small delay between tests
        time.sleep(1)

    if rank == 0:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Maximum successful sequence: {max_successful:,} tokens")
        print(f"Per GPU: {max_successful // world_size:,} tokens")

        if max_successful >= 200000:
            print("\n✅ Successfully processed 200K+ tokens with 2 GPUs!")
        else:
            print(f"\n❌ Could only process up to {max_successful:,} tokens")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
