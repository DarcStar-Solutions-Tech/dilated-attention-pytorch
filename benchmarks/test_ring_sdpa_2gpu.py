#!/usr/bin/env python3
"""
Test Ring Dilated Attention with SDPA on 2 GPUs.

Launch with:
torchrun --nproc_per_node=2 test_ring_sdpa_2gpu.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_sdpa import (
    RingDilatedAttentionSDPA,
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
    total_seq_len: int,
    segment_lengths: list,
    dilation_rates: list,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[bool, float, float]:
    """Test a specific sequence length."""
    device = torch.device(f"cuda:{local_rank}")
    local_seq_len = total_seq_len // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create local input - let model determine optimal dtype
        x_local = torch.randn(
            1,
            local_seq_len,
            768,
            device=device,
            dtype=torch.float32,  # Start with FP32
        )

        # Create model
        model = RingDilatedAttentionSDPA(
            embed_dim=768,
            num_heads=12,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            device=device,
            # dtype will be auto-selected by get_optimal_dtype
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

        # Cleanup
        del x_local, model, output
        gc.collect()
        torch.cuda.empty_cache()

        return True, mem_mb, elapsed

    except Exception as e:
        if rank == 0:
            print(f"Error: {e}")
        return False, 0, 0


def main():
    """Main test function."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print(f"Testing Ring Dilated Attention with SDPA on {world_size} GPUs")
        print("=" * 80)
        print("Using PyTorch's scaled_dot_product_attention for memory efficiency")
        print("Testing with different segment lengths and dilation rates")
        print()

    # Test configurations
    # Using dilated attention significantly reduces computation
    test_configs = [
        # (seq_len, segment_lengths, dilation_rates, label)
        (16384, [2048, 4096], [1, 2], "16K - segments 2K/4K, dilation 1/2"),
        (32768, [2048, 4096], [1, 2], "32K - segments 2K/4K, dilation 1/2"),
        (
            65536,
            [2048, 4096, 8192],
            [1, 2, 4],
            "64K - segments 2K/4K/8K, dilation 1/2/4",
        ),
        (
            100000,
            [2048, 4096, 8192],
            [1, 2, 4],
            "100K - segments 2K/4K/8K, dilation 1/2/4",
        ),
        (
            150000,
            [2048, 4096, 8192],
            [1, 4, 8],
            "150K - segments 2K/4K/8K, dilation 1/4/8",
        ),
        (
            200000,
            [2048, 4096, 8192],
            [1, 4, 8],
            "200K - segments 2K/4K/8K, dilation 1/4/8",
        ),
        (250000, [2048, 4096], [2, 8], "250K - segments 2K/4K, dilation 2/8"),
        (300000, [2048, 4096], [4, 16], "300K - segments 2K/4K, dilation 4/16"),
    ]

    max_successful = 0

    for seq_len, seg_lengths, dil_rates, label in test_configs:
        # Make divisible by world_size
        seq_len = (seq_len // world_size) * world_size

        if dist.is_initialized():
            dist.barrier()

        success, memory, elapsed = test_sequence(
            seq_len, seg_lengths, dil_rates, rank, world_size, local_rank
        )

        if rank == 0:
            if success:
                local_seq = seq_len // world_size
                mem_per_token = memory / local_seq
                throughput = seq_len / elapsed / 1e6
                print(f"✓ {label}")
                print(f"  Total: {seq_len:,} tokens ({local_seq:,} per GPU)")
                print(f"  Memory: {memory:.1f} MB ({mem_per_token:.4f} MB/token)")
                print(f"  Time: {elapsed:.2f}s, {throughput:.2f}M tok/s")
                print()
                max_successful = seq_len
            else:
                print(f"✗ {label} - Failed")
                break

        # Small delay
        time.sleep(0.5)

    if rank == 0:
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Maximum successful sequence: {max_successful:,} tokens")
        print(f"Per GPU: {max_successful // world_size:,} tokens")

        if max_successful >= 200000:
            print("\n✅ Successfully processed 200K+ tokens with 2 GPUs!")
            print("🎉 Using SDPA + Dilated Attention achieved the goal!")
        else:
            print(f"\n❌ Could only process up to {max_successful:,} tokens")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
