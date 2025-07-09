#!/usr/bin/env python3
"""
Find maximum tokens for 2 GPUs with true distributed setup.

Launch with:
torchrun --nproc_per_node=2 test_2gpu_max_tokens.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_correct import (
    RingDilatedAttentionHilbertOptimizedCorrect,
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


def test_tokens(total_seq_len: int, rank: int, world_size: int, local_rank: int):
    """Test specific token count."""
    device = torch.device(f"cuda:{local_rank}")
    local_seq_len = total_seq_len // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create local input
        x_local = torch.randn(1, local_seq_len, 768, device=device, dtype=torch.float32)

        # Create model with conservative settings
        model = RingDilatedAttentionHilbertOptimizedCorrect(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],  # Smaller segments
            dilation_rates=[1, 2],
            dropout=0.0,
            use_hilbert=True,
            device=device,
            dtype=torch.float32,
            memory_efficient=True,
        )
        model = model.to(device)
        model.eval()

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Forward pass
        with torch.no_grad():
            output = model(x_local, total_seq_len=total_seq_len, already_split=True)
            torch.cuda.synchronize()

        # Success
        if rank == 0:
            mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            print(f"✓ {total_seq_len:,} tokens: {mem_mb:.1f} MB per GPU")

        del x_local, model, output
        gc.collect()
        torch.cuda.empty_cache()

        return True

    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(f"✗ {total_seq_len:,} tokens: OOM")
        gc.collect()
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        if rank == 0:
            print(f"✗ {total_seq_len:,} tokens: Error - {e}")
        return False


def main():
    """Find maximum tokens with 2 GPUs."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"Finding maximum tokens with {world_size} GPUs...")
        print("-" * 60)

    # Start conservative and work up
    test_lengths = [
        50_000,  # 50K
        75_000,  # 75K
        100_000,  # 100K
        125_000,  # 125K
        150_000,  # 150K
        175_000,  # 175K
        200_000,  # 200K
        225_000,  # 225K
        250_000,  # 250K
    ]

    max_successful = 0

    for seq_len in test_lengths:
        # Ensure divisible by world_size
        seq_len = (seq_len // world_size) * world_size

        if dist.is_initialized():
            dist.barrier()

        success = test_tokens(seq_len, rank, world_size, local_rank)

        if success:
            max_successful = seq_len
        else:
            # Try to narrow down
            if max_successful > 0:
                # Binary search between last success and current failure
                low = max_successful
                high = seq_len

                while high - low > world_size * 1000:
                    mid = ((low + high) // 2 // world_size) * world_size

                    if dist.is_initialized():
                        dist.barrier()

                    if test_tokens(mid, rank, world_size, local_rank):
                        low = mid
                        max_successful = mid
                    else:
                        high = mid
            break

    if rank == 0:
        print("-" * 60)
        print(f"Maximum with {world_size} GPUs: {max_successful:,} tokens")
        print(f"Per GPU: {max_successful // world_size:,} tokens")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
