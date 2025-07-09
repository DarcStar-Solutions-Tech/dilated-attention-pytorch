#!/usr/bin/env python3
"""
True distributed 2-GPU benchmark for ring attention.

Launch with:
torchrun --nproc_per_node=2 test_true_distributed_2gpu.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_correct import (
    RingDilatedAttentionHilbertOptimizedCorrect,
)
from dilated_attention_pytorch.utils.gpu_utils import get_gpu_info, get_optimal_dtype


def setup_distributed():
    """Setup distributed training."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_sequence_length(total_seq_len: int, batch_size: int = 1):
    """Test a specific sequence length with true distributed setup."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Get GPU info
    gpu_info = get_gpu_info(device)
    optimal_dtype = get_optimal_dtype(device)

    if rank == 0:
        print("=" * 80)
        print(f"Testing {total_seq_len:,} tokens with {world_size} GPUs")
        print("=" * 80)
        print(f"GPU: {gpu_info.name}")
        print(f"Memory per GPU: {gpu_info.total_memory_gb:.1f} GB")
        print(f"World size: {world_size}")
        print()

    local_seq_len = total_seq_len // world_size

    # Cleanup memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_mem = torch.cuda.memory_allocated(device) / 1024 / 1024

    try:
        # Create local input - each GPU gets its chunk
        x_local = torch.randn(
            batch_size, local_seq_len, 768, device=device, dtype=optimal_dtype
        )

        if rank == 0:
            print(f"Rank {rank}: Created input tensor ({local_seq_len:,} tokens)")

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Create model
        model = RingDilatedAttentionHilbertOptimizedCorrect(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[4096, 8192, 16384],
            dilation_rates=[1, 2, 4],
            dropout=0.0,
            use_hilbert=True,
            device=device,
            dtype=optimal_dtype,
            memory_efficient=True,
        )
        model = model.to(device)
        model.eval()

        if rank == 0:
            print("Created model on all ranks")

        # Synchronize before forward pass
        if dist.is_initialized():
            dist.barrier()

        # Test forward pass
        with torch.no_grad():
            if rank == 0:
                print("Running forward pass...")

            start_time = time.perf_counter()
            output = model(x_local, total_seq_len=total_seq_len, already_split=True)
            torch.cuda.synchronize()
            forward_time = time.perf_counter() - start_time

        # Get memory usage
        peak_mem = torch.cuda.memory_allocated(device) / 1024 / 1024
        memory_used = peak_mem - start_mem

        # Gather results from all ranks
        memory_tensor = torch.tensor([memory_used], device=device)
        time_tensor = torch.tensor([forward_time], device=device)

        if dist.is_initialized():
            dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(f"\n✓ SUCCESS - Processed {total_seq_len:,} tokens!")
            print(f"  Local sequence per GPU: {local_seq_len:,} tokens")
            print(f"  Memory per GPU: {memory_tensor.item():.1f} MB")
            print(f"  Memory per token: {memory_tensor.item() / local_seq_len:.4f} MB")
            print(f"  Forward pass time: {time_tensor.item():.3f} seconds")
            print(
                f"  Throughput: {total_seq_len / time_tensor.item() / 1e6:.3f}M tokens/sec"
            )

        # Cleanup
        del x_local, model, output
        gc.collect()
        torch.cuda.empty_cache()

        return True, memory_tensor.item()

    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(
                f"\n✗ OOM - Cannot process {total_seq_len:,} tokens with {world_size} GPUs"
            )
        return False, float("inf")
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error: {e}")
            import traceback

            traceback.print_exc()
        return False, 0
    finally:
        torch.cuda.synchronize()


def find_max_tokens():
    """Find maximum tokens processable with 2 GPUs."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        print(f"\nFinding maximum sequence length for {world_size} GPUs...")
        print("-" * 80)

    # Test specific lengths
    test_lengths = [
        100_000,  # 100K
        150_000,  # 150K
        200_000,  # 200K
        250_000,  # 250K
        300_000,  # 300K
        400_000,  # 400K
        500_000,  # 500K
    ]

    max_successful = 0

    for seq_len in test_lengths:
        # Make sure it's divisible by world_size
        seq_len = (seq_len // world_size) * world_size

        success, memory = test_sequence_length(seq_len)

        if success:
            max_successful = seq_len
        else:
            break

        # Small delay between tests
        if dist.is_initialized():
            dist.barrier()
        time.sleep(2)

    if rank == 0:
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(
            f"Maximum sequence length with {world_size} GPUs: {max_successful:,} tokens"
        )
        print(f"Per GPU: {max_successful // world_size:,} tokens")


def main():
    """Main entry point."""
    try:
        # First test 200K specifically
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("Testing 200K tokens with true distributed setup...")
            print(f"Launch command: torchrun --nproc_per_node=2 {__file__}")
            print()

        # Test 200K
        success, memory = test_sequence_length(200_000)

        if dist.is_initialized():
            dist.barrier()

        # Then find maximum
        find_max_tokens()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
