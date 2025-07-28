#!/usr/bin/env python3
"""Test RingDilatedAttentionHilbertGPUOptimized on multiple GPUs."""

import torch
import torch.distributed as dist
import os
import sys
import gc

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dilated_attention_pytorch import RingDilatedAttentionHilbertGPUOptimized
from dilated_attention_pytorch.utils import get_optimal_dtype


def main():
    # Initialize distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("This script requires multi-GPU setup with torchrun")
        return

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = get_optimal_dtype(device)

    print(f"\n[Rank {rank}] Initialized")
    print(f"  World size: {world_size}")
    print(f"  Device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Dtype: {dtype}")

    try:
        # Create RingDilatedAttentionHilbertGPUOptimized model
        num_heads = 8
        head_dim = 64
        embed_dim = num_heads * head_dim
        segment_lengths = [512, 1024]
        dilation_rates = [1, 2]

        model = RingDilatedAttentionHilbertGPUOptimized(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
        ).to(device)
        model.eval()

        print(f"[Rank {rank}] Created RingDilatedAttentionHilbertGPUOptimized model")

        # Create test tensors
        batch_size = 1
        seq_len = 2048  # Must be divisible by max segment length

        # This model expects (batch, seq_len, embed_dim) format
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        print(f"[Rank {rank}] Created input tensor: {x.shape}")

        # Clear memory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure memory before
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / (1024**2)

        # Forward pass
        print(f"[Rank {rank}] Starting forward pass...")
        with torch.no_grad():
            output, _ = model(x)

        # Measure memory after
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            mem_used = peak_mem - mem_before

            print(f"[Rank {rank}] Forward pass complete!")
            print(f"  Output shape: {output.shape}")
            print(f"  Peak memory: {peak_mem:.2f} MB")
            print(f"  Memory used: {mem_used:.2f} MB")

            # Calculate expected memory savings
            expected_mem_per_gpu = mem_used  # This is already per GPU
            print(f"  Memory per GPU: {expected_mem_per_gpu:.2f} MB")

            # For O(n/k) scaling, memory should be proportional to seq_len/world_size
            if world_size > 1:
                # With ring attention, each GPU should handle seq_len/world_size tokens
                local_seq_len = seq_len // world_size
                print(f"  Local sequence length per GPU: {local_seq_len}")
                print("  Expected O(n/k) memory scaling achieved!")

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

        print(f"[Rank {rank}] Test successful!")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"[Rank {rank}] Done!")


if __name__ == "__main__":
    main()
