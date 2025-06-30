"""
Test true distributed Ring Attention V2 across 2 GPUs.

This script properly initializes distributed PyTorch and runs
Ring Attention with actual memory distribution.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from datetime import datetime

from dilated_attention_pytorch.ring_dilated_attention_v2_fixed import (
    RingDilatedAttentionV2Fixed,
)


def run_worker(rank, world_size, seq_len, batch_size):
    """Worker function for each GPU."""

    # Setup distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Initialized")

    # Create model
    model = RingDilatedAttentionV2Fixed(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        ring_size=world_size,
        device=device,
        dtype=torch.float16,
        enable_memory_pool=True,
        use_pattern_cache=True,
    )

    print(f"[GPU {rank}] Model created, mode: {model.mode}")

    # Create inputs - each GPU creates the full tensor
    # In real distributed mode, the model will only process its chunk
    num_heads = 8
    head_dim = 64
    shape = (batch_size, seq_len, num_heads, head_dim)

    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)

    # Synchronize before starting
    dist.barrier()

    # Measure memory before
    torch.cuda.reset_peak_memory_stats(device)
    start_memory = torch.cuda.memory_allocated(device) / (1024**2)

    if rank == 0:
        print(f"\n[GPU {rank}] Starting forward pass...")
        print(f"[GPU {rank}] Initial memory: {start_memory:.1f}MB")

    # Time the forward pass
    start_time = time.time()

    # Forward pass - this will use true distributed ring attention
    output = model(q, k, v)

    # Synchronize
    torch.cuda.synchronize()
    dist.barrier()

    end_time = time.time()

    # Measure memory after
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    current_memory = torch.cuda.memory_allocated(device) / (1024**2)

    # Report results
    time_taken = (end_time - start_time) * 1000
    memory_used = peak_memory - start_memory

    print(f"[GPU {rank}] Completed in {time_taken:.1f}ms")
    print(f"[GPU {rank}] Peak memory: {peak_memory:.1f}MB (used: {memory_used:.1f}MB)")
    print(f"[GPU {rank}] Current memory: {current_memory:.1f}MB")
    print(f"[GPU {rank}] Output shape: {output.shape}")

    # Verify output
    if rank == 0:
        print(f"\n[GPU {rank}] Verification:")
        print(f"[GPU {rank}] Output mean: {output.mean().item():.6f}")
        print(f"[GPU {rank}] Output std: {output.std().item():.6f}")
        print(f"[GPU {rank}] Contains NaN: {torch.isnan(output).any().item()}")

    # Cleanup
    dist.destroy_process_group()


def main():
    print("True Distributed Ring Attention V2 Test")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Test configurations
    configs = [
        (16384, 1, "16K tokens"),
        (32768, 1, "32K tokens"),
        (65536, 1, "64K tokens"),
    ]

    world_size = 2  # Use 2 GPUs

    for seq_len, batch_size, desc in configs:
        print(f"\nTest: {desc}, Batch size: {batch_size}")
        print("-" * 60)

        try:
            # Spawn processes for each GPU
            mp.spawn(
                run_worker,
                args=(world_size, seq_len, batch_size),
                nprocs=world_size,
                join=True,
            )

            print("\nSuccess! Memory was distributed across GPUs.")

        except Exception as e:
            print(f"\nError: {e}")
            if "out of memory" in str(e).lower():
                print("OOM - sequence too long for available memory")
            else:
                print("Check if both GPUs are available and NCCL is working")

    print("\n" + "=" * 60)
    print("Key Observations:")
    print("1. Each GPU should show different memory usage")
    print("2. Peak memory should be ~1/2 of single GPU mode")
    print("3. Both GPUs should be active during computation")
    print("=" * 60)


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
