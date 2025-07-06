#!/usr/bin/env python3
"""
Test the properly implemented Hilbert optimization.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_proper import (
    RingDilatedAttentionHilbertOptimizedProper,
)


def test_single_gpu():
    """Test on single GPU first."""
    print("Testing single GPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = RingDilatedAttentionHilbertOptimizedProper(
        segment_lengths=[2048, 4096],
        dilation_rates=[8, 16],
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        ring_size=1,
    )

    # Test tensor
    seq_len = 8192
    batch_size = 1
    num_heads = 8
    head_dim = 64

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"Input shapes: {q.shape}")

    # Warmup
    with torch.no_grad():
        _ = model(q, k, v)

    # Time forward pass
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    with torch.no_grad():
        output = model(q, k, v)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()

    print(f"Forward pass time: {end - start:.3f}s")
    print(f"Output shape: {output.shape}")
    print("Single GPU test passed!\n")

    return True


def test_multi_gpu_worker(rank, world_size):
    """Test multi-GPU worker."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Starting test")

    try:
        # Create model
        model = RingDilatedAttentionHilbertOptimizedProper(
            segment_lengths=[2048, 4096],
            dilation_rates=[8, 16],
            dropout=0.0,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Test tensor
        seq_len = 8192
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Synchronize
        dist.barrier()

        # Time forward pass
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v)

        torch.cuda.synchronize()
        end = time.time()

        if rank == 0:
            print(f"[GPU {rank}] Forward pass time: {end - start:.3f}s")
            print(f"[GPU {rank}] Output shape: {output.shape}")
            print("Multi-GPU test passed!")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        dist.destroy_process_group()


def main():
    """Run tests."""
    print("=" * 60)
    print("Testing Properly Implemented Hilbert Optimization")
    print("=" * 60)

    # Test single GPU first
    if test_single_gpu():
        # Test multi-GPU if available
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            print("\nTesting multi-GPU...")
            world_size = 2
            mp.spawn(
                test_multi_gpu_worker, args=(world_size,), nprocs=world_size, join=True
            )
        else:
            print("Skipping multi-GPU test (need at least 2 GPUs)")

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
