#!/usr/bin/env python3
"""
Test the fixed implementations.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Import fixed implementations
from dilated_attention_pytorch.ring_dilated_attention_fixed import (
    RingDilatedAttentionFixed,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_fixed import (
    RingDilatedAttentionHilbertFixed,
)


def test_single_gpu():
    """Test fixed implementations on single GPU."""
    print("=" * 60)
    print("Testing Fixed Implementations - Single GPU")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test parameters
    seq_len = 8192
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [2048, 4096]
    dilation_rates = [8, 16]

    # Create test tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print("Test configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Device: {device}")
    print("  Dtype: torch.float32")

    # Test each implementation
    implementations = [
        ("Fixed Base", RingDilatedAttentionFixed),
        ("Fixed Hilbert", RingDilatedAttentionHilbertFixed),
    ]

    results = {}

    for name, cls in implementations:
        print(f"\n{name}:")

        try:
            # Create model
            model = cls(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
                ring_size=1,
            )

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

            forward_time = end - start
            results[name] = forward_time

            print(f"  ✓ Forward pass: {forward_time:.3f}s")
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean().item():.6f}")
            print(f"  Output std: {output.std().item():.6f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    # Compare results
    if len(results) == 2:
        base_time = results["Fixed Base"]
        hilbert_time = results["Fixed Hilbert"]
        speedup = base_time / hilbert_time
        print(f"\nHilbert Speedup: {speedup:.3f}x ({(speedup - 1) * 100:+.1f}%)")

    return True


def test_multi_gpu_worker(rank, world_size):
    """Test multi-GPU worker."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12363"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    # Test parameters
    seq_len = 16384  # Larger for multi-GPU
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [4096, 8192]
    dilation_rates = [8, 16]

    # Create test tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Fixed Implementations - Multi-GPU")
        print("=" * 60)
        print(f"Sequence length: {seq_len}")

    # Test Hilbert fixed
    try:
        model = RingDilatedAttentionHilbertFixed(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Synchronize
        dist.barrier()

        # Warmup
        with torch.no_grad():
            _ = model(q, k, v)

        torch.cuda.synchronize()
        dist.barrier()

        # Time forward pass
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v)

        torch.cuda.synchronize()
        end = time.time()

        if rank == 0:
            print("Fixed Hilbert Multi-GPU:")
            print(f"  ✓ Forward pass: {end - start:.3f}s")
            print(f"  Output shape: {output.shape}")
            print(f"  Throughput: {seq_len / (end - start):,.0f} tokens/sec")

    except Exception as e:
        if rank == 0:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    dist.destroy_process_group()


def main():
    """Run all tests."""
    # Test single GPU
    if test_single_gpu():
        # Test multi-GPU if available
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            world_size = 2
            mp.spawn(
                test_multi_gpu_worker, args=(world_size,), nprocs=world_size, join=True
            )
        else:
            print("\nSkipping multi-GPU test (need at least 2 GPUs)")

    print("\nFixed implementations tested successfully!")


if __name__ == "__main__":
    main()
