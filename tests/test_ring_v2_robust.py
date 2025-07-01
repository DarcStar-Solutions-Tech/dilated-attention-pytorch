#!/usr/bin/env python3
"""
Test the robust Ring Attention implementation.
"""

import os
import torch
import time

# Check if we're in distributed mode
IS_DISTRIBUTED = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

if IS_DISTRIBUTED:
    import torch.distributed as dist


def test_single_gpu():
    """Test single GPU performance."""
    from dilated_attention_pytorch.ring_dilated_attention_v2_robust import (
        RingDilatedAttentionV2Robust,
    )

    print("=" * 60)
    print("Single GPU Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    seq_len = 4096
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")

    # Create model
    model = RingDilatedAttentionV2Robust(
        segment_lengths=[2048, 2048, 2048, 2048],
        dilation_rates=[1, 2, 4, 8],
        ring_size=1,
        device=device,
    )

    # Determine dtype
    dtype = model.dtype
    print(f"Using dtype: {dtype}")

    # Create input
    x = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, x, x, is_causal=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Measure time
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    num_iterations = 10
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(x, x, x, is_causal=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    avg_time = (time.time() - start) / num_iterations

    # Measure memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = model(x, x, x, is_causal=True)

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = 0

    print("\nResults:")
    print(f"  Average time: {avg_time * 1000:.1f} ms")
    print(f"  Peak memory: {peak_memory_mb:.1f} MB")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")


def test_multi_gpu():
    """Test multi-GPU with ring communication."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from dilated_attention_pytorch.ring_dilated_attention_v2_robust import (
        RingDilatedAttentionV2Robust,
    )
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )

    if rank == 0:
        print("=" * 60)
        print(f"Multi-GPU Test ({world_size} GPUs)")
        print("=" * 60)

    # Test parameters
    seq_lengths = [4096, 8192, 16384]
    batch_size = 1
    num_heads = 8
    head_dim = 64

    for seq_len in seq_lengths:
        if rank == 0:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)

        # Create models
        model_robust = RingDilatedAttentionV2Robust(
            segment_lengths=[2048, 2048, 2048, 2048],
            dilation_rates=[1, 2, 4, 8],
            ring_size=world_size,
            device=device,
            min_seq_length_for_ring=1,  # Force ring mode
        )

        model_collective = RingDilatedAttentionV2Collective(
            segment_lengths=[2048, 2048, 2048, 2048],
            dilation_rates=[1, 2, 4, 8],
            ring_size=world_size,
            device=device,
        )

        # Get dtype from model
        dtype = model_robust.dtype
        if rank == 0:
            print(f"Using dtype: {dtype}")

        # Create input
        x = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Test collective (all-gather) first
        if rank == 0:
            print("\nCollective implementation (all-gather):")

        try:
            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = model_collective(x, x, x, is_causal=False)
                    torch.cuda.synchronize()

            # Time
            torch.cuda.synchronize()
            dist.barrier()
            start = time.time()

            iterations = 5
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model_collective(x, x, x, is_causal=False)
                    torch.cuda.synchronize()

            dist.barrier()
            time_collective = (time.time() - start) / iterations

            # Memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model_collective(x, x, x, is_causal=False)

            memory_collective = torch.cuda.max_memory_allocated() / 1024 / 1024

            if rank == 0:
                print(f"  Time: {time_collective * 1000:.1f} ms")
                print(f"  Memory: {memory_collective:.1f} MB")

        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")
            memory_collective = float("inf")
            time_collective = float("inf")

        # Test robust (ring passing)
        if rank == 0:
            print("\nRobust implementation (ring passing):")

        try:
            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = model_robust(x, x, x, is_causal=False)
                    torch.cuda.synchronize()

            # Time
            torch.cuda.synchronize()
            dist.barrier()
            start = time.time()

            iterations = 5
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model_robust(x, x, x, is_causal=False)
                    torch.cuda.synchronize()

            dist.barrier()
            time_robust = (time.time() - start) / iterations

            # Memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model_robust(x, x, x, is_causal=False)

            memory_robust = torch.cuda.max_memory_allocated() / 1024 / 1024

            if rank == 0:
                print(f"  Time: {time_robust * 1000:.1f} ms")
                print(f"  Memory: {memory_robust:.1f} MB")

                # Compare
                if memory_collective != float("inf"):
                    memory_reduction = (1 - memory_robust / memory_collective) * 100
                    print(f"\n  Memory reduction: {memory_reduction:.1f}%")
                    print(f"  Time ratio: {time_robust / time_collective:.2f}x")

        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()

        # Sync before next test
        dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Summary:")
        print("- Robust implementation uses sendrecv for guaranteed synchronization")
        print("- Collective uses all-gather (simpler but more memory)")
        print("- Ring passing trades communication time for memory efficiency")

    dist.destroy_process_group()


if __name__ == "__main__":
    if IS_DISTRIBUTED:
        test_multi_gpu()
    else:
        test_single_gpu()
