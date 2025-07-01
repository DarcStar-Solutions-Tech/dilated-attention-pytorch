#!/usr/bin/env python3
"""
Clean test of fixed Ring Attention - only tests distributed mode when available.
"""

import os
import torch
import time

# Check if we're in distributed mode
IS_DISTRIBUTED = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

if IS_DISTRIBUTED:
    import torch.distributed as dist
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )
    from dilated_attention_pytorch.ring_dilated_attention_v2_fixed import (
        RingDilatedAttentionV2Fixed,
    )


def test_single_gpu():
    """Test single GPU performance."""
    from dilated_attention_pytorch.ring_dilated_attention_v2_fixed import (
        RingDilatedAttentionV2Fixed,
    )

    print("=" * 60)
    print("Single GPU Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Test parameters
    seq_len = 16384
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")
    print(f"Dtype: {dtype}")

    # Create model
    model = RingDilatedAttentionV2Fixed(
        segment_lengths=[4096, 4096],
        dilation_rates=[1, 2],
        ring_size=1,
        device=device,
        dtype=dtype,
    )

    # Create input
    x = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, x, x, is_causal=True)

    # Measure time
    torch.cuda.synchronize()
    start = time.time()

    num_iterations = 10
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(x, x, x, is_causal=True)
            torch.cuda.synchronize()

    avg_time = (time.time() - start) / num_iterations

    # Measure memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output = model(x, x, x, is_causal=True)

    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("\nResults:")
    print(f"  Average time: {avg_time * 1000:.1f} ms")
    print(f"  Peak memory: {peak_memory_mb:.1f} MB")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print("\nSingle GPU mode uses _single_device_forward (no ring communication)")


def test_multi_gpu():
    """Test multi-GPU with ring communication."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=" * 60)
        print(f"Multi-GPU Test ({world_size} GPUs)")
        print("=" * 60)

    # Determine dtype
    compute_capability = torch.cuda.get_device_capability(device)
    dtype = torch.float32 if compute_capability[0] < 7 else torch.float16

    # Test parameters
    seq_lengths = [4096, 8192]  # Smaller sequences for memory constraints
    batch_size = 1
    num_heads = 8
    head_dim = 64

    for seq_len in seq_lengths:
        if rank == 0:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)

        # Create input
        x = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Test original (all-gather)
        if rank == 0:
            print("Original implementation (all-gather):")

        try:
            model_orig = RingDilatedAttentionV2Collective(
                segment_lengths=[4096, 4096],
                dilation_rates=[1, 2],
                ring_size=world_size,
                device=device,
                dtype=dtype,
            )

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = model_orig(x, x, x, is_causal=False)

            # Time
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                for _ in range(5):
                    _ = model_orig(x, x, x, is_causal=False)
                    torch.cuda.synchronize()

            time_orig = (time.time() - start) / 5

            # Memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model_orig(x, x, x, is_causal=False)

            memory_orig = torch.cuda.max_memory_allocated() / 1024 / 1024

            if rank == 0:
                print(f"  Time: {time_orig * 1000:.1f} ms")
                print(f"  Memory: {memory_orig:.1f} MB")

        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")
            memory_orig = float("inf")

        # Test fixed (ring passing)
        if rank == 0:
            print("\nFixed implementation (ring passing):")

        try:
            model_fixed = RingDilatedAttentionV2Fixed(
                segment_lengths=[4096, 4096],
                dilation_rates=[1, 2],
                ring_size=world_size,
                device=device,
                dtype=dtype,
                min_seq_length_for_ring=8192,  # Force ring mode
            )

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = model_fixed(x, x, x, is_causal=False)

            # Time
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                for _ in range(5):
                    _ = model_fixed(x, x, x, is_causal=False)
                    torch.cuda.synchronize()

            time_fixed = (time.time() - start) / 5

            # Memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model_fixed(x, x, x, is_causal=False)

            memory_fixed = torch.cuda.max_memory_allocated() / 1024 / 1024

            if rank == 0:
                print(f"  Time: {time_fixed * 1000:.1f} ms")
                print(f"  Memory: {memory_fixed:.1f} MB")

                # Compare
                if memory_orig != float("inf"):
                    memory_reduction = (1 - memory_fixed / memory_orig) * 100
                    print(f"\n  Memory reduction: {memory_reduction:.1f}%")
                    print(f"  Time ratio: {time_fixed / time_orig:.2f}x")

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
        print("- Fixed implementation uses ring passing (O(n/ring_size) K/V memory)")
        print("- Original uses all-gather (O(n) K/V memory)")
        print("- Ring passing trades communication time for memory efficiency")

    dist.destroy_process_group()


if __name__ == "__main__":
    if IS_DISTRIBUTED:
        test_multi_gpu()
    else:
        test_single_gpu()
