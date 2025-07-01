#!/usr/bin/env python3
"""
Test the fixed Ring Attention implementation with true ring communication.
"""

import os
import torch
import torch.distributed as dist
import time
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)
from dilated_attention_pytorch.ring_dilated_attention_v2_fixed import (
    RingDilatedAttentionV2Fixed,
)


def setup_distributed():
    """Setup distributed environment."""
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_memory_and_time(model, x, num_iterations=5):
    """Measure peak memory and average time."""
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(x, x, x, is_causal=True)

    # Clear memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Time measurement
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(x, x, x, is_causal=True)
            torch.cuda.synchronize()

    avg_time = (time.time() - start_time) / num_iterations
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return output, avg_time, peak_memory_mb


def main():
    rank, world_size, device = setup_distributed()

    # Test parameters
    seq_lengths = [8192, 16384, 32768]
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Determine optimal dtype
    compute_capability = torch.cuda.get_device_capability(device)
    dtype = torch.float32 if compute_capability[0] < 7 else torch.float16

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Testing Fixed Ring Attention vs Original Implementation")
        print(f"World size: {world_size}, Device: {device}, Dtype: {dtype}")
        print(f"{'=' * 70}\n")

    for seq_length in seq_lengths:
        if rank == 0:
            print(f"\nSequence length: {seq_length}")
            print("-" * 50)

        try:
            # Create input
            x = torch.randn(
                batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
            )

            # Test original implementation
            if rank == 0:
                print("Original (all-gather):")

            model_orig = RingDilatedAttentionV2Collective(
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                ring_size=world_size,
                device=device,
                dtype=dtype,
            )

            try:
                output_orig, time_orig, memory_orig = measure_memory_and_time(
                    model_orig, x
                )

                if rank == 0:
                    print(f"  Time: {time_orig * 1000:.1f} ms")
                    print(f"  Memory: {memory_orig:.1f} MB")
                    print(f"  Output mean: {output_orig.mean().item():.6f}")
            except Exception as e:
                if rank == 0:
                    print(f"  Error: {e}")
                output_orig = None
                time_orig = float("inf")
                memory_orig = float("inf")

            # Test fixed implementation
            if rank == 0:
                print("\nFixed (ring passing):")

            model_fixed = RingDilatedAttentionV2Fixed(
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                ring_size=world_size,
                device=device,
                dtype=dtype,
                min_seq_length_for_ring=8192,  # Use ring for all test sequences
            )

            try:
                output_fixed, time_fixed, memory_fixed = measure_memory_and_time(
                    model_fixed, x
                )

                if rank == 0:
                    print(f"  Time: {time_fixed * 1000:.1f} ms")
                    print(f"  Memory: {memory_fixed:.1f} MB")
                    print(f"  Output mean: {output_fixed.mean().item():.6f}")

                    # Compare results
                    if output_orig is not None:
                        diff = (output_fixed - output_orig).abs().mean().item()
                        print(f"\n  Output difference: {diff:.6f}")

                    # Show improvements
                    if time_orig != float("inf"):
                        speedup = time_orig / time_fixed
                        memory_reduction = (1 - memory_fixed / memory_orig) * 100
                        print(f"  Speedup: {speedup:.2f}x")
                        print(f"  Memory reduction: {memory_reduction:.1f}%")

            except Exception as e:
                if rank == 0:
                    print(f"  Error: {e}")
                    import traceback

                    traceback.print_exc()

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"  OOM for sequence length {seq_length}")

        # Synchronize before next test
        if world_size > 1:
            dist.barrier()

    # Summary
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")

        if world_size == 1:
            print("Single GPU mode - both implementations should be similar")
        else:
            print(f"Multi-GPU mode ({world_size} GPUs):")
            print(
                "- Fixed implementation uses ring passing (O(n/ring_size) K/V memory)"
            )
            print("- Original uses all-gather (O(n) K/V memory)")
            print("\nExpected improvements:")
            print("- Memory usage should decrease with ring size")
            print("- Communication may be slower but more memory efficient")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
