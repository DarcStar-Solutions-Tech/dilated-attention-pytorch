#!/usr/bin/env python3
"""
Test memory scaling of Ring V3.
Run with: torchrun --nproc_per_node=<num_gpus> test_ring_v3_memory.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_memory_scaling():
    """Test if we achieve O(n/p) memory scaling."""

    # Initialize distributed
    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("Testing Ring V3 Memory Scaling")
        print(f"World size: {world_size}")
        print("=" * 50)

    # Test configuration
    seq_len = 8192  # 8K tokens
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # Create model
        model = RingDilatedAttentionV3(
            segment_lengths=[2048],
            dilation_rates=[1],
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ring_size=world_size,
        )

        # Get memory before creating inputs
        if device.type == "cuda":
            mem_model = torch.cuda.memory_allocated() / (1024**2)

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )

        # Get memory after inputs
        if device.type == "cuda":
            mem_inputs = torch.cuda.memory_allocated() / (1024**2)
            input_size = mem_inputs - mem_model

        # Forward pass
        _ = model(q, k, v, is_causal=False)

        # Synchronize and get peak memory
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

            # Gather memory stats from all ranks
            if world_size > 1:
                all_peaks = [None] * world_size
                all_inputs = [None] * world_size
                dist.all_gather_object(all_peaks, peak_memory)
                dist.all_gather_object(all_inputs, input_size)

                if rank == 0:
                    avg_peak = sum(all_peaks) / world_size
                    avg_input = sum(all_inputs) / world_size
                    print(f"\nResults for {seq_len:,} tokens:")
                    print(f"  Input size per GPU: {avg_input:.1f} MB")
                    print(f"  Peak memory per GPU: {avg_peak:.1f} MB")
                    print(f"  Individual GPUs: {[f'{m:.1f}' for m in all_peaks]} MB")

                    # Check if we achieve O(n/p) scaling
                    # With true ring attention, K,V should be split across GPUs
                    expected_kv_per_gpu = input_size * 2 / world_size  # K and V split
                    actual_overhead = avg_peak - avg_input
                    print("\nMemory scaling analysis:")
                    print(f"  Expected K,V per GPU: ~{expected_kv_per_gpu:.1f} MB")
                    print(f"  Actual overhead: {actual_overhead:.1f} MB")

                    # If overhead is close to full K,V size, we're not achieving O(n/p)
                    if actual_overhead > input_size * 1.5:
                        print("  ❌ NOT achieving O(n/p) scaling")
                    else:
                        print("  ✅ Achieving O(n/p) scaling!")
            else:
                print("\nSingle GPU results:")
                print(f"  Input size: {input_size:.1f} MB")
                print(f"  Peak memory: {peak_memory:.1f} MB")

        # Test completed
        success = True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[Rank {rank}] OOM with {seq_len:,} tokens")
        else:
            raise
        success = False

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

    return success


if __name__ == "__main__":
    test_memory_scaling()
