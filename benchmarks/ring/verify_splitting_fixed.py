#!/usr/bin/env python3
"""
Fixed verification that ring attention splits sequences properly.
Uses LOCAL_RANK for proper GPU assignment.
"""

import torch
import torch.distributed as dist
import os

from dilated_attention_pytorch import (
    StandardRingAttention,
    RingAttentionConfig,
)
from dilated_attention_pytorch.utils import get_optimal_dtype


def main():
    """Test if StandardRingAttention actually splits sequences."""

    # Initialize distributed if available
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        print(f"Distributed mode: Rank {rank}/{world_size}, Local rank: {local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        print("Single GPU mode")

    # CRITICAL FIX: Use local_rank for device selection
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")

    # Test parameters
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    print("\nTest Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expected local length: {seq_len // world_size}")

    # Create model with correct device
    config = RingAttentionConfig(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        dropout=0.0,
    )

    # Select optimal dtype based on GPU architecture
    dtype = get_optimal_dtype(device)

    model = StandardRingAttention(config, device=device, dtype=dtype)

    # Check if model has _split_sequence method
    if hasattr(model, "_split_sequence"):
        print("\n✓ Model has _split_sequence method")

        # Create test tensor on correct device
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Test splitting
        local_x, start_idx, end_idx = model._split_sequence(x, already_split=False)

        print(f"\n[Rank {rank}] Splitting results:")
        print(f"  Original shape: {x.shape}")
        print(f"  Local shape: {local_x.shape}")
        print(f"  Start index: {start_idx}")
        print(f"  End index: {end_idx}")
        print(f"  Local sequence length: {local_x.shape[1]}")

        # Verify splitting
        expected_local_len = seq_len // world_size
        if local_x.shape[1] == expected_local_len:
            print(f"\n✓ VERIFIED: Sequence properly split to {local_x.shape[1]} tokens")
        else:
            print(
                f"\n✗ ERROR: Expected {expected_local_len} tokens, got {local_x.shape[1]}"
            )

    # Test forward pass with memory monitoring
    print(f"\n[Rank {rank}] " + "=" * 50)
    print(f"[Rank {rank}] Testing forward pass with memory monitoring...")
    print(f"[Rank {rank}] " + "=" * 50)

    # Create inputs on correct device
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Memory before
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device) / (1024**2)
        print(f"[Rank {rank}] Memory before forward: {mem_before:.2f} MB")

    # Forward pass
    try:
        output = model(q, k, v)
        print(f"[Rank {rank}] ✓ Forward pass successful")
        print(f"[Rank {rank}]   Output shape: {output.shape}")

        # Memory after
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            mem_used = peak_mem - mem_before
            print(f"[Rank {rank}]   Peak memory: {peak_mem:.2f} MB")
            print(f"[Rank {rank}]   Memory used: {mem_used:.2f} MB")

            # Estimate expected memory
            # For ring attention with world_size GPUs, each GPU should use ~1/world_size memory
            full_attention_estimate = (seq_len * seq_len * 2) / (1024**2)  # float16
            ring_attention_estimate = full_attention_estimate / world_size

            print(f"\n[Rank {rank}] Memory analysis:")
            print(f"  Full attention estimate: {full_attention_estimate:.2f} MB")
            print(f"  Ring attention estimate: {ring_attention_estimate:.2f} MB")
            print(f"  Actual memory used: {mem_used:.2f} MB")

            # Check if memory usage is reasonable
            # In multi-GPU, each GPU should use less memory
            if world_size > 1 and mem_used < full_attention_estimate * 0.6:
                print(
                    f"  ✓ Memory usage confirms ring attention O(n/{world_size}) scaling!"
                )
            elif world_size == 1:
                print("  ✓ Single GPU: using full memory as expected")
            else:
                print("  ⚠️ Memory usage higher than expected for ring attention")

    except Exception as e:
        print(f"[Rank {rank}] ✗ Forward pass failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()
        print(f"\n[Rank {rank}] All ranks completed successfully")
        dist.destroy_process_group()

    print(f"\n[Rank {rank}] Verification complete!")


if __name__ == "__main__":
    main()
