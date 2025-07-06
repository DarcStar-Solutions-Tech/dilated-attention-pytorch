#!/usr/bin/env python3
"""Test to verify ring attention applies dilation AFTER splitting, not before."""

import torch
import torch.distributed as dist
import os


def init_distributed():
    """Initialize distributed if not already done."""
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank
    return 0, 0


def test_dilation_order():
    """Test that dilation is applied per chunk, not globally."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print(f"Rank {rank}: Testing dilation order...")

    # Import the implementation
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Test parameters
    seq_len = 16 * world_size  # Small for testing
    batch_size = 1
    num_heads = 2
    head_dim = 4

    # Create model with dilation
    model = RingDilatedAttentionHybridHilbert(
        segment_lengths=[8],
        dilation_rates=[2],  # Dilation rate 2
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        use_hilbert=False,  # Disable Hilbert to focus on dilation
        enable_memory_pool=False,
        use_xformers=False,
    )

    # Create simple inputs to track pattern
    # Q: all ones
    q = torch.ones(batch_size, seq_len, num_heads, head_dim, device=device)
    # K: position indices for tracking
    k = (
        torch.arange(seq_len, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(3)
    )
    k = k.expand(batch_size, seq_len, num_heads, head_dim)
    # V: same as K for easy tracking
    v = k.clone()

    print(f"Rank {rank}: Input sequence length: {seq_len}")
    print(f"Rank {rank}: K values (positions): {k[0, :, 0, 0].tolist()}")

    # Forward pass
    output = model(q, k, v, is_causal=False)

    print(f"Rank {rank}: Output shape: {output.shape}")
    print(f"Rank {rank}: Output sample values: {output[0, :8, 0, 0].tolist()}")

    # Analysis
    # With dilation rate 2, each position should only attend to every 2nd position
    # If dilation is applied CORRECTLY (after splitting):
    # - Each GPU gets seq_len/world_size positions
    # - Then applies dilation within its chunk
    # If dilation is applied INCORRECTLY (before splitting):
    # - Dilation would be applied globally first
    # - Then split across GPUs

    if rank == 0:
        print("\nAnalyzing dilation behavior:")
        print("Expected behavior: Dilation applied AFTER splitting")
        print("- Each GPU processes its chunk independently")
        print("- Dilation pattern is local to each chunk")
        print("\nActual output shows which K positions were attended to")


if __name__ == "__main__":
    test_dilation_order()
