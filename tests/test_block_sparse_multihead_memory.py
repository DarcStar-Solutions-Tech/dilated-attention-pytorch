#!/usr/bin/env python3
"""Test memory pool usage in block sparse multihead attention."""

import gc
import time

import torch

from dilated_attention_pytorch import BlockSparseRingMultiheadDilatedAttention
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    SparsePatternConfig,
)
from dilated_attention_pytorch.core.memory_pool import (
    get_global_memory_pool,
    reset_global_memory_pool,
)


def test_block_sparse_multihead_memory_pool():
    """Test if block sparse multihead uses memory pool."""
    # Reset global pool
    reset_global_memory_pool()

    # Create block sparse multihead attention
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,  # 90% sparse
        block_size=128,
    )

    attention = BlockSparseRingMultiheadDilatedAttention(
        embed_dim=768,
        num_heads=12,
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        batch_first=True,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)

    # Get global pool stats before
    pool = get_global_memory_pool()
    stats_before = pool.get_stats()
    print("Global pool stats before:")
    print(f"  Total buffers: {stats_before['total_buffers']}")
    print(f"  Total allocated bytes: {stats_before['total_allocated_bytes']}")

    # Test parameters
    batch_size = 1
    seq_len = 1024

    # Create inputs
    x = torch.randn(batch_size, seq_len, 768, device=device)

    # Run multiple forward passes
    outputs = []
    for i in range(5):
        print(f"\nForward pass {i + 1}")
        start_time = time.time()

        # Forward pass
        output = attention(x, x, x)
        # Handle tuple return if need_weights=True
        if isinstance(output, tuple):
            output = output[0]
        outputs.append(output)

        # Get pool stats after this pass
        stats_after = pool.get_stats()
        print(f"  Time: {time.time() - start_time:.3f}s")
        print(f"  Total buffers: {stats_after['total_buffers']}")
        print(f"  Total allocated bytes: {stats_after['total_allocated_bytes']}")
        print(f"  Hot cache size: {stats_after['hot_cache_size']}")

        # Force sync to ensure allocation happens
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Final stats
    stats_final = pool.get_stats()
    print("\nFinal global pool stats:")
    print(f"  Total buffers: {stats_final['total_buffers']}")
    print(f"  Total allocated bytes: {stats_final['total_allocated_bytes']}")
    print(f"  Hot cache size: {stats_final['hot_cache_size']}")
    print(f"  Pool sizes: {stats_final['pool_sizes']}")

    # Check if any buffers were allocated
    buffers_allocated = stats_final["total_buffers"] - stats_before["total_buffers"]
    bytes_allocated = (
        stats_final["total_allocated_bytes"] - stats_before["total_allocated_bytes"]
    )

    print("\nMemory pool usage by block sparse multihead:")
    print(f"  Buffers allocated: {buffers_allocated}")
    print(f"  Bytes allocated: {bytes_allocated}")

    if buffers_allocated > 0:
        print("✓ Block sparse multihead DOES use the global memory pool")
    else:
        print("✗ Block sparse multihead does NOT use the global memory pool")
        print("  (It likely has its own memory management)")

    # Check if it has its own memory pool
    if hasattr(attention, "memory_pool"):
        print("\n✓ Has its own memory_pool attribute")
    elif hasattr(attention, "sparse_attention") and hasattr(
        attention.sparse_attention, "memory_pool"
    ):
        print("\n✓ sparse_attention has memory_pool attribute")
    else:
        print("\n✗ No memory_pool attribute found")

    # Clean up
    del outputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_block_sparse_multihead_memory_pool()
