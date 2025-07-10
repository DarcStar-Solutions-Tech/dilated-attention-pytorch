#!/usr/bin/env python3
"""Test memory pool usage in block sparse multihead attention."""

import gc
import time

import torch

from dilated_attention_pytorch import BlockSparseRingMultiheadDilatedAttention
from dilated_attention_pytorch import (
    SparsePatternConfig,
)
from dilated_attention_pytorch.core.unified_memory_pool import (
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
    print(f"  Available keys: {list(stats_before.keys())}")
    print(f"  Cached tensors: {stats_before.get('cached_tensors', 0)}")
    print(f"  Allocations: {stats_before.get('allocations', 0)}")
    print(f"  Reuses: {stats_before.get('reuses', 0)}")

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
        print(f"  Cached tensors: {stats_after.get('cached_tensors', 0)}")
        print(f"  Allocations: {stats_after.get('allocations', 0)}")
        print(f"  Reuses: {stats_after.get('reuses', 0)}")

        # Force sync to ensure allocation happens
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Final stats
    stats_final = pool.get_stats()
    print("\nFinal global pool stats:")
    print(f"  Cached tensors: {stats_final.get('cached_tensors', 0)}")
    print(f"  Total allocations: {stats_final.get('allocations', 0)}")
    print(f"  Total reuses: {stats_final.get('reuses', 0)}")
    print(f"  Total deallocations: {stats_final.get('deallocations', 0)}")

    # Check if any memory pool activity occurred
    allocations_made = stats_final.get("allocations", 0) - stats_before.get(
        "allocations", 0
    )
    reuses_made = stats_final.get("reuses", 0) - stats_before.get("reuses", 0)
    total_activity = allocations_made + reuses_made

    print("\nMemory pool usage by block sparse multihead:")
    print(f"  New allocations: {allocations_made}")
    print(f"  Buffer reuses: {reuses_made}")
    print(f"  Total activity: {total_activity}")

    if total_activity > 0:
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
