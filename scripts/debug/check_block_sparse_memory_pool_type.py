#!/usr/bin/env python3
"""Check what type of memory pool block sparse attention uses."""

from dilated_attention_pytorch import BlockSparseRingMultiheadDilatedAttention
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    SparsePatternConfig,
)


def check_memory_pool_type():
    """Check the type of memory pool used by block sparse attention."""

    # Create block sparse multihead attention
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse", sparsity_ratio=0.1, block_size=128
    )

    attention = BlockSparseRingMultiheadDilatedAttention(
        embed_dim=768,
        num_heads=12,
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        batch_first=True,
    )

    print("Checking memory pool in BlockSparseRingMultiheadDilatedAttention:")
    print(f"  Has sparse_attention: {hasattr(attention, 'sparse_attention')}")

    if hasattr(attention, "sparse_attention"):
        sparse_attn = attention.sparse_attention
        print(f"  sparse_attention type: {type(sparse_attn).__name__}")
        print(
            f"  sparse_attention has memory_pool: {hasattr(sparse_attn, 'memory_pool')}"
        )

        if hasattr(sparse_attn, "memory_pool"):
            memory_pool = sparse_attn.memory_pool
            print(f"  Memory pool type: {type(memory_pool).__name__}")
            print(f"  Memory pool module: {type(memory_pool).__module__}")

            # Check if it's the RingDilatedAttention.MemoryPool
            if hasattr(memory_pool, "max_pool_size"):
                print(f"  max_pool_size: {memory_pool.max_pool_size}")
            if hasattr(memory_pool, "max_cache_size"):
                print(f"  max_cache_size: {memory_pool.max_cache_size}")
            if hasattr(memory_pool, "_pools"):
                print(f"  Number of buffers in pool: {len(memory_pool._pools)}")

            # Check methods
            print("\n  Memory pool methods:")
            methods = [
                m
                for m in dir(memory_pool)
                if not m.startswith("_") and callable(getattr(memory_pool, m))
            ]
            for method in sorted(methods):
                print(f"    - {method}")

    # Check parent class
    print("\nParent classes of BlockSparseRingDilatedAttention:")
    for base in type(sparse_attn).__bases__:
        print(f"  - {base.__name__} from {base.__module__}")

        # Check if parent has its own MemoryPool class
        if hasattr(base, "MemoryPool"):
            print("    Has MemoryPool inner class")


if __name__ == "__main__":
    check_memory_pool_type()
