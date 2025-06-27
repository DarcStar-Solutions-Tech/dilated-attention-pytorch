#!/usr/bin/env python3
"""
Simple test to demonstrate memory usage differences.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

import torch

# Import implementations
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    SparsePatternConfig as OriginalSparseConfig,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_v2 import (
    BlockSparseRingDilatedAttentionV2,
    SparsePatternConfig,
)


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_original_implementation():
    """Test original implementation memory usage."""
    print("\n=== Testing Original BlockSparseRingDilatedAttention ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 8192
    batch_size = 2
    num_heads = 8
    head_dim = 64

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_mem = get_memory_mb()
    print(f"Starting memory: {start_mem:.1f} MB")

    # Create module
    sparse_config = OriginalSparseConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,  # 90% sparse
        block_size=128,
    )

    module = BlockSparseRingDilatedAttention(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        ring_size=1,
    ).to(device)

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = q.clone()
    v = q.clone()

    after_init_mem = get_memory_mb()
    print(f"After initialization: {after_init_mem:.1f} MB (+{after_init_mem - start_mem:.1f} MB)")

    # Forward without weights
    output = module(q, k, v)
    after_forward_mem = get_memory_mb()
    print(
        f"After forward (no weights): {after_forward_mem:.1f} MB (+{after_forward_mem - after_init_mem:.1f} MB)"
    )

    # Clear and try with weights
    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    before_weights_mem = get_memory_mb()

    try:
        output, weights = module(q, k, v, return_attention_weights=True)
        after_weights_mem = get_memory_mb()
        print(
            f"After forward (with weights): {after_weights_mem:.1f} MB (+{after_weights_mem - before_weights_mem:.1f} MB)"
        )

        if weights is not None and hasattr(weights, "shape"):
            print(f"Attention weights shape: {weights.shape}")
            weight_memory = weights.numel() * weights.element_size() / 1024 / 1024
            print(f"Attention weights memory: {weight_memory:.1f} MB")
    except RuntimeError as e:
        print(f"Failed with attention weights: {e}")


def test_v2_implementation():
    """Test V2 implementation memory usage."""
    print("\n=== Testing BlockSparseRingDilatedAttentionV2 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 8192
    batch_size = 2
    num_heads = 8
    head_dim = 64

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_mem = get_memory_mb()
    print(f"Starting memory: {start_mem:.1f} MB")

    # Create module
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,  # 90% sparse
        block_size=128,
    )

    module = BlockSparseRingDilatedAttentionV2(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        ring_size=1,
    ).to(device)

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = q.clone()
    v = q.clone()

    after_init_mem = get_memory_mb()
    print(f"After initialization: {after_init_mem:.1f} MB (+{after_init_mem - start_mem:.1f} MB)")

    # Forward without weights
    output = module(q, k, v)
    after_forward_mem = get_memory_mb()
    print(
        f"After forward (no weights): {after_forward_mem:.1f} MB (+{after_forward_mem - after_init_mem:.1f} MB)"
    )

    # Clear and try with weights
    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    before_weights_mem = get_memory_mb()

    output, weights = module(q, k, v, return_attention_weights=True)
    after_weights_mem = get_memory_mb()
    print(
        f"After forward (with weights): {after_weights_mem:.1f} MB (+{after_weights_mem - before_weights_mem:.1f} MB)"
    )

    if weights is not None and isinstance(weights, dict):
        print(f"Attention weights format: Sparse with keys {list(weights.keys())}")
        if "block_indices" in weights:
            num_blocks = len(weights["block_indices"])
            total_possible_blocks = (seq_len // sparse_config.block_size) ** 2
            actual_sparsity = 1 - (num_blocks / total_possible_blocks)
            print(f"Number of active blocks: {num_blocks} out of {total_possible_blocks}")
            print(f"Actual sparsity: {actual_sparsity:.1%}")

            # Estimate memory for sparse format
            # Each block is block_size x block_size
            block_memory = (
                num_blocks * sparse_config.block_size * sparse_config.block_size * 2 / 1024 / 1024
            )
            print(f"Estimated sparse weights memory: {block_memory:.1f} MB")


def main():
    """Run comparison."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, memory measurements will be inaccurate")

    print("Block Sparse Memory Usage Comparison")
    print("=" * 60)

    try:
        test_original_implementation()
    except Exception as e:
        print(f"Original implementation error: {e}")

    try:
        test_v2_implementation()
    except Exception as e:
        print(f"V2 implementation error: {e}")

    print("\n" + "=" * 60)
    print("Key Differences:")
    print("1. Original stores full dense attention matrices [batch, heads, seq, seq]")
    print("2. V2 stores only active blocks in sparse format")
    print("3. For 90% sparsity, V2 uses ~10x less memory for attention weights")


if __name__ == "__main__":
    main()
