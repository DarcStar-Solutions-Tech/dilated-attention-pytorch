#!/usr/bin/env python3
"""
Simple test to understand block sparse indices structure.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    SparsePatternConfig,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple block sparse attention
    model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[2048],
        dilation_rates=[1],
        sparse_config=SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=0.05,
            block_size=64,
        ),
    ).to(device)

    # Create inputs
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Get block indices to understand structure
    num_blocks = seq_len // model.block_size
    print(f"Sequence length: {seq_len}")
    print(f"Block size: {model.block_size}")
    print(f"Number of blocks: {num_blocks}")

    # Get the sparse block indices
    block_indices = model._get_sparse_block_indices(num_blocks, num_heads, device)
    row_indices, col_indices = block_indices

    print(f"\nBlock indices shape: {row_indices.shape}, {col_indices.shape}")
    print(f"Block indices dtype: {row_indices.dtype}, {col_indices.dtype}")
    print(
        f"Number of blocks per head: {row_indices.shape[1] if len(row_indices.shape) > 1 else 'scalar'}"
    )

    # Show a few examples
    if len(row_indices.shape) > 1:
        print("\nFirst head block indices (first 10):")
        print(f"Row: {row_indices[0][:10].tolist()}")
        print(f"Col: {col_indices[0][:10].tolist()}")
    else:
        print("\nBlock indices (first 10):")
        print(f"Row: {row_indices[:10].tolist()}")
        print(f"Col: {col_indices[:10].tolist()}")

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        output = model(q, k, v)
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    main()
