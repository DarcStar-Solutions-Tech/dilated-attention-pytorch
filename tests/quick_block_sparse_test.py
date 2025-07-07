#!/usr/bin/env python3
"""Quick test to verify block-sparse implementations work correctly."""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import (
    create_block_sparse_attention,
    get_block_sparse_preset,
)


def main():
    print("Quick Block-Sparse Test")
    print("=" * 50)

    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test each implementation
    implementations = [
        ("Base", lambda: create_block_sparse_attention("base")),
        ("Hierarchical", lambda: create_block_sparse_attention("hierarchical")),
        ("Adaptive", lambda: create_block_sparse_attention("adaptive")),
        ("Preset-Local", lambda: get_block_sparse_preset("local")),
        ("Ultra-Sparse", lambda: get_block_sparse_preset("ultra_sparse")),
    ]

    for name, create_fn in implementations:
        try:
            print(f"\nTesting {name}...")
            model = create_fn().to(device)

            # Forward pass
            output = model(q, k, v)
            print(f"  Output shape: {output.shape}")
            print(
                f"  Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                if device == "cuda"
                else "  CPU mode"
            )

            # Check for NaN/Inf
            if torch.isnan(output).any():
                print("  ⚠️  Warning: Output contains NaN")
            elif torch.isinf(output).any():
                print("  ⚠️  Warning: Output contains Inf")
            else:
                print("  ✓ Output is valid")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

    # Test multihead
    print("\nTesting Multihead...")
    try:
        embed_dim = num_heads * head_dim
        model = create_block_sparse_attention(
            "multihead", embed_dim=embed_dim, num_heads=num_heads
        ).to(device)

        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        output = model(x, x, x)
        print(f"  Output shape: {output.shape}")
        print("  ✓ Multihead works")

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    main()
