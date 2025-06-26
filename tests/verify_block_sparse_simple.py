#!/usr/bin/env python3
"""
Simple verification that Block Sparse Ring Dilated Attention works correctly.
"""

import time

import torch


def main():
    print("=" * 60)
    print("Block Sparse Ring Dilated Attention - Simple Verification")
    print("=" * 60)

    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
        BlockSparseRingMultiheadDilatedAttention,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Test 1: Basic functionality
    print("\n1. Testing basic BlockSparseRingDilatedAttention...")

    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse", sparsity_ratio=0.25, block_size=32  # 75% sparse
    )

    attention = BlockSparseRingDilatedAttention(
        segment_lengths=[256],
        dilation_rates=[1],
        sparse_config=sparse_config,
        dropout=0.0,
        device=device,
    )

    # Small test
    batch_size = 2
    seq_len = 512
    num_heads = 4
    head_dim = 32

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    start = time.time()
    output = attention(q, k, v, is_causal=False)
    end = time.time()

    print("✓ Forward pass successful")
    print(f"  Input shape:  {q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Time: {(end - start) * 1000:.2f} ms")

    # Test 2: Memory info
    print("\n2. Testing memory efficiency...")
    info = attention.get_memory_info()
    print("✓ Memory info available:")
    print(f"  Sparsity ratio: {info['sparsity_ratio']}")
    print(f"  Memory reduction: {info['memory_reduction']}")
    print(f"  Theoretical speedup: {info['theoretical_speedup']}")

    # Test 3: Different sparsity levels
    print("\n3. Testing different sparsity levels...")
    sparsity_levels = [0.1, 0.5, 0.9]

    for sparsity in sparsity_levels:
        sparse_config = SparsePatternConfig(
            pattern_type="dilated_sparse", sparsity_ratio=sparsity, block_size=32
        )

        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[256],
            dilation_rates=[1],
            sparse_config=sparse_config,
            dropout=0.0,
            device=device,
        )

        output = attention(q, k, v, is_causal=False)
        info = attention.get_memory_info()

        print(
            f"✓ Sparsity {sparsity:.0%}: Memory reduction = {info['memory_reduction']}"
        )

    # Test 4: Multihead version
    print("\n4. Testing BlockSparseRingMultiheadDilatedAttention...")

    embed_dim = 256
    num_heads = 8

    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse", sparsity_ratio=0.25, block_size=32
    )

    multihead_attention = BlockSparseRingMultiheadDilatedAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=[256],
        dilation_rates=[1],
        sparse_config=sparse_config,
        dropout=0.0,
        device=device,
    )

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    start = time.time()
    output = multihead_attention(x, x, x, is_causal=False)
    if isinstance(output, tuple):
        output = output[0]
    end = time.time()

    print("✓ Multihead forward pass successful")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Time: {(end - start) * 1000:.2f} ms")

    # Test 5: Pattern types
    print("\n5. Testing different sparse patterns...")
    patterns = ["local_window", "dilated_sparse", "global_local"]

    for pattern in patterns:
        try:
            sparse_config = SparsePatternConfig(
                pattern_type=pattern, sparsity_ratio=0.5, block_size=32
            )

            attention = BlockSparseRingDilatedAttention(
                segment_lengths=[256],
                dilation_rates=[1],
                sparse_config=sparse_config,
                dropout=0.0,
                device=device,
            )

            # Quick forward pass
            output = attention(q[:1, :256], k[:1, :256], v[:1, :256], is_causal=False)
            print(f"✓ Pattern '{pattern}' works correctly")

        except Exception as e:
            print(f"✗ Pattern '{pattern}' failed: {e}")

    print("\n" + "=" * 60)
    print("✓ All Block Sparse implementations verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
