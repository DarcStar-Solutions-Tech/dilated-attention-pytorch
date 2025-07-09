#!/usr/bin/env python3
"""
Test extreme sparsity levels to push sequence lengths even further.
"""

import torch
import gc
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dilated_attention_pytorch.block_sparse_ring_dilated_attention_fixed import (
    BlockSparseRingDilatedAttentionFixed,
)
from src.dilated_attention_pytorch.core.standardized_api import StandardizedRingConfig


def test_sparsity_levels():
    """Test different sparsity levels to see how far we can push."""
    print("=" * 80)
    print("EXTREME SPARSITY TESTING")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Test different sparsity ratios
    sparsity_ratios = [0.1, 0.05, 0.02, 0.01, 0.005]  # 90%, 95%, 98%, 99%, 99.5% sparse

    # Test sequence lengths
    test_lengths = [
        262144,  # 256K (baseline from previous test)
        524288,  # 512K
        1048576,  # 1M
        2097152,  # 2M
    ]

    results = {}

    for sparsity in sparsity_ratios:
        print(f"\n{'=' * 60}")
        print(
            f"Testing sparsity ratio: {sparsity} ({(1 - sparsity) * 100:.1f}% sparse)"
        )
        print(f"{'=' * 60}")

        max_len = 0

        for seq_len in test_lengths:
            print(f"\nTesting {seq_len:,} tokens...")

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            try:
                # Segment configuration for long sequences
                if seq_len <= 262144:
                    segment_lengths = [4096, 8192, 16384]
                    dilation_rates = [1, 2, 4]
                else:
                    segment_lengths = [8192, 16384, 32768]
                    dilation_rates = [1, 2, 4]

                config = StandardizedRingConfig(
                    dim=head_dim,
                    heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    sparsity_ratio=sparsity,
                    block_size=min(256, seq_len // 64),  # Larger blocks for efficiency
                )

                model = BlockSparseRingDilatedAttentionFixed(config=config).to(device)

                # Create minimal test
                q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
                k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
                v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

                # Get memory before forward
                _ = torch.cuda.memory_allocated() / 1024**3

                # Forward pass
                output = model(q, k, v, is_causal=False)

                # Get peak memory
                alloc_after = torch.cuda.memory_allocated() / 1024**3
                peak_memory = alloc_after

                print(f"  ✓ Success! Peak memory: {peak_memory:.2f} GB")
                print(
                    f"  Non-zero attention values: {sparsity * seq_len * seq_len / 1e6:.1f}M"
                )

                max_len = seq_len

                # Clean up
                del model, q, k, v, output

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("  ✗ OOM")
                    break
                else:
                    print(f"  ✗ Error: {e}")
                    break

        results[sparsity] = max_len

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Maximum Sequence Lengths by Sparsity")
    print("=" * 80)

    for sparsity, max_len in results.items():
        sparse_percent = (1 - sparsity) * 100
        print(f"{sparse_percent:4.1f}% sparse: {max_len:,} tokens")

    # Test specific configuration for 1M tokens
    if 0.01 in results and results[0.01] >= 1_000_000:
        print("\n" + "=" * 80)
        print("1 MILLION TOKEN TEST")
        print("=" * 80)

        seq_len = 1_048_576
        sparsity = 0.01  # 99% sparse

        config = StandardizedRingConfig(
            dim=head_dim,
            heads=num_heads,
            segment_lengths=[8192, 16384, 32768],
            dilation_rates=[1, 2, 4],
            dropout=0.0,
            sparsity_ratio=sparsity,
            block_size=256,
        )

        _ = BlockSparseRingDilatedAttentionFixed(config=config).to(device)

        # Memory calculation
        input_memory = (
            3 * seq_len * num_heads * head_dim * 4 / 1024**3
        )  # 3 tensors, 4 bytes/float
        sparse_memory = sparsity * seq_len * seq_len * 4 / 1024**3  # Attention matrix

        print("Theoretical memory requirements:")
        print(f"  Input tensors (Q,K,V): {input_memory:.2f} GB")
        print(f"  Sparse attention matrix: {sparse_memory:.2f} GB")
        print(f"  Total (minimum): {input_memory + sparse_memory:.2f} GB")
        print(
            f"\nWith 99% sparsity, attention matrix has {sparsity * seq_len * seq_len / 1e9:.1f}B non-zero values"
        )


if __name__ == "__main__":
    test_sparsity_levels()
