#!/usr/bin/env python3
"""Quick test of block-sparse implementations."""

import torch
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    create_multihead_block_sparse,
    SparsePatternConfig,
)
from dilated_attention_pytorch.block_sparse_adaptive_fixed import BlockSparseAdaptive


def test_implementation(name, model, seq_len=2048, multihead=False):
    """Test a single implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_heads = 8
    head_dim = 64

    try:
        # Move model to device
        if hasattr(model, "to"):
            model = model.to(device)

        # Create inputs
        if multihead:
            embed_dim = num_heads * head_dim
            q = torch.randn(
                batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
            )
        else:
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        start = time.time()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000

        # Check output
        expected_shape = q.shape
        actual_shape = output.shape

        if actual_shape == expected_shape:
            print(f"✓ {name}: {elapsed:.1f}ms for {seq_len} tokens")
            return True
        else:
            print(f"✗ {name}: Shape mismatch {actual_shape} != {expected_shape}")
            return False

    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {str(e)[:100]}")
        return False


def main():
    """Quick test of all implementations."""
    print("Quick Block-Sparse Test")
    print("=" * 50)
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print()

    success_count = 0
    total_count = 0

    # 1. Base implementation
    print("1. Testing Base Implementation:")
    model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[2048],
        dilation_rates=[1],
        sparse_config=SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=0.01,
            block_size=64,
        ),
    )
    if test_implementation("BlockSparseBase", model):
        success_count += 1
    total_count += 1
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Multihead implementation
    print("\n2. Testing Multihead Implementation:")
    model = create_multihead_block_sparse(
        embed_dim=512,
        num_heads=8,
        sparsity_ratio=0.05,
        segment_lengths=[2048],
        dilation_rates=[1],
        dtype=torch.float16,
    )
    if test_implementation("BlockSparseMultihead", model, multihead=True):
        success_count += 1
    total_count += 1
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Adaptive implementation
    print("\n3. Testing Adaptive Implementation:")
    model = BlockSparseAdaptive(
        segment_lengths=[2048],
        dilation_rates=[1],
        num_heads=8,
        head_dim=64,
    )
    if test_implementation("BlockSparseAdaptive", model):
        success_count += 1
    total_count += 1
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 4. Test larger sequences
    print("\n4. Testing Larger Sequences (8K tokens):")
    model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[4096],
        dilation_rates=[1],
        sparse_config=SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=0.01,
            block_size=64,
        ),
    )
    if test_implementation("Base-8K", model, seq_len=8192):
        success_count += 1
    total_count += 1
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print("\n" + "=" * 50)
    print(f"Summary: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("\n✅ All implementations working correctly!")
    else:
        print("\n⚠️  Some implementations have issues")

    # Multi-GPU info
    if torch.cuda.device_count() > 1:
        print(f"\n📊 Multi-GPU available: {torch.cuda.device_count()} GPUs")
        print("   Run benchmark_all_block_sparse.py for full multi-GPU tests")
    else:
        print("\n📊 Single GPU mode")
        print("   Multi-GPU tests require 2+ GPUs")


if __name__ == "__main__":
    main()
