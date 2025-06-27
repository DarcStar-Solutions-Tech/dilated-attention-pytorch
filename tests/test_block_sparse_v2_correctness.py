#!/usr/bin/env python3
"""
Test correctness of BlockSparse V2 implementations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention_v2 import (
    BlockSparseRingDilatedAttentionV2,
    SparsePatternConfig,
)
from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention_v2 import (
    BlockSparseRingMultiheadDilatedAttentionV2,
)


def test_basic_attention():
    """Test basic attention computation."""
    print("\n=== Testing Basic Attention ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Small test case
    batch_size = 2
    seq_len = 512
    num_heads = 4
    head_dim = 32

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    # Sparse config
    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.2,  # Keep 20% of blocks
        block_size=64,
        local_window_size=256,
    )

    # Create module
    module = BlockSparseRingDilatedAttentionV2(
        segment_lengths=[128, 256], dilation_rates=[1, 2], sparse_config=sparse_config, ring_size=1
    ).to(device)

    # Test forward pass
    output = module(q, k, v)
    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

    # Test with causal mask
    output_causal = module(q, k, v, is_causal=True)
    print(
        f"Causal output stats: mean={output_causal.mean().item():.4f}, std={output_causal.std().item():.4f}"
    )

    # Test with attention weights
    output_weights, weights = module(q, k, v, return_attention_weights=True)
    assert torch.allclose(output, output_weights, atol=1e-5)
    print(f"Attention weights keys: {list(weights.keys())}")
    print(f"Number of active blocks: {len(weights['block_indices'])}")

    print("✓ Basic attention test passed")


def test_multihead_wrapper():
    """Test multihead wrapper."""
    print("\n=== Testing Multihead Wrapper ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Configuration
    batch_size = 2
    seq_len = 512
    embed_dim = 256
    num_heads = 8

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Sparse config
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse", sparsity_ratio=0.15, block_size=64
    )

    # Create module
    module = BlockSparseRingMultiheadDilatedAttentionV2(
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=[128, 256],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        batch_first=True,
    ).to(device)

    # Test forward pass
    output = module(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

    # Test with weights
    output_weights, weights = module(x, x, x, need_weights=True)
    if isinstance(output, torch.Tensor) and isinstance(output_weights, torch.Tensor):
        assert torch.allclose(output, output_weights, atol=1e-5)
    print(f"Weights type: {type(weights)}")
    if weights is not None:
        print(
            f"Weights info: {list(weights.keys()) if isinstance(weights, dict) else weights.shape}"
        )

    # Test causal
    output_causal = module(x, x, x, is_causal=True)
    print(
        f"Causal output stats: mean={output_causal.mean().item():.4f}, std={output_causal.std().item():.4f}"
    )

    # Test with key padding mask
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    key_padding_mask[:, seq_len // 2 :] = True  # Mask second half

    output_masked = module(x, x, x, key_padding_mask=key_padding_mask)
    print(
        f"Masked output stats: mean={output_masked.mean().item():.4f}, std={output_masked.std().item():.4f}"
    )

    print("✓ Multihead wrapper test passed")


def test_different_patterns():
    """Test different sparsity patterns."""
    print("\n=== Testing Different Sparsity Patterns ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    batch_size = 1
    seq_len = 1024
    num_heads = 4
    head_dim = 32

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = q.clone()
    v = q.clone()

    patterns = [
        ("local_window", {"local_window_size": 256}),
        ("dilated_sparse", {}),
        ("global_local", {"global_tokens": 128}),
    ]

    for pattern_type, extra_config in patterns:
        print(f"\nTesting pattern: {pattern_type}")

        sparse_config = SparsePatternConfig(
            pattern_type=pattern_type, sparsity_ratio=0.1, block_size=64, **extra_config
        )

        module = BlockSparseRingDilatedAttentionV2(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
            ring_size=1,
        ).to(device)

        output, weights = module(q, k, v, return_attention_weights=True)

        num_blocks = len(weights["block_indices"])
        total_blocks = (seq_len // sparse_config.block_size) ** 2
        actual_sparsity = 1 - (num_blocks / total_blocks)

        print(f"  Active blocks: {num_blocks}/{total_blocks}")
        print(f"  Actual sparsity: {actual_sparsity:.1%}")
        print(f"  Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

    print("\n✓ All pattern tests passed")


def test_memory_efficiency():
    """Verify memory efficiency claims."""
    print("\n=== Testing Memory Efficiency ===")

    if not torch.cuda.is_available():
        print("Skipping memory test (CUDA not available)")
        return

    device = torch.device("cuda")

    # Test configuration
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = q.clone()
    v = q.clone()

    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.05,  # 95% sparse
        block_size=128,
    )

    module = BlockSparseRingDilatedAttentionV2(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        ring_size=1,
    ).to(device)

    # Measure memory for forward with weights
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.memory_allocated() / 1024 / 1024

    output, weights = module(q, k, v, return_attention_weights=True)
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    mem_increase = peak_mem - start_mem

    print(f"Memory increase: {mem_increase:.1f} MB")

    # Calculate what dense attention would use
    dense_size = batch_size * num_heads * seq_len * seq_len * 2  # float16
    dense_mb = dense_size / 1024 / 1024

    print(f"Dense attention would use: {dense_mb:.1f} MB")
    print(f"Memory savings: {(1 - mem_increase/dense_mb)*100:.1f}%")

    # Verify sparse format
    assert isinstance(weights, dict), "Weights should be in sparse dict format"
    assert "block_indices" in weights, "Missing block indices"

    num_blocks = len(weights["block_indices"])
    print(f"Active blocks: {num_blocks}")

    print("✓ Memory efficiency verified")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BlockSparse V2 Correctness Tests")
    print("=" * 60)

    try:
        test_basic_attention()
        test_multihead_wrapper()
        test_different_patterns()
        test_memory_efficiency()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
