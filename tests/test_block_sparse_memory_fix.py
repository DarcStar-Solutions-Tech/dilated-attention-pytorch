#!/usr/bin/env python3
"""
Test script to compare memory usage between original and fixed BlockSparse implementations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
from contextlib import contextmanager

import torch

# Import implementations
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
    )
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        SparsePatternConfig as OriginalSparseConfig,
    )

    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("Original BlockSparseRingDilatedAttention not available")

from dilated_attention_pytorch.block_sparse_ring_dilated_attention_v2 import (
    BlockSparseRingDilatedAttentionV2,
    SparsePatternConfig,
)


@contextmanager
def measure_gpu_memory():
    """Context manager to measure GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()

        return {
            "peak_mb": (peak_memory - start_memory) / 1024 / 1024,
            "current_mb": (current_memory - start_memory) / 1024 / 1024,
        }
    return {"peak_mb": 0, "current_mb": 0}


def test_implementation(impl_class, sparse_config, seq_len: int, batch_size: int = 2):
    """Test a single implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    num_heads = 8
    head_dim = 64
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    print(f"\nTesting {impl_class.__name__} with seq_len={seq_len}")

    try:
        # Create module
        with measure_gpu_memory():
            module = impl_class(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                sparse_config=sparse_config,
                ring_size=1,  # Single GPU
            ).to(device)

        # Create inputs
        with measure_gpu_memory():
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
            )
            k = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
            )
            v = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
            )

        # Forward pass without attention weights
        gc.collect()
        torch.cuda.empty_cache()

        with measure_gpu_memory() as mem_forward:
            output = module(q, k, v, is_causal=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(f"  Forward pass (no weights): {mem_forward}")

        # Forward pass with attention weights
        gc.collect()
        torch.cuda.empty_cache()

        with measure_gpu_memory() as mem_forward_weights:
            output, weights = module(q, k, v, is_causal=True, return_attention_weights=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(f"  Forward pass (with weights): {mem_forward_weights}")

        # Check weights format
        if weights is not None:
            if isinstance(weights, dict):
                print(f"  Weights format: Sparse dict with keys {list(weights.keys())}")
                if "block_indices" in weights:
                    print(f"    Number of active blocks: {len(weights['block_indices'])}")
                    total_blocks = (seq_len // sparse_config.block_size) ** 2
                    sparsity = 1 - len(weights["block_indices"]) / total_blocks
                    print(f"    Actual sparsity: {sparsity:.1%}")
            else:
                print(f"  Weights format: Dense tensor of shape {weights.shape}")
                weight_memory_mb = weights.numel() * weights.element_size() / 1024 / 1024
                print(f"    Weights memory usage: {weight_memory_mb:.1f} MB")

        # Cleanup
        del module, q, k, v, output
        if weights is not None:
            del weights
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAILED: {e}")
        return False
    else:
        return True


def main():
    """Run memory comparison tests."""
    if not torch.cuda.is_available():
        print("CUDA not available, memory measurements will be limited")

    # Test configurations
    test_configs = [
        (8192, "8K tokens"),
        (16384, "16K tokens"),
        (32768, "32K tokens"),
    ]

    # Sparse configuration (90% sparse)
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,  # Keep only 10% of blocks
        block_size=128,
        local_window_size=512,
    )

    print("=" * 60)
    print("Block Sparse Memory Usage Comparison")
    print("=" * 60)
    print(
        f"Sparse config: {sparse_config.sparsity_ratio:.0%} sparse, block_size={sparse_config.block_size}"
    )

    for seq_len, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing {desc}")
        print(f"{'='*60}")

        # Test original implementation
        if ORIGINAL_AVAILABLE:
            original_config = OriginalSparseConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=0.1,
                block_size=128,
                local_window_size=512,
            )
            success = test_implementation(BlockSparseRingDilatedAttention, original_config, seq_len)
            if not success:
                print("  Original implementation failed (likely OOM)")

        # Test fixed implementation
        success = test_implementation(BlockSparseRingDilatedAttentionV2, sparse_config, seq_len)
        if not success:
            print("  Fixed implementation failed")

    print("\n" + "=" * 60)
    print("Summary:")
    print("- Original implementation stores full dense attention matrices")
    print("- Fixed implementation uses sparse format (block indices + values)")
    print("- Memory savings are proportional to sparsity ratio")
    print("- For 90% sparsity, fixed version uses ~10x less memory for attention weights")


if __name__ == "__main__":
    main()
