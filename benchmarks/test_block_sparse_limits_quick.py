#!/usr/bin/env python3
"""
Quick test of block-sparse sequence limits and performance.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import gc
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def test_variant(name, config, seq_lengths):
    """Test a variant at different sequence lengths."""
    print(f"\n{name}:")
    print(f"{'Seq Len':>10} {'Success':>10} {'Time (ms)':>12} {'Memory (MB)':>12}")
    print("-" * 46)

    max_len = 0

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Create model and inputs
            model = create_block_sparse_attention(**config)
            model = model.to(device="cuda", dtype=torch.float16)

            batch_size = 1
            num_heads = 8
            head_dim = 64

            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Measure memory before
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**2

            # Time forward pass
            torch.cuda.synchronize()
            start = time.time()
            output = model(q, k, v)
            torch.cuda.synchronize()
            forward_time = (time.time() - start) * 1000

            # Measure memory after
            mem_after = torch.cuda.memory_allocated() / 1024**2
            mem_used = mem_after - mem_before

            print(f"{seq_len:>10,} {'✓':>10} {forward_time:>12.1f} {mem_used:>12.1f}")
            max_len = seq_len

            del model, q, k, v, output

        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:>10,} {'OOM':>10} {'-':>12} {'-':>12}")
            break
        except Exception as e:
            print(f"{seq_len:>10,} {'ERROR':>10} {'-':>12} {'-':>12}")
            print(f"           Error: {str(e)[:50]}...")
            break

    return max_len


def main():
    """Run quick sequence limit tests."""
    print("Block-Sparse Sequence Limits (Quick Test)")
    print("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    # Test sequence lengths (fewer for quick test)
    test_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

    # Variants to test
    variants = {
        "Dense (baseline)": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 1.0,
        },
        "90% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.1,
        },
        "95% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.05,
        },
        "99% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.01,
        },
        "Hierarchical": {
            "variant": "hierarchical",
            "segment_lengths": [2048],
            "dilation_rates": [1],
        },
        "Adaptive": {
            "variant": "adaptive",
            "segment_lengths": [2048],
            "dilation_rates": [1],
        },
    }

    # Test each variant
    max_lengths = {}
    for name, config in variants.items():
        max_len = test_variant(name, config, test_lengths)
        max_lengths[name] = max_len

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Variant':<20} {'Max Sequence Length':>20}")
    print("-" * 40)

    for name, max_len in sorted(max_lengths.items(), key=lambda x: x[1], reverse=True):
        if max_len > 0:
            print(f"{name:<20} {max_len:>20,}")
        else:
            print(f"{name:<20} {'Failed':>20}")

    # Key insights
    print("\nKey Insights:")
    dense_max = max_lengths.get("Dense (baseline)", 1)

    for name, max_len in max_lengths.items():
        if name != "Dense (baseline)" and max_len > 0:
            improvement = max_len / dense_max
            print(f"- {name}: {improvement:.1f}x longer than dense")

    # Performance analysis
    print("\nPerformance Notes:")
    print("- Block-sparse has overhead for short sequences (<16K)")
    print("- Benefits increase with sequence length")
    print("- Memory savings enable longer sequences")
    print("- Actual speedup depends on sparsity pattern efficiency")

    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    main()
