#!/usr/bin/env python3
"""
Summary benchmark showing block-sparse performance benefits.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU

import torch
import time
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def benchmark_sparsity_levels():
    """Benchmark different sparsity levels."""
    print("Block-Sparse Performance Summary")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test configuration
    batch_size = 2
    seq_len = 4096
    num_heads = 8
    head_dim = 64
    num_iterations = 10

    print("\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print("\n" + "-" * 60)
    print(f"{'Variant':<25} {'Forward (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print("-" * 60)

    results = []
    baseline_time = None

    # Test different configurations
    configs = [
        ("Dense (baseline)", {"sparsity_ratio": 1.0}),
        ("90% sparse", {"sparsity_ratio": 0.1}),
        ("95% sparse", {"sparsity_ratio": 0.05}),
        ("99% sparse", {"sparsity_ratio": 0.01}),
        # Hierarchical removed - use dilated_sparse with high sparsity instead
    ]

    for name, kwargs in configs:
        try:
            # Default kwargs
            default_kwargs = {
                "variant": "base",
                "segment_lengths": [2048],
                "dilation_rates": [1],
            }
            default_kwargs.update(kwargs)

            # Create model
            model = create_block_sparse_attention(**default_kwargs)
            model = model.to(device=device, dtype=dtype)

            # Warmup
            for _ in range(3):
                _ = model(q, k, v)
            torch.cuda.synchronize()

            # Measure time
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iterations):
                _ = model(q, k, v)
            torch.cuda.synchronize()
            forward_time = (time.time() - start) / num_iterations * 1000

            # Measure memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = model(q, k, v)
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2

            # Calculate speedup
            if baseline_time is None:
                baseline_time = forward_time
                speedup = 1.0
            else:
                speedup = baseline_time / forward_time

            print(
                f"{name:<25} {forward_time:<15.2f} {memory_mb:<15.1f} {speedup:<10.2f}x"
            )

            results.append(
                {
                    "name": name,
                    "forward_ms": forward_time,
                    "memory_mb": memory_mb,
                    "speedup": speedup,
                }
            )

            # Cleanup
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{name:<25} {'FAILED':<15} {str(e):<25}")

    # Summary
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("-" * 60)

    if len(results) > 1:
        fastest = min(results[1:], key=lambda x: x["forward_ms"])
        most_memory_efficient = min(results[1:], key=lambda x: x["memory_mb"])

        print(f"✓ Fastest: {fastest['name']} ({fastest['speedup']:.2f}x speedup)")
        print(f"✓ Most memory efficient: {most_memory_efficient['name']} ")
        print(
            f"  ({(1 - most_memory_efficient['memory_mb'] / results[0]['memory_mb']) * 100:.1f}% memory reduction)"
        )

        # Show sparsity benefits
        sparse_results = [r for r in results if "sparse" in r["name"]]
        if sparse_results:
            avg_speedup = sum(r["speedup"] for r in sparse_results) / len(
                sparse_results
            )
            print(f"✓ Average speedup from sparsity: {avg_speedup:.2f}x")

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    benchmark_sparsity_levels()
