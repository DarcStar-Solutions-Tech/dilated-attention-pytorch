#!/usr/bin/env python3
"""
Performance comparison of block-sparse implementations.
"""

import torch
import time
import gc
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def time_forward(model, q, k, v, num_iterations=20):
    """Time forward pass."""
    # Warmup
    for _ in range(5):
        _ = model(q, k, v)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = model(q, k, v)
    torch.cuda.synchronize()

    return (time.time() - start) / num_iterations * 1000  # ms


def measure_memory_usage(model, q, k, v):
    """Measure memory usage."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.memory_allocated() / 1024**2  # MB

    # Forward pass
    _ = model(q, k, v)
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
    current_mem = torch.cuda.memory_allocated() / 1024**2  # MB

    return {
        "start_mb": start_mem,
        "peak_mb": peak_mem,
        "current_mb": current_mem,
        "used_mb": peak_mem - start_mem,
    }


def main():
    """Run performance comparison."""
    print("Block-Sparse Performance Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA required for benchmarks")
        return

    device = torch.device("cuda")
    dtype = torch.float16

    # Test configurations
    configs = [
        {"seq_len": 4096, "batch_size": 2},
        {"seq_len": 8192, "batch_size": 1},
        {"seq_len": 16384, "batch_size": 1},
    ]

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        print(f"\n\nSequence Length: {seq_len}, Batch Size: {batch_size}")
        print("-" * 60)

        # Create inputs
        num_heads = 8
        head_dim = 64
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Test different sparsity levels
        sparsity_configs = [
            ("Dense (0% sparse)", 1.0),
            ("90% sparse", 0.1),
            ("95% sparse", 0.05),
            ("99% sparse", 0.01),
        ]

        for name, sparsity_ratio in sparsity_configs:
            try:
                # Create model
                model = create_block_sparse_attention(
                    variant="base",
                    segment_lengths=[2048],
                    dilation_rates=[1],
                    sparsity_ratio=sparsity_ratio,
                    block_size=64,
                ).to(device=device, dtype=dtype)

                # Time forward pass
                forward_time = time_forward(model, q, k, v)

                # Measure memory
                mem_stats = measure_memory_usage(model, q, k, v)

                print(f"\n{name}:")
                print(f"  Forward time: {forward_time:.2f} ms")
                print(f"  Memory used: {mem_stats['used_mb']:.1f} MB")
                print(f"  Peak memory: {mem_stats['peak_mb']:.1f} MB")

                # Cleanup
                del model
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\n{name}: FAILED - {str(e)}")

        # Test hierarchical
        try:
            print("\nHierarchical (multi-scale):")
            model = create_block_sparse_attention(
                variant="hierarchical",
                segment_lengths=[2048],
                dilation_rates=[1],
            ).to(device=device, dtype=dtype)

            forward_time = time_forward(model, q, k, v)
            mem_stats = measure_memory_usage(model, q, k, v)

            print(f"  Forward time: {forward_time:.2f} ms")
            print(f"  Memory used: {mem_stats['used_mb']:.1f} MB")

            # Get sparsity
            stats = model.get_pattern_stats(seq_len)
            print(f"  Sparsity: {stats['sparsity']:.1%}")

            del model
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  FAILED - {str(e)}")

    print("\n\n✅ Performance comparison completed!")


if __name__ == "__main__":
    main()
