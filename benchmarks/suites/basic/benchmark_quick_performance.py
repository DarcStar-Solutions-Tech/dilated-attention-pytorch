#!/usr/bin/env python3
"""
Quick performance benchmark for dilated attention implementations.
"""

import time
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import create_multihead_dilated_attention
from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def time_forward_pass(model, inputs, num_runs=5, warmup=2):
    """Time forward pass with warmup."""
    q, k, v = inputs

    # Warmup
    for _ in range(warmup):
        _ = model(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time runs
    start = time.time()
    for _ in range(num_runs):
        _ = model(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    end = time.time()

    return (end - start) / num_runs


def benchmark_implementation(name, model_fn, seq_len, batch_size=2):
    """Benchmark a single implementation."""
    print(f"\n{name} (seq_len={seq_len:,}):")

    # Create model
    try:
        model = model_fn()
    except Exception as e:
        print(f"  ✗ Failed to create: {e}")
        return None

    # Setup inputs
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = model.to(device=device, dtype=dtype)

    # For multihead attention, input should be (batch, seq_len, embed_dim)
    embed_dim = num_heads * head_dim
    q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Measure memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    # Time forward pass
    try:
        avg_time = time_forward_pass(model, (q, k, v), num_runs=5)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            mem_used = (peak_mem - start_mem) / 1024**2  # MB
            print(f"  ✓ Time: {avg_time * 1000:.1f}ms | Memory: {mem_used:.1f}MB")
        else:
            print(f"  ✓ Time: {avg_time * 1000:.1f}ms")

        return avg_time

    except Exception as e:
        print(f"  ✗ Forward failed: {e}")
        return None


def main():
    """Run quick benchmarks."""
    print("=== Quick Performance Benchmark ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Test configurations
    seq_lengths = [1024, 4096, 8192]

    # Define implementations to test
    implementations = [
        # Standard implementation has been removed, use improved as baseline
        (
            "Improved Dilated",
            lambda: create_multihead_dilated_attention(
                "improved",
                embed_dim=512,
                num_heads=8,
                segment_lengths=[512, 1024, 2048],
                dilation_rates=[1, 2, 4],
            ),
        ),
        (
            "Block-Sparse (90% sparse)",
            lambda: create_block_sparse_attention(
                variant="base",  # Use base variant
                embed_dim=512,
                num_heads=8,
                segment_lengths=[512, 1024, 2048],
                dilation_rates=[1, 2, 4],
                sparsity_ratio=0.1,
            ),
        ),
        (
            "Block-Sparse (95% sparse)",
            lambda: create_block_sparse_attention(
                variant="base",  # Use base variant
                embed_dim=512,
                num_heads=8,
                segment_lengths=[512, 1024, 2048],
                dilation_rates=[1, 2, 4],
                sparsity_ratio=0.05,
            ),
        ),
    ]

    # Run benchmarks
    results = {}
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len:,} ---")
        results[seq_len] = {}

        for name, model_fn in implementations:
            time_taken = benchmark_implementation(name, model_fn, seq_len)
            results[seq_len][name] = time_taken

    # Summary
    print("\n=== Summary ===")
    print("Speedup vs Improved Dilated:")
    for seq_len in seq_lengths:
        print(f"\nSeq Length {seq_len:,}:")
        baseline_time = results[seq_len].get("Improved Dilated")
        if baseline_time:
            for name, time_taken in results[seq_len].items():
                if time_taken and name != "Improved Dilated":
                    speedup = baseline_time / time_taken
                    print(f"  {name}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
