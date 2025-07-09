#!/usr/bin/env python3
"""
Simple benchmark for block-sparse implementations.
"""

import torch
import time
import gc
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import (
    create_block_sparse_attention,
)


def measure_memory(func):
    """Measure peak memory usage of a function."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.memory_allocated()
    result = func()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated()
    return result, (peak_mem - start_mem) / 1024**2  # MB


def benchmark_variant(
    name: str, model, seq_len: int, batch_size: int = 2, num_iterations: int = 10
):
    """Benchmark a single variant."""
    print(f"\n{name} (seq_len={seq_len}):")

    # Create inputs
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Move model to device and dtype
    model = model.to(device=device, dtype=dtype)

    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    # Warmup
    for _ in range(3):
        _ = model(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time forward pass
    start = time.time()
    for _ in range(num_iterations):
        output = model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
    forward_time = (time.time() - start) / num_iterations * 1000  # ms

    # Memory measurement
    if device.type == "cuda":

        def forward_fn():
            return model(q, k, v)

        _, forward_memory = measure_memory(forward_fn)
    else:
        forward_memory = 0.0

    # Time backward pass
    output = model(q, k, v)
    loss = output.sum()

    start = time.time()
    for _ in range(num_iterations):
        loss.backward(retain_graph=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    backward_time = (time.time() - start) / num_iterations * 1000  # ms

    print(f"  Forward: {forward_time:.2f} ms")
    print(f"  Backward: {backward_time:.2f} ms")
    print(f"  Memory: {forward_memory:.1f} MB")

    return {
        "forward_ms": forward_time,
        "backward_ms": backward_time,
        "memory_mb": forward_memory,
    }


def main():
    """Run benchmarks on block-sparse variants."""
    print("Block-Sparse Benchmarks")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU (results may not be representative)")

    # Test configurations
    seq_lengths = [4096, 8192]
    batch_size = 2
    num_iterations = 10

    # Variants to test
    variants = {
        "Base (90% sparse)": {
            "factory_kwargs": {
                "variant": "base",
                "segment_lengths": [2048],
                "dilation_rates": [1],
                "sparsity_ratio": 0.1,
            }
        },
        "Hierarchical": {
            "factory_kwargs": {
                "variant": "hierarchical",
                "segment_lengths": [2048],
                "dilation_rates": [1],
            }
        },
        "Adaptive": {
            "factory_kwargs": {
                "variant": "adaptive",
                "segment_lengths": [2048],
                "dilation_rates": [1],
            }
        },
        "Local Window": {
            "factory_kwargs": {
                "variant": "base",
                "segment_lengths": [2048],
                "dilation_rates": [1],
                "pattern_type": "local_window",
                "sparsity_ratio": 0.1,
            }
        },
        "Ultra Sparse (99%)": {
            "factory_kwargs": {
                "variant": "base",
                "segment_lengths": [2048],
                "dilation_rates": [1],
                "sparsity_ratio": 0.01,  # 99% sparse
            }
        },
    }

    results = {}

    for seq_len in seq_lengths:
        print(f"\n\nSequence Length: {seq_len}")
        print("-" * 50)

        results[seq_len] = {}

        for name, config in variants.items():
            try:
                # Create model
                model = create_block_sparse_attention(**config["factory_kwargs"])

                # Run benchmark
                result = benchmark_variant(
                    name, model, seq_len, batch_size, num_iterations
                )
                results[seq_len][name] = result

                # Cleanup
                del model
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\n{name}: FAILED - {str(e)}")
                results[seq_len][name] = {"error": str(e)}

    # Print summary
    print("\n\nSummary")
    print("=" * 50)

    for seq_len, seq_results in results.items():
        print(f"\nSequence Length {seq_len}:")

        # Find fastest
        valid_results = {k: v for k, v in seq_results.items() if "error" not in v}
        if valid_results:
            fastest_forward = min(
                valid_results.items(), key=lambda x: x[1]["forward_ms"]
            )
            fastest_backward = min(
                valid_results.items(), key=lambda x: x[1]["backward_ms"]
            )
            lowest_memory = min(valid_results.items(), key=lambda x: x[1]["memory_mb"])

            print(
                f"  Fastest Forward: {fastest_forward[0]} ({fastest_forward[1]['forward_ms']:.2f} ms)"
            )
            print(
                f"  Fastest Backward: {fastest_backward[0]} ({fastest_backward[1]['backward_ms']:.2f} ms)"
            )
            print(
                f"  Lowest Memory: {lowest_memory[0]} ({lowest_memory[1]['memory_mb']:.1f} MB)"
            )

    print("\n✅ Benchmarks completed!")


if __name__ == "__main__":
    main()
