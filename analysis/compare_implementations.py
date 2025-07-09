#!/usr/bin/env python3
"""
Compare DilatedAttention vs ImprovedDilatedAttention implementations
"""

import gc
import time
import tracemalloc
from typing import Any

import torch

from dilated_attention_pytorch.dilated_attention import DilatedAttention
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def benchmark_implementation(
    attention_class,
    segment_lengths: list[int],
    dilation_rates: list[int],
    batch_size: int = 1,
    seq_len: int = 8192,
    num_heads: int = 8,
    embed_dim: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Benchmark a single attention implementation"""

    # Initialize model
    if attention_class == DilatedAttention:
        model = attention_class(segment_lengths, dilation_rates)
    else:  # ImprovedDilatedAttention
        model = attention_class(segment_lengths, dilation_rates)

    model = model.to(device).to(dtype)

    # Create test data
    q = torch.randn(
        batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype
    )

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v)

    torch.cuda.synchronize()

    # Memory tracking
    tracemalloc.start()
    torch.cuda.reset_peak_memory_stats()

    # Timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            output = model(q, k, v)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # Memory usage
    gpu_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    cpu_memory = tracemalloc.get_traced_memory()[1] / 1024**2  # MB
    tracemalloc.stop()

    # Store output shape before deletion
    output_shape = list(output.shape)

    # Clear memory
    del model, q, k, v, output
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "gpu_memory_mb": gpu_memory,
        "cpu_memory_mb": cpu_memory,
        "output_shape": output_shape,
    }


def compare_implementations():
    """Compare both implementations across different configurations"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device}, Dtype: {dtype}")
    print("=" * 80)

    # Test configurations
    configs = [
        {
            "name": "Small",
            "segment_lengths": [2048, 4096],
            "dilation_rates": [1, 2],
            "seq_len": 4096,
            "num_heads": 4,
            "embed_dim": 64,
        },
        {
            "name": "Medium",
            "segment_lengths": [2048, 4096, 8192],
            "dilation_rates": [1, 2, 4],
            "seq_len": 8192,
            "num_heads": 8,
            "embed_dim": 64,
        },
        {
            "name": "Large",
            "segment_lengths": [2048, 4096, 8192, 16384],
            "dilation_rates": [1, 2, 4, 8],
            "seq_len": 16384,
            "num_heads": 12,
            "embed_dim": 64,
        },
    ]

    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  Segment lengths: {config['segment_lengths']}")
        print(f"  Dilation rates: {config['dilation_rates']}")
        print(f"  Sequence length: {config['seq_len']}")
        print(f"  Num heads: {config['num_heads']}")
        print(f"  Embed dim: {config['embed_dim']}")

        # Test DilatedAttention
        print("\nDilatedAttention:")
        try:
            results_orig = benchmark_implementation(
                DilatedAttention,
                config["segment_lengths"],
                config["dilation_rates"],
                seq_len=config["seq_len"],
                num_heads=config["num_heads"],
                embed_dim=config["embed_dim"],
                device=device,
                dtype=dtype,
            )
            print(
                f"  Time: {results_orig['avg_time']:.4f}s ± {results_orig['max_time'] - results_orig['min_time']:.4f}s"
            )
            print(f"  GPU Memory: {results_orig['gpu_memory_mb']:.1f} MB")
            print(f"  CPU Memory: {results_orig['cpu_memory_mb']:.1f} MB")
        except Exception as e:
            print(f"  Error: {e}")
            results_orig = None

        # Test ImprovedDilatedAttention
        print("\nImprovedDilatedAttention:")
        try:
            results_improved = benchmark_implementation(
                ImprovedDilatedAttention,
                config["segment_lengths"],
                config["dilation_rates"],
                seq_len=config["seq_len"],
                num_heads=config["num_heads"],
                embed_dim=config["embed_dim"],
                device=device,
                dtype=dtype,
            )
            print(
                f"  Time: {results_improved['avg_time']:.4f}s ± {results_improved['max_time'] - results_improved['min_time']:.4f}s"
            )
            print(f"  GPU Memory: {results_improved['gpu_memory_mb']:.1f} MB")
            print(f"  CPU Memory: {results_improved['cpu_memory_mb']:.1f} MB")
        except Exception as e:
            print(f"  Error: {e}")
            results_improved = None

        # Compare if both successful
        if results_orig and results_improved:
            time_speedup = results_orig["avg_time"] / results_improved["avg_time"]
            memory_ratio = (
                results_improved["gpu_memory_mb"] / results_orig["gpu_memory_mb"]
            )
            print("\nComparison:")
            print(
                f"  Speedup: {time_speedup:.2f}x {'(Improved faster)' if time_speedup > 1 else '(Original faster)'}"
            )
            print(
                f"  Memory ratio: {memory_ratio:.2f}x {'(Improved uses more)' if memory_ratio > 1 else '(Improved uses less)'}"
            )

        print("-" * 60)


def functional_comparison():
    """Compare the functional differences between implementations"""
    print("\n" + "=" * 80)
    print("FUNCTIONAL COMPARISON")
    print("=" * 80)

    # Key differences analysis
    print("Key Implementation Differences:")
    print("1. Attention Backend:")
    print("   - DilatedAttention: Uses xformers.ops.memory_efficient_attention")
    print(
        "   - ImprovedDilatedAttention: Uses F.scaled_dot_product_attention with automatic backend selection"
    )

    print("\n2. Optimization Features:")
    print("   - DilatedAttention: Manual memory management, explicit device handling")
    print(
        "   - ImprovedDilatedAttention: TF32 support, torch.compile integration, automatic backend selection"
    )

    print("\n3. Code Structure:")
    print(
        "   - DilatedAttention: More explicit tensor operations, step-by-step approach"
    )
    print("   - ImprovedDilatedAttention: More concise, optimized tensor operations")

    print("\n4. Error Handling:")
    print(
        "   - DilatedAttention: Explicit error checking for segment/dilation length mismatch"
    )
    print(
        "   - ImprovedDilatedAttention: Uses assertions, skips segments that are too large"
    )

    print("\n5. Performance Optimizations:")
    print("   - DilatedAttention: Relies on xformers optimizations")
    print(
        "   - ImprovedDilatedAttention: Multiple optimizations (TF32, torch.compile, SDPA backends)"
    )


if __name__ == "__main__":
    functional_comparison()
    compare_implementations()
