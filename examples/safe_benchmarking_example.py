#!/usr/bin/env python3
"""
Example of how to use the safe benchmarking utilities.

This example shows:
1. How to check memory before allocation
2. How to use progressive testing
3. How to run benchmarks with safety limits
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.core.utils.safety import (
    SafetyConfig,
    MemorySafetyChecker,
    ProgressiveTester,
    SafeBenchmarkRunner,
    check_memory_before_allocation,
)
from dilated_attention_pytorch import DilatedAttention


def example_1_memory_checking():
    """Example 1: Check memory before allocation."""
    print("=== Example 1: Memory Checking ===\n")

    # Check if we can allocate a large tensor
    batch_size = 4
    seq_len = 32768  # 32K sequence
    num_heads = 16
    head_dim = 64

    shape = (batch_size, seq_len, num_heads, head_dim)

    print(f"Checking if we can allocate tensor of shape {shape}")
    print(
        f"Estimated size: {batch_size * seq_len * num_heads * head_dim * 4 / 1e9:.2f} GB (float32)"
    )

    can_allocate = check_memory_before_allocation(
        shape,
        dtype=torch.float32,
        num_tensors=3,  # Q, K, V
    )

    if can_allocate:
        print("✓ Memory check passed - safe to allocate\n")
    else:
        print("✗ Memory check failed - allocation would exceed limits\n")


def example_2_progressive_testing():
    """Example 2: Progressive testing to find limits."""
    print("=== Example 2: Progressive Testing ===\n")

    def test_attention(seq_len, batch_size=2):
        """Test dilated attention at given sequence length."""
        model = DilatedAttention(
            segment_lengths=[1024, 2048, 4096], dilation_rates=[1, 2, 4]
        )

        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # Create inputs
        num_heads = 8
        head_dim = 64
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        output = model(q, k, v)

        print(f"  Successfully processed seq_len={seq_len}")
        return output

    # Create progressive tester
    tester = ProgressiveTester()

    # Test with progressively larger sequences
    print("Testing with progressive sequence lengths:")
    try:
        _ = tester.test_with_safety(
            test_attention,
            {"batch_size": 2},
            size_param_name="seq_len",
            target_size=32768,  # Target 32K sequence
        )
        print("\nProgressive testing completed successfully!\n")
    except Exception as e:
        print(f"\nProgressive testing stopped: {e}\n")


def example_3_safe_benchmarking():
    """Example 3: Run benchmarks with safety limits."""
    print("=== Example 3: Safe Benchmarking ===\n")

    def benchmark_attention(seq_len, batch_size, num_heads=8):
        """Benchmark function."""
        import time

        model = DilatedAttention(
            segment_lengths=[1024, 2048, 4096], dilation_rates=[1, 2, 4]
        )

        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # Create inputs
        head_dim = 64
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(3):
            _ = model(q, k, v)

        if device == "cuda":
            torch.cuda.synchronize()

        # Time it
        start = time.time()
        for _ in range(10):
            _ = model(q, k, v)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start
        avg_time = elapsed / 10

        print(
            f"  seq_len={seq_len}, batch={batch_size}: {avg_time * 1000:.1f}ms per forward"
        )

        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "time_ms": avg_time * 1000,
        }

    # Configure safety limits
    config = SafetyConfig(
        max_memory_fraction=0.7,  # Use max 70% of GPU
        min_free_memory_gb=2.0,  # Keep 2GB free
        progressive_steps=4,  # 4 progressive steps
    )

    # Create benchmark runner
    runner = SafeBenchmarkRunner(config)

    # Define benchmark configurations
    configs = [
        {"seq_len": 8192, "batch_size": 4},
        {"seq_len": 16384, "batch_size": 2},
        {"seq_len": 32768, "batch_size": 1},
    ]

    print("Running benchmarks with safety limits:")
    results = runner.run_benchmark(
        benchmark_attention, configs, size_param_name="seq_len"
    )

    print("\nBenchmark Results:")
    for i, result in enumerate(results):
        if result:
            print(f"  Config {i + 1}: {result['time_ms']:.1f}ms")
        else:
            print(f"  Config {i + 1}: Failed")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Safe Benchmarking Examples")
    print("=" * 60 + "\n")

    # Show system info
    checker = MemorySafetyChecker()
    if torch.cuda.is_available():
        used, free, total = checker.get_gpu_memory_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total\n")
    else:
        print("No GPU available, using CPU\n")

    # Run examples
    example_1_memory_checking()
    example_2_progressive_testing()
    example_3_safe_benchmarking()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
