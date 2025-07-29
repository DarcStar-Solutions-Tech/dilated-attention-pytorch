#!/usr/bin/env python3
"""
Test that the Enhanced implementation now uses Triton kernel for sparse patterns.
"""

import torch
import time
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimized,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_sparse_pattern(
    impl_class,
    impl_name,
    seq_len,
    dilation_rate,
    batch_size=2,
    hidden_dim=512,
    num_heads=8,
    warmup=3,
    runs=10,
):
    """Benchmark sparse pattern performance."""

    segment_size = 128

    # Create config
    config = {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "segment_size": segment_size,
        "dilation_rate": dilation_rate,
        "hilbert_threshold": 1024,
    }

    if impl_name == "Enhanced":
        config["enable_8k_optimization"] = True
        config["enable_multi_row"] = True

    # Create module and input
    module = impl_class(**config).cuda()
    module.eval()
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        with torch.no_grad():
            out = module(x)
        torch.cuda.synchronize()

    end = time.perf_counter()
    avg_time = (end - start) / runs * 1000  # ms

    return avg_time, out


def main():
    print("=== Sparse Performance Fix Verification ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Test configurations
    test_configs = [
        (2048, 2, "2K sparse (d=2)"),
        (4096, 2, "4K sparse (d=2)"),
        (4096, 4, "4K sparse (d=4)"),
        (8192, 4, "8K sparse (d=4)"),
    ]

    implementations = [
        ("Unified", UnifiedHilbertAttention),
        ("Optimized", UnifiedHilbertAttentionOptimized),
        ("Enhanced", UnifiedHilbertAttentionOptimizedEnhanced),
    ]

    print("Before fix: Enhanced used PyTorch fallback for sparse patterns")
    print("After fix: Enhanced should use Triton kernel like Unified\n")

    results = {}

    for seq_len, dilation, desc in test_configs:
        print(f"\n{desc}:")
        results[desc] = {}

        for impl_name, impl_class in implementations:
            time_ms, out = benchmark_sparse_pattern(
                impl_class, impl_name, seq_len, dilation
            )
            results[desc][impl_name] = time_ms
            print(f"  {impl_name}: {time_ms:.2f}ms")

        # Calculate speedup
        if results[desc]["Unified"] > 0:
            enhanced_speedup = results[desc]["Unified"] / results[desc]["Enhanced"]
            print(f"  Enhanced vs Unified: {enhanced_speedup:.2f}x")

    # Summary table
    print("\n\n=== Performance Summary ===")
    print(
        f"{'Config':<20} | {'Unified':<10} | {'Optimized':<10} | {'Enhanced':<10} | {'Status':<15}"
    )
    print("-" * 75)

    for desc in results:
        unified_time = results[desc]["Unified"]
        optimized_time = results[desc]["Optimized"]
        enhanced_time = results[desc]["Enhanced"]

        # Check if Enhanced is now competitive
        if enhanced_time <= unified_time * 1.2:  # Within 20% is good
            status = "✓ Fixed"
        else:
            status = "✗ Still slow"

        print(
            f"{desc:<20} | {unified_time:>8.2f}ms | {optimized_time:>8.2f}ms | {enhanced_time:>8.2f}ms | {status}"
        )

    # Verify outputs match
    print("\n\nVerifying output correctness...")

    # Test one configuration
    seq_len, dilation = 2048, 2
    batch_size = 1
    hidden_dim = 256
    num_heads = 8

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Create modules
    unified = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=dilation,
        )
        .cuda()
        .eval()
    )

    enhanced = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=dilation,
            enable_8k_optimization=True,
        )
        .cuda()
        .eval()
    )

    with torch.no_grad():
        out_unified = unified(x)
        out_enhanced = enhanced(x)

    # Check similarity
    max_diff = torch.max(torch.abs(out_unified - out_enhanced)).item()
    print(f"Max difference between Unified and Enhanced: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Outputs match!")
    else:
        print("✗ Outputs differ significantly")

    print("\n\n=== Conclusion ===")

    # Calculate average improvement
    total_improvement = 0
    count = 0
    for desc in results:
        if results[desc]["Enhanced"] > 0 and results[desc]["Unified"] > 0:
            improvement = results[desc]["Unified"] / results[desc]["Enhanced"]
            total_improvement += improvement
            count += 1

    avg_improvement = total_improvement / count if count > 0 else 0

    if avg_improvement > 0.8:  # If Enhanced is at least 80% as fast as Unified
        print(
            "✓ SUCCESS: Enhanced implementation now uses efficient Triton kernel for sparse patterns!"
        )
        print(f"  Average performance vs Unified: {avg_improvement:.2f}x")
    else:
        print("✗ ISSUE: Enhanced implementation still slower than expected")
        print("  May need further investigation")


if __name__ == "__main__":
    main()
