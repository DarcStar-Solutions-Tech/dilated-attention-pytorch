"""
Test and benchmark the optimized unfold implementation
"""

import time

import torch

from dilated_attention_pytorch import RingDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention_unfold_optimized import (
    OptimizedUnfoldRingDilatedAttention,
)


def benchmark_all_implementations():
    """Benchmark all implementations including the optimized unfold version."""
    print("Comprehensive Benchmark: Original vs Optimized Unfold")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test configurations
    configs = [
        (1, 2048, 8, 64, [512, 1024, 2048], [1, 2, 4], "Small"),
        (1, 8192, 12, 64, [2048, 4096, 8192], [1, 2, 4], "Medium"),
        (1, 16384, 16, 64, [4096, 8192, 16384], [1, 2, 4], "Large"),
    ]

    results = []

    for (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        segments,
        dilations,
        size_name,
    ) in configs:
        print(f"\n{size_name} ({seq_len:,} tokens):")

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        # Create modules
        orig_module = RingDilatedAttention(segments, dilations, 0.0, ring_size=1).to(device, dtype)
        opt_unfold_module = OptimizedUnfoldRingDilatedAttention(
            segments, dilations, 0.0, ring_size=1
        ).to(device, dtype)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = orig_module(q, k, v)
                _ = opt_unfold_module(q, k, v)

        # Benchmark original
        torch.cuda.synchronize()
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            with torch.no_grad():
                _ = orig_module(q, k, v)
        torch.cuda.synchronize()
        orig_time = (time.time() - start) / iterations * 1000

        # Benchmark optimized unfold
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _ = opt_unfold_module(q, k, v)
        torch.cuda.synchronize()
        opt_time = (time.time() - start) / iterations * 1000

        # Verify correctness
        with torch.no_grad():
            orig_result = orig_module(q, k, v)
            opt_result = opt_unfold_module(q, k, v)

        max_diff = (orig_result - opt_result).abs().max().item()

        # Results
        speedup = orig_time / opt_time
        print(f"  Original (index_select): {orig_time:.2f}ms")
        print(f"  Optimized unfold: {opt_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max difference: {max_diff:.2e}")

        results.append(
            {
                "size": size_name,
                "seq_len": seq_len,
                "orig_time": orig_time,
                "opt_time": opt_time,
                "speedup": speedup,
                "correct": max_diff < 1e-3,
            }
        )

        # Memory comparison
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = orig_module(q, k, v)
            orig_memory = torch.cuda.max_memory_allocated() / (1024**2)

            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = opt_unfold_module(q, k, v)
            opt_memory = torch.cuda.max_memory_allocated() / (1024**2)

            print(f"  Memory - Original: {orig_memory:.1f}MB, Optimized: {opt_memory:.1f}MB")
            print(f"  Memory reduction: {(1 - opt_memory / orig_memory) * 100:.1f}%")

        # Cleanup
        del q, k, v, orig_module, opt_unfold_module
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(
        f"{'Size':<10} {'Seq Len':<10} {'Original':<12} {'Optimized':<12} {'Speedup':<10} {'Correct':<10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['size']:<10} {r['seq_len']:<10} {r['orig_time']:<12.2f} {r['opt_time']:<12.2f} {r['speedup']:<10.2f} {'✓' if r['correct'] else '✗':<10}"
        )

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 80)
    print(f"Average speedup: {avg_speedup:.2f}x")


def test_specific_optimizations():
    """Test specific optimization paths."""
    print("\n\nTesting Specific Optimization Paths")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Test offset=0 optimization (should be fastest)
    print("\n1. Testing offset=0 optimization (pure strided slicing):")

    segments = [256, 512, 1024]
    dilations = [1, 2, 4]  # offset will be 0 for first group

    q = torch.randn(1, 1024, 3, 64, device=device, dtype=dtype)  # 3 heads = 1 head per group
    k = torch.randn(1, 1024, 3, 64, device=device, dtype=dtype)
    v = torch.randn(1, 1024, 3, 64, device=device, dtype=dtype)

    opt_module = OptimizedUnfoldRingDilatedAttention(segments, dilations, 0.0).to(device, dtype)

    # Time just the dilated attention block
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = opt_module(q, k, v)
    torch.cuda.synchronize()
    time_offset0 = (time.time() - start) / 100 * 1000

    print(f"  Time with offset=0 optimization: {time_offset0:.2f}ms")

    # Test non-zero offset
    print("\n2. Testing non-zero offset optimization:")

    # Use 6 heads to trigger different offsets
    q = torch.randn(1, 1024, 6, 64, device=device, dtype=dtype)
    k = torch.randn(1, 1024, 6, 64, device=device, dtype=dtype)
    v = torch.randn(1, 1024, 6, 64, device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = opt_module(q, k, v)
    torch.cuda.synchronize()
    time_offset_nonzero = (time.time() - start) / 100 * 1000

    print(f"  Time with non-zero offset: {time_offset_nonzero:.2f}ms")
    print(f"  Offset=0 is {time_offset_nonzero / time_offset0:.2f}x faster")


if __name__ == "__main__":
    benchmark_all_implementations()
    test_specific_optimizations()
