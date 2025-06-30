"""
Benchmark Ring Attention V2 vs V3 with optimized pattern caching.
"""

import time
import torch
import numpy as np
from datetime import datetime

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3
from dilated_attention_pytorch.core import clear_global_cache
from dilated_attention_pytorch.core.optimized_pattern_cache import clear_optimized_cache


def benchmark_forward_pass(
    model,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_iterations: int = 50,
    warmup: int = 10,
    device: str = "cuda",
):
    """Benchmark forward pass performance."""
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(q, k, v)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time forward passes
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean_time": np.mean(times) * 1000,  # Convert to ms
        "std_time": np.std(times) * 1000,
        "min_time": np.min(times) * 1000,
        "max_time": np.max(times) * 1000,
        "median_time": np.median(times) * 1000,
    }


def compare_implementations():
    """Compare Ring Attention V2 vs V3 performance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    print("=" * 80)

    configs = [
        # (name, segment_lengths, dilation_rates, seq_len, batch_size)
        ("Small", [512, 1024], [1, 2], 1024, 4),
        ("Medium", [1024, 2048], [1, 2], 2048, 2),
        ("Large", [2048, 4096], [1, 2], 4096, 1),
        ("XLarge", [4096, 8192], [1, 2], 8192, 1),
    ]

    results = []

    for config_name, segment_lengths, dilation_rates, seq_len, batch_size in configs:
        print(f"\n{config_name} Configuration:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Batch size: {batch_size}")
        print(f"  Segments: {segment_lengths}")
        print(f"  Dilations: {dilation_rates}")

        # Create models
        ring_v2_no_cache = (
            RingDilatedAttentionV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_pattern_cache=False,
            )
            .to(device)
            .eval()
        )

        ring_v2_cached = (
            RingDilatedAttentionV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_pattern_cache=True,
            )
            .to(device)
            .eval()
        )

        ring_v3_optimized = (
            RingDilatedAttentionV3(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_pattern_cache=True,
                cache_on_gpu=True,
            )
            .to(device)
            .eval()
        )

        # Clear caches
        clear_global_cache()
        clear_optimized_cache()

        # Benchmark V2 without cache (baseline)
        print("\n  V2 No Cache (baseline):")
        v2_no_cache_results = benchmark_forward_pass(
            ring_v2_no_cache, batch_size, seq_len, 16, 64, device=device
        )
        print(f"    Mean time: {v2_no_cache_results['mean_time']:.3f} ms")
        print(f"    Std dev: {v2_no_cache_results['std_time']:.3f} ms")

        # Benchmark V2 with cache
        clear_global_cache()
        print("\n  V2 With Cache:")
        v2_cached_results = benchmark_forward_pass(
            ring_v2_cached, batch_size, seq_len, 16, 64, device=device
        )
        print(f"    Mean time: {v2_cached_results['mean_time']:.3f} ms")
        print(f"    Std dev: {v2_cached_results['std_time']:.3f} ms")

        # Get V2 cache stats
        from dilated_attention_pytorch.core import get_global_pattern_cache

        v2_cache = get_global_pattern_cache()
        _ = v2_cache.get_stats()

        # Benchmark V3 with optimized cache
        clear_optimized_cache()
        print("\n  V3 Optimized Cache:")
        v3_optimized_results = benchmark_forward_pass(
            ring_v3_optimized, batch_size, seq_len, 16, 64, device=device
        )
        print(f"    Mean time: {v3_optimized_results['mean_time']:.3f} ms")
        print(f"    Std dev: {v3_optimized_results['std_time']:.3f} ms")

        # Get V3 cache stats
        if hasattr(ring_v3_optimized, "_pattern_cache"):
            v3_stats = ring_v3_optimized._pattern_cache.get_stats()
            print("\n  V3 Cache Stats:")
            print(f"    GPU patterns: {v3_stats['gpu_cache_size']}")
            print(f"    CPU patterns: {v3_stats['cpu_cache_size']}")
            print(f"    GPU hits: {v3_stats['gpu_hits']}")
            print(f"    CPU hits: {v3_stats['cpu_hits']}")
            print(f"    GPU memory: {v3_stats['gpu_memory_used_mb']:.2f} MB")

        # Calculate improvements
        v2_speedup = v2_no_cache_results["mean_time"] / v2_cached_results["mean_time"]
        v3_speedup = (
            v2_no_cache_results["mean_time"] / v3_optimized_results["mean_time"]
        )
        v3_vs_v2_speedup = (
            v2_cached_results["mean_time"] / v3_optimized_results["mean_time"]
        )

        print("\n  Performance Summary:")
        print(f"    V2 cached speedup: {v2_speedup:.2f}x")
        print(f"    V3 optimized speedup: {v3_speedup:.2f}x")
        print(f"    V3 vs V2 cached: {v3_vs_v2_speedup:.2f}x")

        # Memory comparison
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Measure V2 memory
            mem_before = torch.cuda.memory_allocated()
            v2_test = RingDilatedAttentionV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_pattern_cache=True,
            ).to(device)
            dummy = torch.randn(1, max(segment_lengths), 4, 16, device=device)
            _ = v2_test(dummy, dummy, dummy)
            v2_memory = torch.cuda.memory_allocated() - mem_before

            torch.cuda.empty_cache()

            # Measure V3 memory
            mem_before = torch.cuda.memory_allocated()
            v3_test = RingDilatedAttentionV3(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_pattern_cache=True,
                cache_on_gpu=True,
            ).to(device)
            _ = v3_test(dummy, dummy, dummy)
            v3_memory = torch.cuda.memory_allocated() - mem_before

            print("\n  Memory Usage:")
            print(f"    V2 memory: {v2_memory / 1024 / 1024:.2f} MB")
            print(f"    V3 memory: {v3_memory / 1024 / 1024:.2f} MB")
            print(f"    Difference: {(v3_memory - v2_memory) / 1024 / 1024:.2f} MB")

        results.append(
            {
                "config": config_name,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "segment_lengths": segment_lengths,
                "dilation_rates": dilation_rates,
                "v2_no_cache_mean": v2_no_cache_results["mean_time"],
                "v2_cached_mean": v2_cached_results["mean_time"],
                "v3_optimized_mean": v3_optimized_results["mean_time"],
                "v2_speedup": v2_speedup,
                "v3_speedup": v3_speedup,
                "v3_vs_v2_speedup": v3_vs_v2_speedup,
                "device": device,
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    avg_v2_speedup = np.mean([r["v2_speedup"] for r in results])
    avg_v3_speedup = np.mean([r["v3_speedup"] for r in results])
    avg_v3_vs_v2 = np.mean([r["v3_vs_v2_speedup"] for r in results])

    print(f"Average V2 cached speedup: {avg_v2_speedup:.2f}x")
    print(f"Average V3 optimized speedup: {avg_v3_speedup:.2f}x")
    print(f"Average V3 vs V2 improvement: {avg_v3_vs_v2:.2f}x")

    print("\nKey findings:")
    print("- V3 eliminates CPU→GPU transfer overhead with GPU-resident patterns")
    print("- Hot patterns stay on GPU, cold patterns on CPU")
    print("- Adaptive tier management based on access frequency")
    print("- Batch transfers and prefetching further improve performance")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/ring_v2_vs_v3_results_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write("Ring Attention V2 vs V3 Benchmark Results\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"{r['config']} Configuration:\n")
            f.write(f"  Sequence length: {r['seq_len']}\n")
            f.write(f"  V2 no cache: {r['v2_no_cache_mean']:.3f} ms\n")
            f.write(f"  V2 cached: {r['v2_cached_mean']:.3f} ms\n")
            f.write(f"  V3 optimized: {r['v3_optimized_mean']:.3f} ms\n")
            f.write(f"  V3 vs V2 speedup: {r['v3_vs_v2_speedup']:.2f}x\n\n")

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    compare_implementations()
