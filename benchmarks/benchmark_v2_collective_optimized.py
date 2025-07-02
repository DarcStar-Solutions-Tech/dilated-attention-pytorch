#!/usr/bin/env python3
"""
Benchmark V2 Collective after optimization and cleanup.
Compare with previous results to measure improvements.
"""

import json
import time
from datetime import datetime

import torch
import torch.cuda
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def benchmark_single_run(attention, q, k, v, is_causal, warmup=5, iterations=100):
    """Benchmark a single configuration."""
    # Warmup
    for _ in range(warmup):
        _ = attention(q, k, v, is_causal)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time iterations
    start = time.perf_counter()
    for _ in range(iterations):
        _ = attention(q, k, v, is_causal)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms

    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = attention(q, k, v, is_causal)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        peak_memory = 0

    return avg_time, peak_memory


def run_benchmarks():
    """Run comprehensive benchmarks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test configurations
    configs = [
        # (seq_len, batch_size, num_heads, head_dim, segment_lengths, dilation_rates, name)
        (512, 2, 8, 64, [64, 128], [1, 2], "Small"),
        (2048, 2, 8, 64, [256, 512], [1, 2], "Medium"),
        (8192, 2, 8, 64, [1024, 2048], [1, 2], "Large"),
        (16384, 1, 8, 64, [2048, 4096], [1, 2], "Very Large"),
        # Edge cases
        (128, 4, 16, 32, [32, 64], [1, 2], "Many Heads"),
        (4096, 2, 8, 64, [512, 1024], [2, 4], "High Dilation"),
    ]

    results = []

    print("=" * 80)
    print("V2 Collective Optimized Performance Benchmark")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 80)

    for seq_len, batch_size, num_heads, head_dim, seg_lens, dil_rates, name in configs:
        print(
            f"\nTesting {name}: seq_len={seq_len}, batch={batch_size}, heads={num_heads}"
        )

        # Create attention module
        attention = RingDilatedAttentionV2Collective(
            segment_lengths=seg_lens,
            dilation_rates=dil_rates,
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        if device.type == "cuda":
            q = q.half()
            k = k.half()
            v = v.half()

        # Benchmark causal and non-causal
        for is_causal in [False, True]:
            try:
                avg_time, peak_memory = benchmark_single_run(
                    attention, q, k, v, is_causal
                )

                result = {
                    "name": name,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "segment_lengths": seg_lens,
                    "dilation_rates": dil_rates,
                    "is_causal": is_causal,
                    "avg_time_ms": avg_time,
                    "peak_memory_mb": peak_memory,
                    "throughput_tokens_per_sec": (seq_len * batch_size)
                    / (avg_time / 1000),
                }

                results.append(result)

                print(
                    f"  {'Causal' if is_causal else 'Non-causal'}: "
                    f"{avg_time:.2f}ms, {peak_memory:.1f}MB, "
                    f"{result['throughput_tokens_per_sec']:.0f} tokens/sec"
                )

            except Exception as e:
                print(f"  {'Causal' if is_causal else 'Non-causal'}: Failed - {e}")

    return results


def compare_with_previous(current_results):
    """Compare with previous benchmark results if available."""
    # Try to load most recent previous results
    import glob
    import os

    csv_files = glob.glob("benchmarks/ring_attention_comparison_*.csv")
    if not csv_files:
        print("\nNo previous results to compare with.")
        return

    # Get most recent file
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"\nComparing with previous results from: {latest_file}")

    import pandas as pd

    try:
        prev_df = pd.read_csv(latest_file)
        # Filter for V2 Collective results
        prev_collective = prev_df[
            prev_df["implementation"].str.contains("Collective", na=False)
        ]

        if not prev_collective.empty:
            print("\nPerformance Comparison:")
            print("-" * 60)

            # Compare average times
            prev_avg_time = prev_collective["time_ms"].mean()
            current_avg_time = sum(r["avg_time_ms"] for r in current_results) / len(
                current_results
            )

            improvement = (prev_avg_time - current_avg_time) / prev_avg_time * 100

            print(f"Previous average time: {prev_avg_time:.2f}ms")
            print(f"Current average time: {current_avg_time:.2f}ms")
            print(f"Improvement: {improvement:.1f}%")

            # Compare memory usage if available
            if "memory_mb" in prev_collective.columns:
                prev_avg_memory = prev_collective["memory_mb"].mean()
                current_avg_memory = sum(
                    r["peak_memory_mb"] for r in current_results
                ) / len(current_results)

                memory_improvement = (
                    (prev_avg_memory - current_avg_memory) / prev_avg_memory * 100
                )

                print(f"\nPrevious average memory: {prev_avg_memory:.1f}MB")
                print(f"Current average memory: {current_avg_memory:.1f}MB")
                print(f"Memory improvement: {memory_improvement:.1f}%")

    except Exception as e:
        print(f"Could not load previous results: {e}")


def main():
    """Run benchmarks and save results."""
    results = run_benchmarks()

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_file = f"benchmarks/v2_collective_optimized_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Compare with previous results
    compare_with_previous(results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    avg_time = sum(r["avg_time_ms"] for r in results) / len(results)
    avg_memory = sum(r["peak_memory_mb"] for r in results) / len(results)
    avg_throughput = sum(r["throughput_tokens_per_sec"] for r in results) / len(results)

    print(f"Average time: {avg_time:.2f}ms")
    print(f"Average memory: {avg_memory:.1f}MB")
    print(f"Average throughput: {avg_throughput:.0f} tokens/sec")

    # Find best and worst cases
    fastest = min(results, key=lambda x: x["avg_time_ms"])
    slowest = max(results, key=lambda x: x["avg_time_ms"])

    print(f"\nFastest: {fastest['name']} - {fastest['avg_time_ms']:.2f}ms")
    print(f"Slowest: {slowest['name']} - {slowest['avg_time_ms']:.2f}ms")


if __name__ == "__main__":
    main()
