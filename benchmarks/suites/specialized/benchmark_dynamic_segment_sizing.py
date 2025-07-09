#!/usr/bin/env python3
"""
Benchmark comparing dynamic vs fixed segment sizing in dilated attention.

This benchmark tests:
1. Performance difference between dynamic and fixed configurations
2. Memory efficiency under different conditions
3. Adaptation to varying sequence lengths
4. Impact on different batch sizes
"""

import time
import gc
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime

from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
    DynamicDilatedAttention,
    SegmentSelectionConfig,
)


def get_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def benchmark_attention(
    attention_module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
) -> Dict[str, float]:
    """Benchmark an attention module."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Move module to device
    attention_module = attention_module.to(device)

    # Warmup
    for _ in range(warmup_iterations):
        _ = attention_module(q, k, v)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure memory before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    memory_before = get_memory_usage()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = attention_module(q, k, v)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    # Measure peak memory
    peak_memory = get_memory_usage()
    memory_used = peak_memory - memory_before

    # Clean up
    del output

    # Calculate statistics
    times = np.array(times)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "memory_gb": memory_used,
        "throughput": (batch_size * seq_len) / np.mean(times) / 1e6,  # M tokens/sec
    }


def benchmark_configurations():
    """Compare different configurations."""
    results = []

    # Test configurations
    test_configs = [
        # (batch_size, seq_len, description)
        (2, 1024, "Short sequence"),
        (2, 4096, "Medium sequence"),
        (2, 16384, "Long sequence"),
        (4, 8192, "Large batch"),
        (1, 32768, "Very long sequence"),
    ]

    # Fixed configurations to test
    fixed_configs = [
        ([1024, 2048, 4096], [1, 2, 4], "Conservative"),
        ([2048, 4096, 8192], [1, 2, 4], "Standard"),
        ([4096, 8192, 16384], [1, 2, 4], "Aggressive"),
    ]

    num_heads = 8
    head_dim = 64

    print("Starting benchmarks...")
    print("=" * 80)

    for batch_size, seq_len, desc in test_configs:
        print(f"\nTesting: {desc} (batch={batch_size}, seq_len={seq_len})")
        print("-" * 60)

        # Skip if sequence length incompatible
        valid_configs = []
        for segments, rates, name in fixed_configs:
            if seq_len % max(segments) == 0:
                valid_configs.append((segments, rates, name))

        if not valid_configs:
            print(f"Skipping - no valid fixed configurations for seq_len={seq_len}")
            continue

        # Test dynamic attention
        print("Dynamic attention...")
        try:
            dynamic_attention = DynamicDilatedAttention(
                min_segment_size=512,
                max_segment_size=min(32768, seq_len),
                improved=True,
            )

            dynamic_results = benchmark_attention(
                dynamic_attention, batch_size, seq_len, num_heads, head_dim
            )

            # Get selected configuration
            segments, rates = dynamic_attention.get_current_configuration()

            result = {
                "test_case": desc,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "config_type": "Dynamic",
                "segments": segments,
                "rates": rates,
                **dynamic_results,
            }
            results.append(result)

            print(f"  Segments: {segments}")
            print(f"  Time: {dynamic_results['mean_time'] * 1000:.2f}ms")
            print(f"  Memory: {dynamic_results['memory_gb']:.2f}GB")
            print(f"  Throughput: {dynamic_results['throughput']:.2f} M tokens/sec")

        except Exception as e:
            print(f"  Failed: {e}")

        # Test fixed configurations
        for segments, rates, name in valid_configs:
            print(f"\nFixed attention ({name})...")
            try:
                fixed_attention = ImprovedDilatedAttention(
                    segment_lengths=segments, dilation_rates=rates
                )

                fixed_results = benchmark_attention(
                    fixed_attention, batch_size, seq_len, num_heads, head_dim
                )

                result = {
                    "test_case": desc,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "config_type": f"Fixed-{name}",
                    "segments": segments,
                    "rates": rates,
                    **fixed_results,
                }
                results.append(result)

                print(f"  Segments: {segments}")
                print(f"  Time: {fixed_results['mean_time'] * 1000:.2f}ms")
                print(f"  Memory: {fixed_results['memory_gb']:.2f}GB")
                print(f"  Throughput: {fixed_results['throughput']:.2f} M tokens/sec")

            except Exception as e:
                print(f"  Failed: {e}")

        # Clean up between tests
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def analyze_results(results: List[Dict]):
    """Analyze and summarize benchmark results."""
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group by test case
    test_cases = {}
    for r in results:
        case = r["test_case"]
        if case not in test_cases:
            test_cases[case] = []
        test_cases[case].append(r)

    for case, case_results in test_cases.items():
        print(f"\n{case}:")

        # Find dynamic result
        dynamic = next((r for r in case_results if r["config_type"] == "Dynamic"), None)
        if not dynamic:
            continue

        print(f"  Dynamic config: segments={dynamic['segments']}")

        # Compare with fixed configs
        for r in case_results:
            if r["config_type"] != "Dynamic":
                speedup = dynamic["mean_time"] / r["mean_time"]
                memory_ratio = (
                    dynamic["memory_gb"] / r["memory_gb"] if r["memory_gb"] > 0 else 1.0
                )

                print(f"  vs {r['config_type']}:")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Memory ratio: {memory_ratio:.2f}x")
                print(f"    Fixed segments: {r['segments']}")


def test_memory_adaptation():
    """Test how dynamic sizing adapts to memory pressure."""
    print("\n" + "=" * 80)
    print("MEMORY ADAPTATION TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("Skipping - requires CUDA")
        return

    # Create memory pressure by allocating tensors
    seq_len = 8192
    num_heads = 8
    head_dim = 64

    # Test with increasing batch sizes
    batch_sizes = [1, 2, 4, 8, 16]

    # Conservative memory config
    config = SegmentSelectionConfig(memory_safety_factor=0.5, min_free_memory_gb=0.5)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        try:
            attention = DynamicDilatedAttention(
                selector_config=config, max_segment_size=16384
            )

            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")

            # Force recomputation
            _ = attention(q, k, v, force_segment_update=True)

            segments, _ = attention.get_current_configuration()
            print(f"  Selected segments: {segments}")
            print(f"  Max segment: {max(segments)}")

            # Clean up
            del q, k, v, attention
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  Out of memory - as expected")
                torch.cuda.empty_cache()
                break
            else:
                raise


def main():
    """Run all benchmarks."""
    print("Dynamic Segment Sizing Benchmark")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("Running on CPU")

    # Run main benchmarks
    results = benchmark_configurations()

    # Analyze results
    analyze_results(results)

    # Test memory adaptation
    test_memory_adaptation()

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"dynamic_segment_benchmark_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write("Dynamic Segment Sizing Benchmark Results\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"{r['test_case']} - {r['config_type']}:\n")
            f.write(f"  Segments: {r['segments']}\n")
            f.write(
                f"  Time: {r['mean_time'] * 1000:.2f}ms (±{r['std_time'] * 1000:.2f}ms)\n"
            )
            f.write(f"  Memory: {r['memory_gb']:.2f}GB\n")
            f.write(f"  Throughput: {r['throughput']:.2f} M tokens/sec\n\n")

    print(f"\nResults saved to: {filename}")
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
