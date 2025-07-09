#!/usr/bin/env python3
"""
Benchmark testing aggressive dynamic segment sizing configurations.

This script compares conservative vs aggressive dynamic segment selection
to see the performance and stability trade-offs.
"""

import time
import gc
from typing import Dict, List, Tuple
import torch
import numpy as np
from datetime import datetime

from dilated_attention_pytorch import (
    DynamicDilatedAttention,
    SegmentSelectionConfig,
    ImprovedDilatedAttention,
)


def get_memory_usage() -> Tuple[float, float]:
    """Get current GPU memory usage and available memory in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        free, total = torch.cuda.mem_get_info()
        available = free / 1e9
        return allocated, available
    return 0.0, 0.0


def create_conservative_config() -> SegmentSelectionConfig:
    """Create a conservative configuration (default)."""
    return SegmentSelectionConfig(
        memory_safety_factor=0.8,  # Use only 80% of available memory
        min_free_memory_gb=0.5,  # Keep 500MB free
        base_segment_size=2048,
        max_segment_size=65536,
        prefer_power_of_2=True,
        max_num_segments=5,
    )


def create_aggressive_config() -> SegmentSelectionConfig:
    """Create an aggressive configuration that pushes limits."""
    return SegmentSelectionConfig(
        memory_safety_factor=0.95,  # Use 95% of available memory (very aggressive!)
        min_free_memory_gb=0.1,  # Keep only 100MB free (risky!)
        base_segment_size=4096,  # Start with larger base segments
        max_segment_size=131072,  # Allow very large segments (128K)
        prefer_power_of_2=True,
        max_num_segments=8,  # Allow more segments for flexibility
        geometric_ratio=2.5,  # Faster growth rate
    )


def create_balanced_aggressive_config() -> SegmentSelectionConfig:
    """Create a balanced aggressive configuration."""
    return SegmentSelectionConfig(
        memory_safety_factor=0.9,  # Use 90% of available memory
        min_free_memory_gb=0.25,  # Keep 250MB free
        base_segment_size=4096,  # Larger base segments
        max_segment_size=65536,  # Standard max
        prefer_power_of_2=True,
        max_num_segments=6,
        geometric_ratio=2.0,
    )


def benchmark_configuration(
    config_name: str,
    config: SegmentSelectionConfig,
    test_cases: List[Tuple[int, int, str]],
    num_iterations: int = 5,
) -> List[Dict]:
    """Benchmark a specific configuration."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"Testing {config_name} Configuration")
    print(f"{'=' * 60}")
    print(f"Memory safety factor: {config.memory_safety_factor}")
    print(f"Min free memory: {config.min_free_memory_gb}GB")
    print(f"Base segment size: {config.base_segment_size}")
    print(f"Max segment size: {config.max_segment_size}")

    for batch_size, seq_len, desc in test_cases:
        print(f"\n{desc} (batch={batch_size}, seq_len={seq_len}):")

        # Create attention module
        attention = DynamicDilatedAttention(selector_config=config, improved=True).to(
            device
        )

        # Check initial memory
        allocated_before, available_before = get_memory_usage()
        print(
            f"  Memory before: {allocated_before:.2f}GB allocated, {available_before:.2f}GB available"
        )

        try:
            # Create inputs
            q = torch.randn(batch_size, seq_len, 8, 64, device=device)
            k = torch.randn(batch_size, seq_len, 8, 64, device=device)
            v = torch.randn(batch_size, seq_len, 8, 64, device=device)

            # Force segment selection
            _ = attention(q, k, v, force_segment_update=True)

            # Get selected configuration
            segments, rates = attention.get_current_configuration()
            print(f"  Selected segments: {segments}")
            print(f"  Dilation rates: {rates}")
            print(f"  Max segment: {max(segments) if segments else 0}")

            # Benchmark performance
            times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = attention(q, k, v)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

            mean_time = np.mean(times)
            throughput = (batch_size * seq_len) / mean_time / 1e6

            # Check memory after
            allocated_after, available_after = get_memory_usage()
            peak_memory_used = allocated_after - allocated_before

            result = {
                "config": config_name,
                "test_case": desc,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "segments": segments,
                "rates": rates,
                "mean_time": mean_time,
                "throughput": throughput,
                "peak_memory_gb": peak_memory_used,
                "status": "success",
            }

            print(f"  Time: {mean_time * 1000:.2f}ms")
            print(f"  Throughput: {throughput:.2f} M tokens/sec")
            print(f"  Peak memory used: {peak_memory_used:.2f}GB")
            print("  ✓ Success")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result = {
                    "config": config_name,
                    "test_case": desc,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "status": "oom",
                    "error": str(e),
                }
                print("  ✗ Out of Memory!")
                # Clear memory
                torch.cuda.empty_cache()
            else:
                result = {
                    "config": config_name,
                    "test_case": desc,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "status": "error",
                    "error": str(e),
                }
                print(f"  ✗ Error: {e}")

        except Exception as e:
            result = {
                "config": config_name,
                "test_case": desc,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ Error: {e}")

        results.append(result)

        # Clean up
        del q, k, v
        if "output" in locals():
            del output
        del attention
        gc.collect()
        torch.cuda.empty_cache()

        # Brief pause to let memory settle
        time.sleep(0.5)

    return results


def compare_fixed_segments(
    test_cases: List[Tuple[int, int, str]], num_iterations: int = 5
) -> List[Dict]:
    """Compare with fixed segment configurations for reference."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("Testing Fixed Segment Configuration (Reference)")
    print(f"{'=' * 60}")

    # Aggressive fixed configuration
    fixed_segments = [4096, 8192, 16384, 32768]
    fixed_rates = [1, 2, 4, 8]

    print(f"Fixed segments: {fixed_segments}")
    print(f"Fixed rates: {fixed_rates}")

    for batch_size, seq_len, desc in test_cases:
        # Skip if incompatible
        if seq_len % max(fixed_segments) != 0:
            continue

        print(f"\n{desc} (batch={batch_size}, seq_len={seq_len}):")

        try:
            attention = ImprovedDilatedAttention(
                segment_lengths=fixed_segments, dilation_rates=fixed_rates
            ).to(device)

            # Create inputs
            q = torch.randn(batch_size, seq_len, 8, 64, device=device)
            k = torch.randn(batch_size, seq_len, 8, 64, device=device)
            v = torch.randn(batch_size, seq_len, 8, 64, device=device)

            # Benchmark
            times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = attention(q, k, v)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

            mean_time = np.mean(times)
            throughput = (batch_size * seq_len) / mean_time / 1e6

            result = {
                "config": "Fixed-Aggressive",
                "test_case": desc,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "segments": fixed_segments,
                "rates": fixed_rates,
                "mean_time": mean_time,
                "throughput": throughput,
                "status": "success",
            }

            print(f"  Time: {mean_time * 1000:.2f}ms")
            print(f"  Throughput: {throughput:.2f} M tokens/sec")
            print("  ✓ Success")

        except Exception as e:
            result = {
                "config": "Fixed-Aggressive",
                "test_case": desc,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ Error: {e}")

        results.append(result)

        # Clean up
        del q, k, v, attention
        if "output" in locals():
            del output
        gc.collect()
        torch.cuda.empty_cache()

    return results


def analyze_results(all_results: List[Dict]):
    """Analyze and compare results."""
    print(f"\n{'=' * 60}")
    print("ANALYSIS")
    print(f"{'=' * 60}")

    # Group by test case
    test_cases = {}
    for r in all_results:
        case = r["test_case"]
        if case not in test_cases:
            test_cases[case] = {}
        test_cases[case][r["config"]] = r

    for case, configs in test_cases.items():
        print(f"\n{case}:")

        # Find successful runs
        successful = {
            name: r for name, r in configs.items() if r["status"] == "success"
        }

        if not successful:
            print("  No successful runs!")
            continue

        # Compare configurations
        for config_name, result in successful.items():
            print(f"\n  {config_name}:")
            print(f"    Segments: {result.get('segments', 'N/A')}")
            print(f"    Time: {result.get('mean_time', 0) * 1000:.2f}ms")
            print(f"    Throughput: {result.get('throughput', 0):.2f} M tokens/sec")
            if "peak_memory_gb" in result:
                print(f"    Peak memory: {result['peak_memory_gb']:.2f}GB")

        # Find fastest
        if len(successful) > 1:
            fastest = min(
                successful.items(), key=lambda x: x[1].get("mean_time", float("inf"))
            )
            print(f"\n  Fastest: {fastest[0]}")

            # Calculate speedups
            fastest_time = fastest[1]["mean_time"]
            for name, r in successful.items():
                if name != fastest[0]:
                    speedup = r["mean_time"] / fastest_time
                    print(f"  {fastest[0]} is {speedup:.2f}x faster than {name}")


def main():
    """Run aggressive configuration benchmarks."""
    print("Aggressive Dynamic Segment Sizing Benchmark")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
        _, available = get_memory_usage()
        print(f"Available Memory: {available:.1f} GB")
    else:
        print("Running on CPU (not recommended)")

    # Test cases - push the limits!
    test_cases = [
        (2, 4096, "Small sequence"),
        (4, 8192, "Medium sequence"),
        (2, 16384, "Long sequence"),
        (1, 32768, "Very long sequence"),
        (1, 65536, "Extreme sequence"),
        (8, 8192, "Large batch"),
        (16, 4096, "Very large batch"),
    ]

    all_results = []

    # Test conservative configuration
    conservative_config = create_conservative_config()
    conservative_results = benchmark_configuration(
        "Conservative", conservative_config, test_cases
    )
    all_results.extend(conservative_results)

    # Test balanced aggressive configuration
    balanced_config = create_balanced_aggressive_config()
    balanced_results = benchmark_configuration(
        "Balanced-Aggressive", balanced_config, test_cases
    )
    all_results.extend(balanced_results)

    # Test very aggressive configuration
    aggressive_config = create_aggressive_config()
    aggressive_results = benchmark_configuration(
        "Very-Aggressive", aggressive_config, test_cases
    )
    all_results.extend(aggressive_results)

    # Test fixed configuration for comparison
    fixed_results = compare_fixed_segments(test_cases)
    all_results.extend(fixed_results)

    # Analyze results
    analyze_results(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"aggressive_dynamic_benchmark_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write("Aggressive Dynamic Segment Sizing Benchmark Results\n")
        f.write("=" * 80 + "\n\n")

        for r in all_results:
            f.write(f"{r['config']} - {r['test_case']}:\n")
            if r["status"] == "success":
                f.write(f"  Segments: {r.get('segments', 'N/A')}\n")
                f.write(f"  Time: {r.get('mean_time', 0) * 1000:.2f}ms\n")
                f.write(f"  Throughput: {r.get('throughput', 0):.2f} M tokens/sec\n")
                if "peak_memory_gb" in r:
                    f.write(f"  Peak memory: {r['peak_memory_gb']:.2f}GB\n")
            else:
                f.write(f"  Status: {r['status']}\n")
                if "error" in r:
                    f.write(f"  Error: {r['error']}\n")
            f.write("\n")

    print(f"\nResults saved to: {filename}")
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
