#!/usr/bin/env python3
"""
Example of using the unified benchmark output system.

This demonstrates how to properly structure benchmark scripts for consistent output.
"""

import time
from pathlib import Path

import torch

# Import the unified output manager
from benchmarks.core import BenchmarkOutputManager


def run_example_benchmark():
    """Example benchmark function."""
    # Create output manager with benchmark type and parameters
    output_manager = BenchmarkOutputManager(
        benchmark_type="example-benchmark",
        parameters={
            "seq_len": 4096,
            "batch_size": 2,
            "num_heads": 8,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )

    # Run some fake benchmarks
    implementations = ["baseline", "optimized", "experimental"]
    results = {}

    for impl in implementations:
        # Simulate benchmark
        start = time.time()
        time.sleep(0.1)  # Simulate work
        elapsed = (time.time() - start) * 1000  # Convert to ms

        # Add result
        results[impl] = {
            "execution_time_ms": elapsed,
            "memory_mb": 100 + len(impl) * 10,  # Fake memory usage
            "throughput_tokens_per_sec": 1000 / elapsed,
        }

    # Add all results
    output_manager.add_result("implementations", results)

    # Add summary
    best_impl = min(results.items(), key=lambda x: x[1]["execution_time_ms"])
    output_manager.set_summary(
        {
            "best_implementation": best_impl[0],
            "best_time_ms": best_impl[1]["execution_time_ms"],
            "implementations_tested": len(implementations),
        }
    )

    # Save all outputs
    paths = output_manager.save_results()

    print("Benchmark results saved:")
    for output_type, path in paths.items():
        print(f"  {output_type}: {path}")

    # Example of saving a plot
    if torch.cuda.is_available():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        impl_names = list(results.keys())
        times = [results[impl]["execution_time_ms"] for impl in impl_names]

        ax.bar(impl_names, times)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Example Benchmark Results")

        # Save plot temporarily
        plot_path = Path("/tmp/example_plot.png")
        fig.savefig(plot_path)
        plt.close()

        # Save through output manager
        final_plot_path = output_manager.save_plot(plot_path)
        print(f"  plot: {final_plot_path}")


if __name__ == "__main__":
    print("Running example benchmark with unified output system...")
    run_example_benchmark()
