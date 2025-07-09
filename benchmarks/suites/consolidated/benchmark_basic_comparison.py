#!/usr/bin/env python3
"""
Basic performance comparison benchmark suite.

This consolidates functionality from:
- benchmark_simple_comparison.py
- benchmark_quick_performance.py
- benchmark_original_dilated_quick.py
- benchmark_basic_comparison.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.core.config import BenchmarkPreset  # noqa: E402
from benchmarks.core.unified_runner import UnifiedBenchmarkRunner  # noqa: E402


def main():
    """Run basic comparison benchmarks."""
    parser = argparse.ArgumentParser(description="Basic performance comparison")
    parser.add_argument(
        "--preset",
        type=str,
        default="standard",
        choices=["quick", "standard", "comprehensive"],
        help="Benchmark preset to use",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["standard", "improved"],
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--seq-len", type=int, default=None, help="Override sequence length"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="table",
        choices=["table", "csv", "json"],
        help="Output format",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to file"
    )

    args = parser.parse_args()

    # Get preset config
    config = BenchmarkPreset.get_preset(args.preset)

    # Override with command line args
    config.implementations = args.implementations
    config.output_format = args.output_format
    config.save_results = args.save_results

    if args.batch_size is not None:
        config.batch_sizes = [args.batch_size]

    if args.seq_len is not None:
        config.sequence_lengths = [args.seq_len]

    # Adjust segment lengths based on sequence lengths
    if args.preset == "quick":
        config.segment_lengths = [[512, 1024]]
        config.dilation_rates = [[1, 2]]

    # Print configuration
    print("=" * 80)
    print("Basic Performance Comparison")
    print("=" * 80)
    print(f"Preset: {args.preset}")
    print(f"Implementations: {', '.join(config.implementations)}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Sequence lengths: {config.sequence_lengths}")
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    print("=" * 80)

    # Run benchmarks
    runner = UnifiedBenchmarkRunner(config)
    results = runner.run_benchmarks()

    # Print summary statistics
    if results:
        print("\nSummary Statistics:")
        print("-" * 80)

        # Group by implementation
        impl_results = {}
        for r in results:
            if r.error:
                continue
            if r.implementation not in impl_results:
                impl_results[r.implementation] = []
            impl_results[r.implementation].append(r.forward_time_ms)

        # Print average performance
        for impl, times in impl_results.items():
            avg_time = sum(times) / len(times)
            print(f"{impl:20s}: {avg_time:8.2f} ms (avg)")

        # Find best implementation
        if impl_results:
            best_impl = min(impl_results.items(), key=lambda x: sum(x[1]) / len(x[1]))
            print(f"\nBest implementation: {best_impl[0]}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
