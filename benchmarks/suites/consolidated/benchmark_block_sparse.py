#!/usr/bin/env python3
"""
Block sparse attention benchmark suite.

This consolidates functionality from:
- benchmark_block_sparse_simple.py
- benchmark_block_sparse_comprehensive.py
- benchmark_block_sparse_focused.py
- benchmark_block_sparse_variants.py
- benchmark_all_block_sparse.py
- test_block_sparse_basic.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.core.config import BenchmarkConfig  # noqa: E402
from benchmarks.core.unified_runner import UnifiedBenchmarkRunner  # noqa: E402


class BlockSparseBenchmark:
    """Comprehensive block sparse attention benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize block sparse benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.runner = UnifiedBenchmarkRunner(config)

    def benchmark_sparsity_impact(self):
        """Benchmark impact of different sparsity ratios."""
        print("\nSparsity Impact Analysis")
        print("=" * 80)

        results_by_sparsity = {}

        for sparsity in self.config.sparsity_ratios:
            print(f"\nTesting sparsity={sparsity:.1%}")
            print("-" * 40)

            results = []
            for seq_len in self.config.sequence_lengths:
                # Find appropriate segment config
                segment_lengths = None
                dilation_rates = None

                for segs, dils in zip(
                    self.config.segment_lengths, self.config.dilation_rates
                ):
                    if seq_len % max(segs) == 0:
                        segment_lengths = segs
                        dilation_rates = dils
                        break

                if segment_lengths is None:
                    continue

                result = self.runner.benchmark_single_configuration(
                    implementation="block_sparse",
                    batch_size=self.config.batch_sizes[0],
                    seq_len=seq_len,
                    num_heads=self.config.num_heads[0],
                    embed_dim=self.config.embed_dims[0],
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparsity_ratio=sparsity,
                    block_size=self.config.block_sizes[0],
                    pattern_type="dilated_sparse",
                )

                if not result.error:
                    results.append(result)
                    print(
                        f"  L={seq_len:5d}: {result.forward_time_ms:6.2f}ms, "
                        f"{result.peak_memory_mb:6.1f}MB"
                    )

            results_by_sparsity[sparsity] = results

        return results_by_sparsity

    def benchmark_pattern_comparison(self):
        """Compare different sparse attention patterns."""
        print("\nPattern Comparison")
        print("=" * 80)

        results_by_pattern = {}

        # Fixed configuration for pattern comparison
        seq_len = self.config.sequence_lengths[0]
        batch_size = self.config.batch_sizes[0]
        embed_dim = self.config.embed_dims[0]
        num_heads = self.config.num_heads[0]

        # Find segment config
        segment_lengths = self.config.segment_lengths[0]
        dilation_rates = self.config.dilation_rates[0]

        for pattern in self.config.pattern_types:
            print(f"\nTesting pattern: {pattern}")
            print("-" * 40)

            results = []
            for sparsity in [0.1, 0.5, 0.9]:  # Test key sparsity levels
                result = self.runner.benchmark_single_configuration(
                    implementation="block_sparse",
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparsity_ratio=sparsity,
                    block_size=128,
                    pattern_type=pattern,
                )

                if not result.error:
                    results.append(result)
                    speedup = seq_len**2 / (seq_len**2 * (1 - sparsity))
                    print(
                        f"  Sparsity={sparsity:.1%}: {result.forward_time_ms:6.2f}ms "
                        f"(theoretical speedup: {speedup:.1f}x)"
                    )

            results_by_pattern[pattern] = results

        return results_by_pattern

    def benchmark_block_size_impact(self):
        """Benchmark impact of different block sizes."""
        print("\nBlock Size Impact Analysis")
        print("=" * 80)

        results_by_block_size = {}

        # Fixed configuration
        seq_len = self.config.sequence_lengths[0]
        batch_size = self.config.batch_sizes[0]
        embed_dim = self.config.embed_dims[0]
        num_heads = self.config.num_heads[0]
        sparsity = 0.9  # High sparsity to see block size impact

        segment_lengths = self.config.segment_lengths[0]
        dilation_rates = self.config.dilation_rates[0]

        for block_size in self.config.block_sizes:
            print(f"\nTesting block_size={block_size}")

            result = self.runner.benchmark_single_configuration(
                implementation="block_sparse",
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                embed_dim=embed_dim,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                sparsity_ratio=sparsity,
                block_size=block_size,
                pattern_type="dilated_sparse",
            )

            if not result.error:
                results_by_block_size[block_size] = result
                print(
                    f"  Time: {result.forward_time_ms:6.2f}ms, "
                    f"Memory: {result.peak_memory_mb:6.1f}MB"
                )

        return results_by_block_size

    def plot_results(self, results_dict, title, xlabel, ylabel="Time (ms)"):
        """Plot benchmark results.

        Args:
            results_dict: Dictionary of results
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if not results_dict:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Time plot
        x_values = list(results_dict.keys())
        y_times = []
        y_memory = []

        for key, results in results_dict.items():
            if isinstance(results, list):
                avg_time = np.mean([r.forward_time_ms for r in results if not r.error])
                avg_memory = np.mean([r.peak_memory_mb for r in results if not r.error])
            else:
                avg_time = results.forward_time_ms
                avg_memory = results.peak_memory_mb

            y_times.append(avg_time)
            y_memory.append(avg_memory)

        ax1.plot(x_values, y_times, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"{title} - Performance")
        ax1.grid(True, alpha=0.3)

        # Memory plot
        ax2.plot(x_values, y_memory, "ro-", linewidth=2, markersize=8)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Memory (MB)")
        ax2.set_title(f"{title} - Memory Usage")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"block_sparse_{title.lower().replace(' ', '_')}.png")
        plt.close()

    def run_comprehensive_benchmark(self):
        """Run comprehensive block sparse benchmarks."""
        print("=" * 80)
        print("Block Sparse Attention Comprehensive Benchmark")
        print("=" * 80)

        # 1. Sparsity impact
        sparsity_results = self.benchmark_sparsity_impact()
        if self.config.plot_results:
            self.plot_results(sparsity_results, "Sparsity Impact", "Sparsity Ratio")

        # 2. Pattern comparison
        pattern_results = self.benchmark_pattern_comparison()

        # 3. Block size impact
        block_results = self.benchmark_block_size_impact()
        if self.config.plot_results:
            self.plot_results(block_results, "Block Size Impact", "Block Size")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Best sparsity
        if sparsity_results:
            best_sparsity = min(
                sparsity_results.items(),
                key=lambda x: np.mean([r.forward_time_ms for r in x[1] if not r.error]),
            )
            print(f"Best sparsity ratio: {best_sparsity[0]:.1%}")

        # Best pattern
        if pattern_results:
            best_pattern = min(
                pattern_results.items(),
                key=lambda x: np.mean([r.forward_time_ms for r in x[1] if not r.error]),
            )
            print(f"Best pattern type: {best_pattern[0]}")

        # Best block size
        if block_results:
            best_block = min(block_results.items(), key=lambda x: x[1].forward_time_ms)
            print(f"Best block size: {best_block[0]}")


def main():
    """Run block sparse benchmarks."""
    parser = argparse.ArgumentParser(description="Block sparse attention benchmarks")
    parser.add_argument(
        "--sparsity-ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Sparsity ratios to test",
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256],
        help="Block sizes to test",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["local_window", "dilated_sparse", "global_local"],
        help="Pattern types to test",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[2048, 4096],
        help="Sequence lengths to test",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        implementations=["block_sparse"],
        sparsity_ratios=args.sparsity_ratios,
        block_sizes=args.block_sizes,
        pattern_types=args.patterns,
        sequence_lengths=args.seq_lengths,
        batch_sizes=[2],
        num_heads=[12],
        embed_dims=[768],
        segment_lengths=[[1024, 2048], [2048, 4096]],
        dilation_rates=[[1, 2], [1, 2]],
        plot_results=args.plot,
        warmup_iterations=2,
        benchmark_iterations=5,
    )

    # Run benchmarks
    benchmark = BlockSparseBenchmark(config)
    benchmark.run_comprehensive_benchmark()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
