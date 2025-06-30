"""
Benchmark the impact of pattern caching and memory pooling optimizations.

This script measures:
1. Pattern caching effectiveness across sequence lengths
2. Memory pool impact on allocation/deallocation overhead
3. Combined optimization benefits
4. Cache hit rates and memory pool statistics
"""

import torch
import time
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import gc
import matplotlib.pyplot as plt

from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3
from dilated_attention_pytorch.core import (
    get_global_pattern_cache,
    reset_global_pattern_cache,
)
from dilated_attention_pytorch.core import (
    get_global_memory_pool,
    reset_global_memory_pool,
)

from benchmarks.benchmark_utils import BenchmarkOutputManager


@dataclass
class OptimizationBenchmarkResult:
    """Result for optimization impact benchmark."""

    implementation: str
    sequence_length: int
    segment_lengths: List[int]
    dilation_rates: List[int]
    batch_size: int
    num_heads: int
    head_dim: int

    # Timing results
    baseline_time_ms: float
    pattern_cache_time_ms: float
    memory_pool_time_ms: float
    combined_time_ms: float

    # Memory results
    baseline_memory_mb: float
    pattern_cache_memory_mb: float
    memory_pool_memory_mb: float
    combined_memory_mb: float

    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Memory pool statistics
    pool_allocations: int = 0
    pool_deallocations: int = 0
    pool_reuse_rate: float = 0.0

    # Speedup metrics
    pattern_cache_speedup: float = field(init=False)
    memory_pool_speedup: float = field(init=False)
    combined_speedup: float = field(init=False)

    # Memory reduction metrics
    pattern_cache_memory_reduction: float = field(init=False)
    memory_pool_memory_reduction: float = field(init=False)
    combined_memory_reduction: float = field(init=False)

    def __post_init__(self):
        """Calculate derived metrics."""
        # Speedup calculations
        self.pattern_cache_speedup = (
            self.baseline_time_ms / self.pattern_cache_time_ms
            if self.pattern_cache_time_ms > 0
            else 0
        )
        self.memory_pool_speedup = (
            self.baseline_time_ms / self.memory_pool_time_ms
            if self.memory_pool_time_ms > 0
            else 0
        )
        self.combined_speedup = (
            self.baseline_time_ms / self.combined_time_ms
            if self.combined_time_ms > 0
            else 0
        )

        # Memory reduction calculations
        if self.baseline_memory_mb > 0:
            self.pattern_cache_memory_reduction = (
                self.baseline_memory_mb - self.pattern_cache_memory_mb
            ) / self.baseline_memory_mb
            self.memory_pool_memory_reduction = (
                self.baseline_memory_mb - self.memory_pool_memory_mb
            ) / self.baseline_memory_mb
            self.combined_memory_reduction = (
                self.baseline_memory_mb - self.combined_memory_mb
            ) / self.baseline_memory_mb
        else:
            self.pattern_cache_memory_reduction = 0
            self.memory_pool_memory_reduction = 0
            self.combined_memory_reduction = 0


class OptimizationBenchmarker:
    """Benchmark optimization impact."""

    def __init__(self, device: str = "cuda", output_dir: str = "benchmark_results"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_manager = BenchmarkOutputManager(base_dir=output_dir)

    def get_segment_config(self, seq_len: int) -> Tuple[List[int], List[int]]:
        """Get appropriate segment configuration."""
        if seq_len <= 8192:
            return [1024, 2048, 4096], [1, 2, 4]
        elif seq_len <= 32768:
            return [2048, 4096, 8192], [1, 2, 4]
        elif seq_len <= 131072:
            return [4096, 16384, 65536], [1, 2, 4]
        else:
            return [8192, 32768, 131072], [1, 4, 16]

    def create_model(
        self,
        implementation: str,
        segment_lengths: List[int],
        dilation_rates: List[int],
        num_heads: int,
        head_dim: int,
        enable_pattern_cache: bool = False,
        enable_memory_pool: bool = False,
    ) -> torch.nn.Module:
        """Create model with specified optimizations."""
        kwargs = {
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
        }

        if implementation == "dilated":
            kwargs["enable_memory_pool"] = enable_memory_pool
            return DilatedAttention(**kwargs).to(self.device)
        elif implementation == "improved":
            kwargs["enable_memory_pool"] = enable_memory_pool
            return ImprovedDilatedAttention(**kwargs).to(self.device)
        elif implementation == "ring_v2":
            kwargs["enable_memory_pool"] = enable_memory_pool
            kwargs["use_pattern_cache"] = enable_pattern_cache
            return RingDilatedAttentionV2(**kwargs).to(self.device)
        elif implementation == "ring_v3":
            kwargs["enable_memory_pool"] = enable_memory_pool
            kwargs["use_pattern_cache"] = enable_pattern_cache
            return RingDilatedAttentionV3(**kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def benchmark_configuration(
        self,
        implementation: str,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        enable_pattern_cache: bool,
        enable_memory_pool: bool,
        warmup_steps: int = 3,
        benchmark_steps: int = 10,
    ) -> Tuple[float, float, Dict]:
        """Benchmark a single configuration."""
        # Get configuration
        segment_lengths, dilation_rates = self.get_segment_config(seq_len)

        # Ensure divisibility
        seq_len = (seq_len // segment_lengths[-1]) * segment_lengths[-1]

        # Reset caches and pools
        reset_global_pattern_cache()
        reset_global_memory_pool()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        try:
            # Create model
            model = self.create_model(
                implementation,
                segment_lengths,
                dilation_rates,
                num_heads,
                head_dim,
                enable_pattern_cache,
                enable_memory_pool,
            )

            # Create input tensors
            shape = (batch_size, seq_len, num_heads, head_dim)
            q = torch.randn(shape, device=self.device, dtype=torch.float16)
            k = torch.randn(shape, device=self.device, dtype=torch.float16)
            v = torch.randn(shape, device=self.device, dtype=torch.float16)

            # Get initial stats
            if enable_pattern_cache and implementation in ["ring_v2", "ring_v3"]:
                pattern_cache = get_global_pattern_cache()
                initial_cache_stats = pattern_cache.get_stats()

            if enable_memory_pool:
                memory_pool = get_global_memory_pool()
                initial_pool_stats = memory_pool.get_stats()

            # Warmup
            for _ in range(warmup_steps):
                _ = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated() / 1024 / 1024

            # Benchmark
            start_time = time.time()
            for _ in range(benchmark_steps):
                _ = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            # Calculate metrics
            time_ms = (end_time - start_time) * 1000 / benchmark_steps

            if self.device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_mb = peak_memory - start_memory
            else:
                memory_mb = 0.0

            # Collect statistics
            stats = {}

            if enable_pattern_cache and implementation in ["ring_v2", "ring_v3"]:
                final_cache_stats = pattern_cache.get_stats()
                stats["cache_hits"] = (
                    final_cache_stats["hits"] - initial_cache_stats["hits"]
                )
                stats["cache_misses"] = (
                    final_cache_stats["misses"] - initial_cache_stats["misses"]
                )
                total_accesses = stats["cache_hits"] + stats["cache_misses"]
                stats["cache_hit_rate"] = (
                    stats["cache_hits"] / total_accesses if total_accesses > 0 else 0.0
                )

            if enable_memory_pool:
                final_pool_stats = memory_pool.get_stats()
                stats["pool_allocations"] = (
                    final_pool_stats["allocation_count"]
                    - initial_pool_stats["allocation_count"]
                )
                stats["pool_deallocations"] = (
                    final_pool_stats["deallocation_count"]
                    - initial_pool_stats["deallocation_count"]
                )
                stats["pool_reuse_rate"] = final_pool_stats.get("reuse_rate", 0.0)

            return time_ms, memory_mb, stats

        finally:
            # Cleanup
            if "model" in locals():
                del model
            if "q" in locals():
                del q, k, v
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def benchmark_optimization_impact(
        self,
        implementation: str,
        sequence_lengths: List[int],
        num_heads: int = 8,
        head_dim: int = 64,
        batch_size: int = 2,
    ) -> List[OptimizationBenchmarkResult]:
        """Benchmark optimization impact across sequence lengths."""
        results = []

        for seq_len in sequence_lengths:
            print(f"\nBenchmarking {implementation} at {seq_len:,} tokens...")

            # Get segment configuration
            segment_lengths, dilation_rates = self.get_segment_config(seq_len)

            # Baseline (no optimizations)
            print("  Baseline...", end=" ", flush=True)
            baseline_time, baseline_memory, _ = self.benchmark_configuration(
                implementation,
                seq_len,
                num_heads,
                head_dim,
                batch_size,
                enable_pattern_cache=False,
                enable_memory_pool=False,
            )
            print(f"✓ {baseline_time:.1f}ms, {baseline_memory:.1f}MB")

            # Pattern cache only
            if implementation in ["ring_v2", "ring_v3"]:
                print("  Pattern cache...", end=" ", flush=True)
                cache_time, cache_memory, cache_stats = self.benchmark_configuration(
                    implementation,
                    seq_len,
                    num_heads,
                    head_dim,
                    batch_size,
                    enable_pattern_cache=True,
                    enable_memory_pool=False,
                )
                print(
                    f"✓ {cache_time:.1f}ms, {cache_memory:.1f}MB, hit rate: {cache_stats.get('cache_hit_rate', 0):.1%}"
                )
            else:
                cache_time = baseline_time
                cache_memory = baseline_memory
                cache_stats = {}

            # Memory pool only
            print("  Memory pool...", end=" ", flush=True)
            pool_time, pool_memory, pool_stats = self.benchmark_configuration(
                implementation,
                seq_len,
                num_heads,
                head_dim,
                batch_size,
                enable_pattern_cache=False,
                enable_memory_pool=True,
            )
            print(f"✓ {pool_time:.1f}ms, {pool_memory:.1f}MB")

            # Combined optimizations
            if implementation in ["ring_v2", "ring_v3"]:
                print("  Combined...", end=" ", flush=True)
                combined_time, combined_memory, combined_stats = (
                    self.benchmark_configuration(
                        implementation,
                        seq_len,
                        num_heads,
                        head_dim,
                        batch_size,
                        enable_pattern_cache=True,
                        enable_memory_pool=True,
                    )
                )
                print(f"✓ {combined_time:.1f}ms, {combined_memory:.1f}MB")
            else:
                combined_time = pool_time
                combined_memory = pool_memory
                _ = pool_stats

            # Create result
            result = OptimizationBenchmarkResult(
                implementation=implementation,
                sequence_length=seq_len,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                baseline_time_ms=baseline_time,
                pattern_cache_time_ms=cache_time,
                memory_pool_time_ms=pool_time,
                combined_time_ms=combined_time,
                baseline_memory_mb=baseline_memory,
                pattern_cache_memory_mb=cache_memory,
                memory_pool_memory_mb=pool_memory,
                combined_memory_mb=combined_memory,
                cache_hits=cache_stats.get("cache_hits", 0),
                cache_misses=cache_stats.get("cache_misses", 0),
                cache_hit_rate=cache_stats.get("cache_hit_rate", 0.0),
                pool_allocations=pool_stats.get("pool_allocations", 0),
                pool_deallocations=pool_stats.get("pool_deallocations", 0),
                pool_reuse_rate=pool_stats.get("pool_reuse_rate", 0.0),
            )

            results.append(result)

            # Print summary
            print(
                f"\n  Speedups: Pattern cache: {result.pattern_cache_speedup:.2f}x, "
                f"Memory pool: {result.memory_pool_speedup:.2f}x, "
                f"Combined: {result.combined_speedup:.2f}x"
            )
            print(
                f"  Memory reduction: Pattern cache: {result.pattern_cache_memory_reduction:.1%}, "
                f"Memory pool: {result.memory_pool_memory_reduction:.1%}, "
                f"Combined: {result.combined_memory_reduction:.1%}"
            )

        return results

    def generate_report(
        self,
        all_results: Dict[str, List[OptimizationBenchmarkResult]],
        output_file: str,
    ):
        """Generate comprehensive optimization impact report."""
        report_lines = []
        report_lines.append("# Optimization Impact Benchmark Report")
        report_lines.append(
            f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
        )
        report_lines.append(f"\nDevice: {self.device}")

        # Summary statistics
        report_lines.append("\n## Summary Statistics")

        for impl, results in all_results.items():
            report_lines.append(f"\n### {impl}")

            # Average improvements
            avg_pattern_speedup = sum(r.pattern_cache_speedup for r in results) / len(
                results
            )
            avg_pool_speedup = sum(r.memory_pool_speedup for r in results) / len(
                results
            )
            avg_combined_speedup = sum(r.combined_speedup for r in results) / len(
                results
            )

            avg_pattern_mem_reduction = sum(
                r.pattern_cache_memory_reduction for r in results
            ) / len(results)
            avg_pool_mem_reduction = sum(
                r.memory_pool_memory_reduction for r in results
            ) / len(results)
            avg_combined_mem_reduction = sum(
                r.combined_memory_reduction for r in results
            ) / len(results)

            avg_cache_hit_rate = sum(r.cache_hit_rate for r in results) / len(results)

            report_lines.append("\n**Average Speedups:**")
            report_lines.append(f"- Pattern Cache: {avg_pattern_speedup:.2f}x")
            report_lines.append(f"- Memory Pool: {avg_pool_speedup:.2f}x")
            report_lines.append(f"- Combined: {avg_combined_speedup:.2f}x")

            report_lines.append("\n**Average Memory Reductions:**")
            report_lines.append(f"- Pattern Cache: {avg_pattern_mem_reduction:.1%}")
            report_lines.append(f"- Memory Pool: {avg_pool_mem_reduction:.1%}")
            report_lines.append(f"- Combined: {avg_combined_mem_reduction:.1%}")

            if impl in ["ring_v2", "ring_v3"]:
                report_lines.append(
                    f"\n**Average Cache Hit Rate:** {avg_cache_hit_rate:.1%}"
                )

            # Detailed results table
            report_lines.append("\n#### Detailed Results by Sequence Length")
            report_lines.append(
                "\n| Seq Length | Baseline (ms) | Pattern Cache | Memory Pool | Combined | Cache Hit Rate |"
            )
            report_lines.append(
                "|------------|---------------|---------------|-------------|----------|----------------|"
            )

            for result in results:
                cache_hit_str = (
                    f"{result.cache_hit_rate:.1%}"
                    if result.cache_hit_rate > 0
                    else "N/A"
                )
                report_lines.append(
                    f"| {result.sequence_length:,} | {result.baseline_time_ms:.1f} | "
                    f"{result.pattern_cache_speedup:.2f}x | {result.memory_pool_speedup:.2f}x | "
                    f"{result.combined_speedup:.2f}x | {cache_hit_str} |"
                )

        # Best practices
        report_lines.append("\n## Optimization Recommendations")
        report_lines.append("\n### When to Enable Pattern Caching:")
        report_lines.append(
            "- Multiple forward passes with same sequence configuration"
        )
        report_lines.append("- Ring Attention implementations (V2/V3)")
        report_lines.append("- Sequence lengths that reuse segment patterns")
        report_lines.append(f"- Average speedup observed: {avg_pattern_speedup:.2f}x")

        report_lines.append("\n### When to Enable Memory Pooling:")
        report_lines.append("- Long sequences (>32K tokens)")
        report_lines.append("- Memory-constrained environments")
        report_lines.append("- Frequent allocation/deallocation patterns")
        report_lines.append(
            f"- Average memory reduction observed: {avg_pool_mem_reduction:.1%}"
        )

        report_lines.append("\n### Combined Optimizations:")
        report_lines.append("- Best for production deployments")
        report_lines.append("- Long-running training or inference")
        report_lines.append(f"- Combined speedup observed: {avg_combined_speedup:.2f}x")
        report_lines.append(
            f"- Combined memory reduction: {avg_combined_mem_reduction:.1%}"
        )

        # Save report
        report_content = "\n".join(report_lines)
        report_path = self.output_manager.get_output_path(output_file, "md")
        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"\nReport saved to: {report_path}")

        # Generate visualizations
        self.generate_visualizations(all_results, output_file)

    def generate_visualizations(
        self,
        all_results: Dict[str, List[OptimizationBenchmarkResult]],
        base_filename: str,
    ):
        """Generate visualization plots."""
        try:
            # Speedup comparison plot
            plt.figure(figsize=(12, 8))

            for impl, results in all_results.items():
                seq_lengths = [r.sequence_length for r in results]
                pattern_speedups = [r.pattern_cache_speedup for r in results]
                pool_speedups = [r.memory_pool_speedup for r in results]
                combined_speedups = [r.combined_speedup for r in results]

                plt.subplot(2, 2, 1)
                plt.plot(
                    seq_lengths,
                    pattern_speedups,
                    marker="o",
                    label=f"{impl} - Pattern Cache",
                )
                plt.xlabel("Sequence Length")
                plt.ylabel("Speedup")
                plt.title("Pattern Cache Speedup")
                plt.xscale("log")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(2, 2, 2)
                plt.plot(
                    seq_lengths,
                    pool_speedups,
                    marker="s",
                    label=f"{impl} - Memory Pool",
                )
                plt.xlabel("Sequence Length")
                plt.ylabel("Speedup")
                plt.title("Memory Pool Speedup")
                plt.xscale("log")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(2, 2, 3)
                plt.plot(
                    seq_lengths,
                    combined_speedups,
                    marker="^",
                    label=f"{impl} - Combined",
                )
                plt.xlabel("Sequence Length")
                plt.ylabel("Speedup")
                plt.title("Combined Optimization Speedup")
                plt.xscale("log")
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Memory reduction subplot
                plt.subplot(2, 2, 4)
                combined_reductions = [
                    r.combined_memory_reduction * 100 for r in results
                ]
                plt.plot(seq_lengths, combined_reductions, marker="d", label=f"{impl}")
                plt.xlabel("Sequence Length")
                plt.ylabel("Memory Reduction (%)")
                plt.title("Combined Memory Reduction")
                plt.xscale("log")
                plt.grid(True, alpha=0.3)
                plt.legend()

            plt.tight_layout()
            plot_path = self.output_manager.get_output_path(
                base_filename + "_speedup", "png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"Visualization saved to: {plot_path}")

        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimization impact")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["dilated", "improved", "ring_v2", "ring_v3"],
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768, 65536],
        help="Sequence lengths to test",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default="benchmark_results")

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = OptimizationBenchmarker(output_dir=args.output_dir)

    # Run benchmarks
    all_results = {}

    for impl in args.implementations:
        print(f"\n{'=' * 80}")
        print(f"Benchmarking {impl} optimization impact")
        print(f"{'=' * 80}")

        results = benchmarker.benchmark_optimization_impact(
            implementation=impl,
            sequence_lengths=args.sequence_lengths,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            batch_size=args.batch_size,
        )

        all_results[impl] = results

    # Generate report
    benchmarker.generate_report(all_results, "optimization_impact_benchmark")

    print("\nOptimization impact benchmarking complete!")


if __name__ == "__main__":
    main()
