#!/usr/bin/env python3
"""
Comprehensive performance benchmarking suite for standardized ring attention implementations.

This script benchmarks all 4 ring attention variants across different:
- Sequence lengths (1K to 1M tokens)
- GPU configurations (1, 2, 4, 8 GPUs)
- Memory usage patterns
- Throughput characteristics
"""

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import time
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from dilated_attention_pytorch import (
    StandardRingAttention,
    DistributedRingAttention,
    HilbertRingAttention,
    RingBlockSparseAttention,
    RingAttentionConfig,
)


class RingAttentionBenchmark:
    """Comprehensive benchmark suite for ring attention implementations."""

    def __init__(
        self,
        implementations: List[str] = None,
        sequence_lengths: List[int] = None,
        batch_sizes: List[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        segment_lengths: List[int] = None,
        dilation_rates: List[int] = None,
        output_dir: str = "benchmarks/results/ring",
    ):
        self.implementations = implementations or [
            "standard",
            "distributed",
            "hilbert",
            "block_sparse",
        ]
        self.sequence_lengths = sequence_lengths or [
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.batch_sizes = batch_sizes or [1, 2, 4]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.segment_lengths = segment_lengths or [2048, 4096, 8192]
        self.dilation_rates = dilation_rates or [1, 2, 4]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def create_attention_module(self, impl_name: str) -> torch.nn.Module:
        """Create attention module based on implementation name."""
        config = RingAttentionConfig(
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            dropout=0.0,
            ring_size=self.world_size if self.world_size > 1 else None,
        )

        impl_map = {
            "standard": StandardRingAttention,
            "distributed": DistributedRingAttention,
            "hilbert": HilbertRingAttention,
            "block_sparse": RingBlockSparseAttention,
        }

        if impl_name not in impl_map:
            raise ValueError(f"Unknown implementation: {impl_name}")

        cls = impl_map[impl_name]

        # Special handling for block sparse
        if impl_name == "block_sparse":
            return cls(
                config=config,
                sparsity_ratio=0.1,  # 90% sparse
                pattern_type="dilated_sparse",
                device=self.device,
                dtype=self.dtype,
            )
        else:
            return cls(config=config, device=self.device, dtype=self.dtype)

    def measure_memory_and_time(
        self,
        model: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        warmup_iters: int = 3,
        benchmark_iters: int = 10,
    ) -> Tuple[float, float, float]:
        """Measure memory usage and execution time."""
        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = model(q, k, v)

        # Clear memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Measure
        start_mem = torch.cuda.memory_allocated() if self.device.type == "cuda" else 0

        # Time measurement with CUDA events
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
        else:
            start_time = time.perf_counter()

        # Benchmark iterations
        with torch.no_grad():
            for _ in range(benchmark_iters):
                _ = model(q, k, v)

        # Record end time
        if self.device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = (
                start_event.elapsed_time(end_event) / 1000.0
            )  # Convert to seconds
        else:
            elapsed_time = time.perf_counter() - start_time

        avg_time = elapsed_time / benchmark_iters

        # Memory measurement
        if self.device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() - start_mem
            current_mem = torch.cuda.memory_allocated() - start_mem
        else:
            peak_mem = current_mem = 0

        return avg_time, peak_mem / (1024**2), current_mem / (1024**2)  # Convert to MB

    def benchmark_implementation(
        self,
        impl_name: str,
        seq_len: int,
        batch_size: int,
    ) -> Dict:
        """Benchmark a single implementation with given parameters."""
        print(f"\nBenchmarking {impl_name} - Seq: {seq_len}, Batch: {batch_size}")

        try:
            # Create model
            model = self.create_attention_module(impl_name)
            model.eval()

            # Create inputs
            q = torch.randn(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Measure
            time_taken, peak_mem, current_mem = self.measure_memory_and_time(
                model, q, k, v
            )

            # Calculate throughput
            total_tokens = batch_size * seq_len
            throughput = total_tokens / time_taken

            # Memory per token
            mem_per_token = peak_mem / total_tokens

            result = {
                "implementation": impl_name,
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "world_size": self.world_size,
                "time_ms": time_taken * 1000,
                "peak_memory_mb": peak_mem,
                "current_memory_mb": current_mem,
                "throughput_tokens_per_sec": throughput,
                "memory_per_token_kb": mem_per_token * 1024,
                "success": True,
                "error": None,
            }

            # Cleanup
            del model, q, k, v
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            return result

        except Exception as e:
            print(f"  Error: {str(e)}")
            return {
                "implementation": impl_name,
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "world_size": self.world_size,
                "time_ms": None,
                "peak_memory_mb": None,
                "current_memory_mb": None,
                "throughput_tokens_per_sec": None,
                "memory_per_token_kb": None,
                "success": False,
                "error": str(e),
            }

    def run_benchmarks(self):
        """Run all benchmarks."""
        print("Starting comprehensive ring attention benchmarks")
        print(f"World size: {self.world_size}, Rank: {self.rank}")
        print(f"Device: {self.device}, Dtype: {self.dtype}")

        # Iterate through all combinations
        for impl in self.implementations:
            for seq_len in self.sequence_lengths:
                for batch_size in self.batch_sizes:
                    # Skip very large sequences on small GPU counts
                    if seq_len > 65536 and self.world_size < 4:
                        print(f"Skipping {impl} seq_len={seq_len} (need 4+ GPUs)")
                        continue

                    result = self.benchmark_implementation(impl, seq_len, batch_size)
                    self.results.append(result)

                    # Save intermediate results
                    if len(self.results) % 10 == 0:
                        self.save_results()

        # Final save
        self.save_results()

        # Generate visualizations
        if self.rank == 0:  # Only on main process
            self.create_visualizations()

    def save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = self.output_dir / f"ring_benchmark_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / f"ring_benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        print("\nResults saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")

    def create_visualizations(self):
        """Create visualization plots from results."""
        df = pd.DataFrame(self.results)
        successful_df = df[df["success"]]

        if len(successful_df) == 0:
            print("No successful benchmarks to visualize")
            return

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

        # 1. Memory scaling plot
        fig, ax = plt.subplots(figsize=(12, 8))
        for impl in self.implementations:
            impl_df = successful_df[successful_df["implementation"] == impl]
            if len(impl_df) > 0:
                # Group by sequence length and average across batch sizes
                grouped = (
                    impl_df.groupby("sequence_length")
                    .agg({"peak_memory_mb": "mean", "memory_per_token_kb": "mean"})
                    .reset_index()
                )

                ax.plot(
                    grouped["sequence_length"],
                    grouped["peak_memory_mb"],
                    marker="o",
                    label=impl,
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax.set_ylabel("Peak Memory Usage (MB)", fontsize=12)
        ax.set_title(
            f"Memory Scaling: Ring Attention Implementations (World Size: {self.world_size})",
            fontsize=14,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        memory_plot_path = (
            self.output_dir / f"ring_memory_scaling_{self.world_size}gpu.png"
        )
        plt.tight_layout()
        plt.savefig(memory_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Throughput comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        for impl in self.implementations:
            impl_df = successful_df[successful_df["implementation"] == impl]
            if len(impl_df) > 0:
                grouped = (
                    impl_df.groupby("sequence_length")
                    .agg({"throughput_tokens_per_sec": "mean"})
                    .reset_index()
                )

                ax.plot(
                    grouped["sequence_length"],
                    grouped["throughput_tokens_per_sec"],
                    marker="o",
                    label=impl,
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)
        ax.set_title(
            f"Throughput Comparison: Ring Attention Implementations (World Size: {self.world_size})",
            fontsize=14,
        )
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        throughput_plot_path = (
            self.output_dir / f"ring_throughput_{self.world_size}gpu.png"
        )
        plt.tight_layout()
        plt.savefig(throughput_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Memory per token (efficiency)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Bar plot for different sequence lengths
        seq_lengths_to_plot = [2048, 8192, 32768]
        x = np.arange(len(self.implementations))
        width = 0.25

        for i, seq_len in enumerate(seq_lengths_to_plot):
            seq_df = successful_df[successful_df["sequence_length"] == seq_len]
            if len(seq_df) > 0:
                mem_per_token = []
                for impl in self.implementations:
                    impl_val = seq_df[seq_df["implementation"] == impl][
                        "memory_per_token_kb"
                    ].mean()
                    mem_per_token.append(impl_val if not pd.isna(impl_val) else 0)

                ax.bar(
                    x + i * width,
                    mem_per_token,
                    width,
                    label=f"{seq_len} tokens",
                    alpha=0.8,
                )

        ax.set_xlabel("Implementation", fontsize=12)
        ax.set_ylabel("Memory per Token (KB)", fontsize=12)
        ax.set_title(
            f"Memory Efficiency: Ring Attention Implementations (World Size: {self.world_size})",
            fontsize=14,
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.implementations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        efficiency_plot_path = (
            self.output_dir / f"ring_memory_efficiency_{self.world_size}gpu.png"
        )
        plt.tight_layout()
        plt.savefig(efficiency_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("\nVisualizations saved to:")
        print(f"  - {memory_plot_path}")
        print(f"  - {throughput_plot_path}")
        print(f"  - {efficiency_plot_path}")

        # 4. Generate summary report
        self.generate_summary_report(successful_df)

    def generate_summary_report(self, df: pd.DataFrame):
        """Generate a summary report of the benchmarks."""
        report_path = (
            self.output_dir / f"ring_benchmark_summary_{self.world_size}gpu.md"
        )

        with open(report_path, "w") as f:
            f.write("# Ring Attention Benchmark Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**World Size**: {self.world_size} GPU(s)\n")
            f.write(f"**Device**: {self.device}\n\n")

            f.write("## Summary Statistics\n\n")

            # Best implementation for each metric
            f.write("### Best Performance by Metric\n\n")

            # Lowest memory usage
            min_mem = df.loc[df["peak_memory_mb"].idxmin()]
            f.write(
                f"- **Lowest Memory Usage**: {min_mem['implementation']} "
                f"({min_mem['peak_memory_mb']:.2f} MB at {min_mem['sequence_length']} tokens)\n"
            )

            # Highest throughput
            max_throughput = df.loc[df["throughput_tokens_per_sec"].idxmax()]
            f.write(
                f"- **Highest Throughput**: {max_throughput['implementation']} "
                f"({max_throughput['throughput_tokens_per_sec']:.0f} tokens/sec at {max_throughput['sequence_length']} tokens)\n"
            )

            # Most memory efficient
            min_mem_per_token = df.loc[df["memory_per_token_kb"].idxmin()]
            f.write(
                f"- **Most Memory Efficient**: {min_mem_per_token['implementation']} "
                f"({min_mem_per_token['memory_per_token_kb']:.3f} KB/token)\n\n"
            )

            # Detailed table for each implementation
            f.write("### Performance by Implementation\n\n")

            for impl in self.implementations:
                impl_df = df[df["implementation"] == impl]
                if len(impl_df) > 0:
                    f.write(f"#### {impl.title()} Ring Attention\n\n")

                    # Create summary table
                    summary = (
                        impl_df.groupby("sequence_length")
                        .agg(
                            {
                                "time_ms": "mean",
                                "peak_memory_mb": "mean",
                                "throughput_tokens_per_sec": "mean",
                                "memory_per_token_kb": "mean",
                            }
                        )
                        .round(2)
                    )

                    f.write(summary.to_markdown())
                    f.write("\n\n")

            # Maximum sequence length achieved
            f.write("### Maximum Sequence Lengths\n\n")
            for impl in self.implementations:
                impl_df = df[(df["implementation"] == impl) & df["success"]]
                if len(impl_df) > 0:
                    max_seq = impl_df["sequence_length"].max()
                    f.write(f"- **{impl}**: {max_seq:,} tokens\n")

        print(f"\nSummary report saved to: {report_path}")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for ring attention implementations"
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["standard", "distributed", "hilbert", "block_sparse"],
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=None,
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/ring",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create benchmark
    benchmark = RingAttentionBenchmark(
        implementations=args.implementations,
        sequence_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        output_dir=args.output_dir,
    )

    # Run benchmarks
    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
