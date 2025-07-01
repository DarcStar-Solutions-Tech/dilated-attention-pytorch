#!/usr/bin/env python3
"""
Analysis script for RingDilatedAttentionV2Collective benchmark results.

This script analyzes scaling efficiency, communication overhead, and performance
characteristics across different GPU configurations.
"""

import json
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class RingV2ScalingAnalyzer:
    """Analyzes scaling performance of RingDilatedAttentionV2Collective."""

    def __init__(self, results_dir: str = "benchmark_results/ring_v2_collective"):
        self.results_dir = results_dir
        self.data = []
        self.load_results()

    def load_results(self):
        """Load all benchmark results from the results directory."""
        if not os.path.exists(self.results_dir):
            print(f"Results directory not found: {self.results_dir}")
            return

        for filename in os.listdir(self.results_dir):
            if filename.endswith(".json") and filename.startswith("ring_v2_collective"):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        self.data.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded {len(self.data)} benchmark result files")

    def create_dataframe(self) -> pd.DataFrame:
        """Convert loaded data to pandas DataFrame."""
        rows = []
        for dataset in self.data:
            world_size = dataset.get("world_size", 1)
            for result in dataset.get("results", []):
                row = {
                    "world_size": world_size,
                    "seq_length": result["seq_length"],
                    "ring_size": result["ring_size"],
                    "forward_time_ms": result["forward_time_ms"],
                    "backward_time_ms": result["backward_time_ms"],
                    "total_time_ms": result["total_time_ms"],
                    "throughput": result["throughput_tokens_per_sec"],
                    "memory_allocated_gb": result["memory_allocated_gb"],
                    "communication_time_ms": result.get("communication_time_ms", 0),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def plot_scaling_efficiency(self, df: pd.DataFrame, output_dir: str):
        """Plot scaling efficiency across different configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("RingDilatedAttentionV2Collective Scaling Analysis", fontsize=16)

        # Group by sequence length
        seq_lengths = sorted(df["seq_length"].unique())

        # 1. Throughput vs Ring Size
        ax = axes[0, 0]
        for seq_len in seq_lengths:
            data = df[df["seq_length"] == seq_len]
            grouped = data.groupby("ring_size")["throughput"].mean()
            ax.plot(
                grouped.index,
                grouped.values,
                marker="o",
                label=f"Seq {seq_len}",
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel("Ring Size")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput Scaling with Ring Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Speedup vs Ring Size
        ax = axes[0, 1]
        for seq_len in seq_lengths:
            data = df[df["seq_length"] == seq_len]
            baseline = data[data["ring_size"] == 1]["total_time_ms"].mean()
            grouped = data.groupby("ring_size")["total_time_ms"].mean()
            speedup = baseline / grouped
            ax.plot(
                speedup.index,
                speedup.values,
                marker="o",
                label=f"Seq {seq_len}",
                linewidth=2,
                markersize=8,
            )
            # Add ideal scaling line
            if seq_len == seq_lengths[0]:
                ax.plot(
                    speedup.index,
                    speedup.index,
                    "k--",
                    alpha=0.5,
                    label="Ideal scaling",
                )
        ax.set_xlabel("Ring Size")
        ax.set_ylabel("Speedup")
        ax.set_title("Speedup vs Ring Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Efficiency vs Ring Size
        ax = axes[1, 0]
        for seq_len in seq_lengths:
            data = df[df["seq_length"] == seq_len]
            baseline = data[data["ring_size"] == 1]["total_time_ms"].mean()
            grouped = data.groupby("ring_size")["total_time_ms"].mean()
            speedup = baseline / grouped
            efficiency = (speedup / grouped.index) * 100
            ax.plot(
                efficiency.index,
                efficiency.values,
                marker="o",
                label=f"Seq {seq_len}",
                linewidth=2,
                markersize=8,
            )
        ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="100% efficiency")
        ax.set_xlabel("Ring Size")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title("Scaling Efficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Communication Overhead
        ax = axes[1, 1]
        comm_data = df[df["communication_time_ms"] > 0]
        if not comm_data.empty:
            for seq_len in seq_lengths:
                data = comm_data[comm_data["seq_length"] == seq_len]
                if not data.empty:
                    grouped = data.groupby("ring_size")["communication_time_ms"].mean()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        marker="o",
                        label=f"Seq {seq_len}",
                        linewidth=2,
                        markersize=8,
                    )
            ax.set_xlabel("Ring Size")
            ax.set_ylabel("Communication Time (ms)")
            ax.set_title("Communication Overhead")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No communication data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "scaling_efficiency_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved scaling efficiency plot to: {output_path}")
        plt.close()

    def plot_memory_analysis(self, df: pd.DataFrame, output_dir: str):
        """Plot memory usage analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Memory Usage Analysis", fontsize=16)

        # 1. Memory vs Sequence Length
        for ring_size in sorted(df["ring_size"].unique()):
            data = df[df["ring_size"] == ring_size]
            grouped = data.groupby("seq_length")["memory_allocated_gb"].mean()
            ax1.plot(
                grouped.index,
                grouped.values,
                marker="o",
                label=f"Ring size {ring_size}",
                linewidth=2,
                markersize=8,
            )
        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Memory Allocated (GB)")
        ax1.set_title("Memory Usage vs Sequence Length")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Memory Efficiency (tokens per GB)
        for ring_size in sorted(df["ring_size"].unique()):
            data = df[df["ring_size"] == ring_size]
            data["tokens_per_gb"] = (
                data["seq_length"] * 2
            ) / data[  # batch_size=2 by default
                "memory_allocated_gb"
            ]
            grouped = data.groupby("seq_length")["tokens_per_gb"].mean()
            ax2.plot(
                grouped.index,
                grouped.values,
                marker="o",
                label=f"Ring size {ring_size}",
                linewidth=2,
                markersize=8,
            )
        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Tokens per GB")
        ax2.set_title("Memory Efficiency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "memory_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved memory analysis plot to: {output_path}")
        plt.close()

    def plot_performance_heatmap(self, df: pd.DataFrame, output_dir: str):
        """Create heatmap of performance across configurations."""
        # Pivot data for heatmap
        pivot_throughput = df.pivot_table(
            values="throughput",
            index="seq_length",
            columns="ring_size",
            aggfunc="mean",
        )

        pivot_efficiency = pd.DataFrame()
        for seq_len in df["seq_length"].unique():
            baseline = df[(df["seq_length"] == seq_len) & (df["ring_size"] == 1)][
                "total_time_ms"
            ].mean()
            for ring_size in df["ring_size"].unique():
                time = df[
                    (df["seq_length"] == seq_len) & (df["ring_size"] == ring_size)
                ]["total_time_ms"].mean()
                speedup = baseline / time
                efficiency = (speedup / ring_size) * 100
                pivot_efficiency.loc[seq_len, ring_size] = efficiency

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Performance Heatmaps", fontsize=16)

        # Throughput heatmap
        sns.heatmap(
            pivot_throughput,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            ax=ax1,
            cbar_kws={"label": "Tokens/sec"},
        )
        ax1.set_title("Throughput Heatmap")
        ax1.set_xlabel("Ring Size")
        ax1.set_ylabel("Sequence Length")

        # Efficiency heatmap
        sns.heatmap(
            pivot_efficiency,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=100,
            vmin=50,
            vmax=100,
            ax=ax2,
            cbar_kws={"label": "Efficiency (%)"},
        )
        ax2.set_title("Scaling Efficiency Heatmap")
        ax2.set_xlabel("Ring Size")
        ax2.set_ylabel("Sequence Length")

        plt.tight_layout()
        output_path = os.path.join(output_dir, "performance_heatmaps.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved performance heatmaps to: {output_path}")
        plt.close()

    def generate_report(self, df: pd.DataFrame, output_dir: str):
        """Generate a comprehensive text report."""
        report_path = os.path.join(output_dir, "scaling_analysis_report.txt")

        with open(report_path, "w") as f:
            f.write("RingDilatedAttentionV2Collective Scaling Analysis Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary statistics
            f.write("Summary Statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total configurations tested: {len(df)}\n")
            f.write(f"Sequence lengths: {sorted(df['seq_length'].unique())}\n")
            f.write(f"Ring sizes: {sorted(df['ring_size'].unique())}\n")
            f.write(f"World sizes: {sorted(df['world_size'].unique())}\n\n")

            # Best configurations
            f.write("Best Configurations (by throughput)\n")
            f.write("-" * 40 + "\n")
            for seq_len in sorted(df["seq_length"].unique()):
                best = df[df["seq_length"] == seq_len].nlargest(1, "throughput").iloc[0]
                f.write(f"Seq {seq_len}: Ring size {best['ring_size']:.0f}, ")
                f.write(f"Throughput {best['throughput']:.0f} tokens/sec\n")

            # Scaling efficiency
            f.write("\nScaling Efficiency Analysis\n")
            f.write("-" * 40 + "\n")
            for seq_len in sorted(df["seq_length"].unique()):
                f.write(f"\nSequence Length: {seq_len}\n")
                data = df[df["seq_length"] == seq_len]
                baseline = data[data["ring_size"] == 1]["total_time_ms"].mean()

                for ring_size in sorted(data["ring_size"].unique()):
                    if ring_size == 1:
                        continue
                    time = data[data["ring_size"] == ring_size]["total_time_ms"].mean()
                    speedup = baseline / time
                    efficiency = (speedup / ring_size) * 100
                    f.write(
                        f"  Ring {ring_size}: {speedup:.2f}x speedup, "
                        f"{efficiency:.1f}% efficiency\n"
                    )

            # Communication overhead
            f.write("\nCommunication Overhead Analysis\n")
            f.write("-" * 40 + "\n")
            comm_data = df[df["communication_time_ms"] > 0]
            if not comm_data.empty:
                for ring_size in sorted(comm_data["ring_size"].unique()):
                    if ring_size == 1:
                        continue
                    data = comm_data[comm_data["ring_size"] == ring_size]
                    avg_comm = data["communication_time_ms"].mean()
                    avg_total = data["total_time_ms"].mean()
                    comm_percent = (avg_comm / avg_total) * 100
                    f.write(
                        f"Ring {ring_size}: {avg_comm:.2f}ms "
                        f"({comm_percent:.1f}% of total time)\n"
                    )
            else:
                f.write("No communication overhead data available\n")

            # Memory usage
            f.write("\nMemory Usage Summary\n")
            f.write("-" * 40 + "\n")
            memory_summary = df.groupby(["seq_length", "ring_size"])[
                "memory_allocated_gb"
            ].mean()
            for (seq_len, ring_size), memory in memory_summary.items():
                f.write(f"Seq {seq_len}, Ring {ring_size}: {memory:.2f} GB allocated\n")

        print(f"Saved analysis report to: {report_path}")

    def run_analysis(self, output_dir: Optional[str] = None):
        """Run complete analysis and generate all outputs."""
        if not self.data:
            print("No data loaded. Please check the results directory.")
            return

        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)

        # Create dataframe
        df = self.create_dataframe()
        print(f"Analyzing {len(df)} benchmark results...")

        # Generate plots
        self.plot_scaling_efficiency(df, output_dir)
        self.plot_memory_analysis(df, output_dir)
        self.plot_performance_heatmap(df, output_dir)

        # Generate report
        self.generate_report(df, output_dir)

        print(f"\nAnalysis complete! Results saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze RingDilatedAttentionV2Collective benchmark results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results/ring_v2_collective",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for analysis results (default: results_dir/analysis)",
    )

    args = parser.parse_args()

    analyzer = RingV2ScalingAnalyzer(args.results_dir)
    analyzer.run_analysis(args.output_dir)


if __name__ == "__main__":
    main()
