#!/usr/bin/env python3
"""
Visualize Phase 1 Performance Results

Creates charts and visualizations for Phase 1 benchmark results.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_latest_results():
    """Load the most recent benchmark results."""
    benchmark_dir = Path("benchmarks")
    result_files = list(benchmark_dir.glob("phase1-performance-results-*.json"))

    if not result_files:
        print("No benchmark results found!")
        return None

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, "r") as f:
        return json.load(f)


def create_speedup_chart(results):
    """Create speedup comparison chart."""
    implementations = []
    speedups = []
    colors = []

    # Define color scheme
    color_map = {
        "baseline": "#808080",
        "phase1_standard": "#3498db",
        "phase1_ring": "#2ecc71",
        "phase1_sparse": "#e74c3c",
        "phase1_v2": "#9b59b6",
    }

    # Calculate average speedups
    baseline_times = {}
    for impl_key, impl_data in results["implementations"].items():
        if impl_key == "baseline":
            for seq_len, metrics in impl_data["metrics"].items():
                if metrics["success"]:
                    baseline_times[seq_len] = metrics["time_ms"]

    for impl_key, impl_data in results["implementations"].items():
        if impl_key != "baseline":
            impl_speedups = []
            for seq_len, metrics in impl_data["metrics"].items():
                if metrics["success"] and seq_len in baseline_times:
                    speedup = baseline_times[seq_len] / metrics["time_ms"]
                    impl_speedups.append(speedup)

            if impl_speedups:
                implementations.append(impl_data["name"])
                speedups.append(np.mean(impl_speedups))
                colors.append(color_map.get(impl_key, "#95a5a6"))

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(implementations)), speedups, color=colors)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Implementation")
    plt.ylabel("Average Speedup vs Baseline")
    plt.title("Phase 1 Performance Improvements")
    plt.xticks(range(len(implementations)), implementations, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save chart
    timestamp = results["timestamp"]
    chart_file = f"docs/reports/phase1-speedup-chart-{timestamp}.png"
    os.makedirs(os.path.dirname(chart_file), exist_ok=True)
    plt.savefig(chart_file, dpi=150)
    print(f"Saved speedup chart to: {chart_file}")
    plt.close()


def create_scaling_chart(results):
    """Create sequence length scaling chart."""
    plt.figure(figsize=(12, 7))

    # Define markers and colors
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#808080", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, (impl_key, impl_data) in enumerate(results["implementations"].items()):
        seq_lengths = []
        throughputs = []

        for seq_len_str, metrics in sorted(
            impl_data["metrics"].items(), key=lambda x: int(x[0])
        ):
            if metrics["success"]:
                seq_lengths.append(int(seq_len_str))
                throughputs.append(metrics["throughput_tokens_sec"])

        if seq_lengths:
            plt.plot(
                seq_lengths,
                throughputs,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=impl_data["name"],
                linewidth=2,
                markersize=8,
            )

    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Phase 1 Throughput Scaling")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save chart
    timestamp = results["timestamp"]
    chart_file = f"docs/reports/phase1-scaling-chart-{timestamp}.png"
    plt.savefig(chart_file, dpi=150)
    print(f"Saved scaling chart to: {chart_file}")
    plt.close()


def create_summary_table(results):
    """Create a summary table of key metrics."""

    # Extract key metrics
    summary_data = []

    for impl_key, impl_data in results["implementations"].items():
        max_seq_len = 0
        avg_throughput = []

        for seq_len_str, metrics in impl_data["metrics"].items():
            if metrics["success"]:
                seq_len = int(seq_len_str)
                max_seq_len = max(max_seq_len, seq_len)
                avg_throughput.append(metrics["throughput_tokens_sec"])

        summary_data.append(
            {
                "name": impl_data["name"],
                "max_seq_len": max_seq_len,
                "avg_throughput": np.mean(avg_throughput) if avg_throughput else 0,
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    # Create table data
    headers = ["Implementation", "Max Sequence", "Avg Throughput"]
    table_data = []

    for data in summary_data:
        table_data.append(
            [
                data["name"],
                f"{data['max_seq_len']:,}",
                f"{data['avg_throughput']:.0f} tok/s",
            ]
        )

    # Create table
    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="left", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Phase 1 Implementation Summary", fontsize=14, fontweight="bold", pad=20)

    # Save table
    timestamp = results["timestamp"]
    table_file = f"docs/reports/phase1-summary-table-{timestamp}.png"
    plt.savefig(table_file, dpi=150, bbox_inches="tight")
    print(f"Saved summary table to: {table_file}")
    plt.close()


def main():
    """Generate visualizations for Phase 1 results."""

    # Load results
    results = load_latest_results()
    if not results:
        return

    print("\nGenerating visualizations...")

    # Create charts
    create_speedup_chart(results)
    create_scaling_chart(results)
    create_summary_table(results)

    print("\n✓ Visualizations completed!")


if __name__ == "__main__":
    main()
