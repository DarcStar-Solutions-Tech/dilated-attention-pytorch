#!/usr/bin/env python3
"""
Visualize performance metrics over time using plotly.

This script creates interactive charts showing:
- Performance trends over time
- Memory usage patterns
- Regression detection
"""

import json
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_performance_history(
    baseline_dir: str = "tests/performance_baselines",
) -> list[dict]:
    """Load performance history from JSON file."""
    history_file = Path(baseline_dir) / "history.json"
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return []


def create_performance_dashboard(
    history: list[dict], output_file: str = "performance_dashboard.html"
):
    """Create interactive performance dashboard."""
    if not history:
        print("No performance history to visualize.")
        return

    # Group data by implementation and configuration
    data_by_impl = {}
    for entry in history:
        impl = entry["implementation"]
        config = entry["config"]
        key = f"{impl}_{config}"

        if key not in data_by_impl:
            data_by_impl[key] = {
                "timestamps": [],
                "execution_times": [],
                "memory_allocated": [],
                "regression_pcts": [],
                "passed": [],
            }

        data = data_by_impl[key]
        data["timestamps"].append(datetime.fromisoformat(entry["timestamp"]))
        data["execution_times"].append(entry["metrics"]["execution_time_ms"])
        data["memory_allocated"].append(entry["metrics"].get("memory_allocated_mb", 0))
        data["regression_pcts"].append(entry["regression_pct"])
        data["passed"].append(entry["passed"])

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Execution Time Over Time",
            "Memory Usage Over Time",
            "Performance Regression Percentage",
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Add traces for each implementation/config
    for idx, (key, data) in enumerate(data_by_impl.items()):
        color = colors[idx % len(colors)]
        impl_name, config = key.rsplit("_", 3)[0], key.rsplit("_", 3)[-3:]
        display_name = f"{impl_name} ({'.'.join(config)})"

        # Execution time trace
        fig.add_trace(
            go.Scatter(
                x=data["timestamps"],
                y=data["execution_times"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Memory usage trace
        if any(data["memory_allocated"]):  # Only show if memory data exists
            fig.add_trace(
                go.Scatter(
                    x=data["timestamps"],
                    y=data["memory_allocated"],
                    mode="lines+markers",
                    name=display_name,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Regression percentage trace
        fig.add_trace(
            go.Scatter(
                x=data["timestamps"],
                y=data["regression_pcts"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color),
                marker=dict(
                    symbol=["circle" if p else "x" for p in data["passed"]], size=8
                ),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Add regression threshold line
    if history:
        fig.add_hline(
            y=15.0,  # Default threshold
            line_dash="dash",
            line_color="red",
            annotation_text="Regression Threshold (15%)",
            row=3,
            col=1,
        )

    # Update layout
    fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
    fig.update_yaxes(title_text="Regression %", row=3, col=1)

    fig.update_layout(
        title="Dilated Attention Performance Dashboard",
        height=1000,
        hovermode="x unified",
        template="plotly_white",
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"Performance dashboard saved to: {output_file}")


def create_comparison_chart(
    history: list[dict], output_file: str = "performance_comparison.html"
):
    """Create comparison chart of different implementations."""
    if not history:
        print("No performance history to compare.")
        return

    # Get latest metrics for each implementation/config
    latest_metrics = {}
    for entry in history:
        key = f"{entry['implementation']}_{entry['config']}"
        latest_metrics[key] = entry["metrics"]["execution_time_ms"]

    # Parse and group by configuration
    configs = {}
    for key, time_ms in latest_metrics.items():
        impl, config = key.rsplit("_", 3)[0], "_".join(key.rsplit("_", 3)[-3:])
        if config not in configs:
            configs[config] = {}
        configs[config][impl] = time_ms

    # Create grouped bar chart
    fig = go.Figure()

    implementations = list(
        set(impl for config_data in configs.values() for impl in config_data.keys())
    )

    for impl in sorted(implementations):
        times = []
        config_names = []
        for config, impl_data in sorted(configs.items()):
            if impl in impl_data:
                times.append(impl_data[impl])
                config_names.append(config)

        fig.add_trace(
            go.Bar(
                name=impl,
                x=config_names,
                y=times,
                text=[f"{t:.1f}ms" for t in times],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Performance Comparison Across Implementations",
        xaxis_title="Configuration",
        yaxis_title="Execution Time (ms)",
        barmode="group",
        template="plotly_white",
        height=600,
    )

    fig.write_html(output_file)
    print(f"Comparison chart saved to: {output_file}")


def main():
    """Generate all visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize performance metrics")
    parser.add_argument(
        "--baseline-dir",
        default="tests/performance_baselines",
        help="Directory containing performance data",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output files")

    args = parser.parse_args()

    # Load history
    history = load_performance_history(args.baseline_dir)

    if not history:
        print("No performance history found. Run performance tests first.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    create_performance_dashboard(
        history, str(output_dir / "performance_dashboard.html")
    )
    create_comparison_chart(history, str(output_dir / "performance_comparison.html"))

    print(f"\nFound {len(history)} performance entries")
    implementations = set(entry["implementation"] for entry in history)
    print(f"Implementations tracked: {', '.join(sorted(implementations))}")


if __name__ == "__main__":
    main()
