"""Benchmark result aggregation and trend analysis."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .storage import BenchmarkStorage


class BenchmarkAggregator:
    """Aggregates and analyzes benchmark results across runs."""

    def __init__(self, storage: BenchmarkStorage | None = None):
        """Initialize aggregator."""
        self.storage = storage or BenchmarkStorage()

    def load_results(
        self, benchmark_type: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Load benchmark results with optional filtering."""
        results = []
        benchmarks = self.storage.list_benchmarks(benchmark_type, limit)

        for benchmark_info in benchmarks:
            try:
                with open(benchmark_info["path"]) as f:
                    data = json.load(f)
                    data["_info"] = benchmark_info
                    results.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        return results

    def compare_implementations(
        self, benchmark_type: str, metric: str = "execution_time_ms"
    ) -> dict[str, Any]:
        """Compare different implementations for a benchmark type."""
        results = self.load_results(benchmark_type, limit=10)
        if not results:
            return {}

        comparison = defaultdict(list)

        for result in results:
            if "results" not in result:
                continue

            timestamp = result["_info"]["timestamp"]
            for impl, metrics in result["results"].items():
                if isinstance(metrics, dict) and metric in metrics:
                    comparison[impl].append(
                        {
                            "timestamp": timestamp,
                            "value": metrics[metric],
                            "git_commit": result.get("metadata", {}).get("git_commit"),
                        }
                    )

        # Calculate statistics
        stats = {}
        for impl, values in comparison.items():
            if values:
                metric_values = [v["value"] for v in values]
                stats[impl] = {
                    "mean": np.mean(metric_values),
                    "std": np.std(metric_values),
                    "min": np.min(metric_values),
                    "max": np.max(metric_values),
                    "latest": values[0]["value"],  # Assuming sorted by timestamp
                    "history": values[:5],  # Last 5 runs
                }

        return stats

    def detect_regressions(
        self, benchmark_type: str, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """Detect performance regressions compared to baseline."""
        results = self.load_results(benchmark_type, limit=10)
        if len(results) < 2:
            return []

        regressions = []
        latest = results[0]
        baseline = results[1]  # Previous run as baseline

        if "results" not in latest or "results" not in baseline:
            return []

        for impl in latest["results"]:
            if impl not in baseline["results"]:
                continue

            latest_metrics = latest["results"][impl]
            baseline_metrics = baseline["results"][impl]

            if not isinstance(latest_metrics, dict) or not isinstance(
                baseline_metrics, dict
            ):
                continue

            # Check execution time regression
            if (
                "execution_time_ms" in latest_metrics
                and "execution_time_ms" in baseline_metrics
            ):
                latest_time = latest_metrics["execution_time_ms"]
                baseline_time = baseline_metrics["execution_time_ms"]

                if baseline_time > 0:
                    regression_pct = (latest_time - baseline_time) / baseline_time

                    if regression_pct > threshold:
                        regressions.append(
                            {
                                "implementation": impl,
                                "metric": "execution_time_ms",
                                "baseline_value": baseline_time,
                                "latest_value": latest_time,
                                "regression_pct": regression_pct * 100,
                                "baseline_commit": baseline.get("metadata", {}).get(
                                    "git_commit", "unknown"
                                ),
                                "latest_commit": latest.get("metadata", {}).get(
                                    "git_commit", "unknown"
                                ),
                            }
                        )

        return regressions

    def generate_trend_report(self, days: int = 7) -> str:
        """Generate markdown report showing performance trends."""
        cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        all_results = self.load_results()

        # Filter by date
        recent_results = []
        for result in all_results:
            try:
                timestamp_str = result["_info"]["timestamp"]
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d-%H%M-UTC")
                if dt.timestamp() >= cutoff:
                    recent_results.append(result)
            except (ValueError, KeyError):
                continue

        # Group by benchmark type
        by_type = defaultdict(list)
        for result in recent_results:
            by_type[result["_info"]["type"]].append(result)

        # Generate report
        lines = [
            "# Performance Trend Report",
            "",
            f"**Period**: Last {days} days  ",
            f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
            "",
        ]

        for benchmark_type, results in sorted(by_type.items()):
            lines.extend([f"## {benchmark_type.replace('-', ' ').title()}", ""])

            # Get latest statistics
            stats = self.compare_implementations(benchmark_type)
            if stats:
                lines.extend(["### Performance Summary", ""])
                lines.append(
                    "| Implementation | Latest (ms) | Mean (ms) | Std Dev | Trend |"
                )
                lines.append("|---|---|---|---|---|")

                for impl, data in sorted(stats.items()):
                    # Calculate trend
                    if len(data["history"]) >= 2:
                        trend_values = [h["value"] for h in data["history"]]
                        if trend_values[0] > trend_values[-1]:
                            trend = "📈 Improving"
                        elif trend_values[0] < trend_values[-1]:
                            trend = "📉 Degrading"
                        else:
                            trend = "➡️ Stable"
                    else:
                        trend = "➡️ Stable"

                    lines.append(
                        f"| {impl} | {data['latest']:.2f} | {data['mean']:.2f} | "
                        f"{data['std']:.2f} | {trend} |"
                    )

                lines.append("")

            # Check for regressions
            regressions = self.detect_regressions(benchmark_type)
            if regressions:
                lines.extend(["### ⚠️ Performance Regressions", ""])
                for reg in regressions:
                    lines.append(
                        f"- **{reg['implementation']}**: {reg['regression_pct']:.1f}% "
                        f"slower ({reg['baseline_value']:.2f}ms → {reg['latest_value']:.2f}ms)"
                    )
                lines.append("")

        return "\n".join(lines)

    def generate_comparison_matrix(self, benchmark_type: str) -> str:
        """Generate comparison matrix for implementations."""
        stats = self.compare_implementations(benchmark_type)
        if not stats:
            return "No data available"

        # Sort implementations by performance
        sorted_impls = sorted(stats.items(), key=lambda x: x[1]["mean"])

        lines = [
            f"# {benchmark_type.replace('-', ' ').title()} Comparison Matrix",
            "",
            "| Rank | Implementation | Mean Time (ms) | Relative Speed | Memory (MB) |",
            "|---|---|---|---|---|",
        ]

        baseline_time = sorted_impls[0][1]["mean"]

        for rank, (impl, data) in enumerate(sorted_impls, 1):
            relative_speed = baseline_time / data["mean"] if data["mean"] > 0 else 0
            memory = data.get("memory_mb", "N/A")

            lines.append(
                f"| {rank} | {impl} | {data['mean']:.2f} | {relative_speed:.2f}x | {memory} |"
            )

        return "\n".join(lines)

    def export_csv(self, benchmark_type: str, output_path: Path):
        """Export benchmark results to CSV for external analysis."""
        results = self.load_results(benchmark_type)
        if not results:
            return

        # Flatten results for CSV
        rows = []
        for result in results:
            base_row = {
                "timestamp": result["_info"]["timestamp"],
                "git_commit": result.get("metadata", {}).get("git_commit", ""),
                "hardware": result.get("metadata", {})
                .get("hardware", {})
                .get("gpu_names", ["CPU"])[0],
            }

            for impl, metrics in result.get("results", {}).items():
                if isinstance(metrics, dict):
                    row = base_row.copy()
                    row["implementation"] = impl
                    row.update(metrics)
                    rows.append(row)

        # Write CSV
        if rows:
            import csv

            keys = rows[0].keys()
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
