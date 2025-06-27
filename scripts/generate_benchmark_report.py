#!/usr/bin/env python3
"""
Generate comprehensive benchmark report from all stored results.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.core import BenchmarkAggregator, BenchmarkStorage


def generate_comprehensive_report():
    """Generate comprehensive benchmark report."""
    storage = BenchmarkStorage()
    aggregator = BenchmarkAggregator(storage)

    # Generate reports
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M-UTC")

    print("Generating benchmark reports...\n")

    # 1. Trend report (last 30 days)
    print("1. Generating trend report...")
    trend_report = aggregator.generate_trend_report(days=30)
    trend_path = storage.base_dir / f"trend-report-{timestamp}.md"
    trend_path.write_text(trend_report)
    print(f"   Saved: {trend_path}")

    # 2. Implementation comparison for each benchmark type
    print("\n2. Generating implementation comparisons...")
    benchmark_types = set()
    for item in storage.by_type_dir.iterdir():
        if item.is_dir():
            benchmark_types.add(item.name)

    comparison_reports = []
    for bench_type in sorted(benchmark_types):
        print(f"   - {bench_type}")
        comparison = aggregator.generate_comparison_matrix(bench_type)
        comparison_reports.append(comparison)

    # 3. Regression detection
    print("\n3. Checking for performance regressions...")
    all_regressions = []
    for bench_type in benchmark_types:
        regressions = aggregator.detect_regressions(bench_type, threshold=0.1)
        if regressions:
            all_regressions.extend(regressions)
            print(f"   ⚠️  Found {len(regressions)} regressions in {bench_type}")

    # 4. Generate master report
    print("\n4. Generating master report...")
    master_lines = [
        "# Benchmark Report Summary",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Overview",
        f"- Total benchmark types: {len(benchmark_types)}",
        f"- Performance regressions: {len(all_regressions)}",
        "",
    ]

    # Add regression summary if any
    if all_regressions:
        master_lines.extend(
            [
                "## ⚠️ Performance Regressions Detected",
                "",
            ]
        )
        for reg in all_regressions:
            master_lines.append(
                f"- **{reg['implementation']}** in {reg.get('benchmark_type', 'unknown')}: "
                f"{reg['regression_pct']:.1f}% slower"
            )
        master_lines.append("")

    # Add comparison matrices
    master_lines.extend(
        [
            "## Implementation Comparisons",
            "",
        ]
    )
    for comparison in comparison_reports:
        master_lines.append(comparison)
        master_lines.append("")

    # Add best implementations summary
    master_lines.extend(
        [
            "## Best Implementations by Benchmark Type",
            "",
            "| Benchmark Type | Best Implementation | Mean Time (ms) |",
            "|---|---|---|",
        ]
    )

    for bench_type in sorted(benchmark_types):
        stats = aggregator.compare_implementations(bench_type)
        if stats:
            best = min(stats.items(), key=lambda x: x[1]["mean"])
            master_lines.append(f"| {bench_type} | {best[0]} | {best[1]['mean']:.2f} |")

    master_lines.append("")

    # Save master report
    master_path = storage.base_dir / f"benchmark-report-{timestamp}.md"
    master_path.write_text("\n".join(master_lines))
    print(f"   Saved: {master_path}")

    # 5. Update latest symlink
    print("\n5. Updating latest report link...")
    latest_report = storage.base_dir / "LATEST_REPORT.md"
    if latest_report.exists():
        latest_report.unlink()
    latest_report.write_text("\n".join(master_lines))
    print(f"   Updated: {latest_report}")

    print("\n✅ Report generation complete!")
    print(f"   View at: {latest_report}")


def export_data_for_analysis():
    """Export benchmark data to CSV for external analysis."""
    storage = BenchmarkStorage()
    aggregator = BenchmarkAggregator(storage)

    print("\nExporting benchmark data to CSV...")

    export_dir = storage.base_dir / "exports"
    export_dir.mkdir(exist_ok=True)

    # Export each benchmark type
    for type_dir in storage.by_type_dir.iterdir():
        if type_dir.is_dir():
            bench_type = type_dir.name
            csv_path = export_dir / f"{bench_type}.csv"

            try:
                aggregator.export_csv(bench_type, csv_path)
                print(f"   Exported: {csv_path.name}")
            except Exception as e:
                print(f"   Error exporting {bench_type}: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("BENCHMARK REPORT GENERATOR")
    print("=" * 60)

    generate_comprehensive_report()
    export_data_for_analysis()

    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
