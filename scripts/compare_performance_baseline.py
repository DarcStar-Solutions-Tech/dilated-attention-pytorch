#!/usr/bin/env python3
"""
Compare performance test results with baseline and generate reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_baseline(baseline_path: Path) -> dict[str, dict[str, float]]:
    """Load baseline performance metrics."""
    if not baseline_path.exists():
        return {}

    with open(baseline_path) as f:
        data = json.load(f)

    # Convert to simple format: impl -> config -> time_ms
    baselines = {}
    for impl, configs in data.items():
        baselines[impl] = {}
        for config, metrics in configs.items():
            baselines[impl][config] = metrics.get("execution_time_ms", 0)

    return baselines


def load_results(results_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load test results."""
    with open(results_path) as f:
        return json.load(f)


def compare_performance(
    current: float, baseline: float, threshold: float = 15.0
) -> tuple[bool, float]:
    """Compare current performance with baseline."""
    if baseline == 0:
        return True, 0.0

    regression_pct = ((current - baseline) / baseline) * 100
    passed = regression_pct <= threshold

    return passed, regression_pct


def generate_report(
    baselines: dict[str, dict[str, float]],
    results: dict[str, list[dict[str, Any]]],
    threshold: float = 15.0,
) -> dict[str, Any]:
    """Generate performance comparison report."""
    report = {
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "no_baseline": 0,
        },
        "implementations": {},
        "regressions": [],
    }

    # Process each implementation
    for impl_name, test_results in results.items():
        impl_baselines = baselines.get(impl_name, {})
        impl_report = {"tests": 0, "passed": 0, "failed": 0, "details": []}

        for result in test_results:
            config = result.get("config", "unknown")
            current_time = result.get("mean_time_ms", 0)
            baseline_time = impl_baselines.get(config, 0)

            if baseline_time == 0:
                report["summary"]["no_baseline"] += 1
                status = "no_baseline"
                regression_pct = 0
                passed = True
            else:
                passed, regression_pct = compare_performance(
                    current_time, baseline_time, threshold
                )
                status = "passed" if passed else "failed"

                if not passed:
                    report["regressions"].append(
                        {
                            "implementation": impl_name,
                            "config": config,
                            "baseline_ms": baseline_time,
                            "current_ms": current_time,
                            "regression_pct": regression_pct,
                        }
                    )

            impl_report["tests"] += 1
            if passed:
                impl_report["passed"] += 1
            else:
                impl_report["failed"] += 1

            impl_report["details"].append(
                {
                    "config": config,
                    "status": status,
                    "baseline_ms": baseline_time,
                    "current_ms": current_time,
                    "regression_pct": regression_pct,
                }
            )

        report["implementations"][impl_name] = impl_report
        report["summary"]["total_tests"] += impl_report["tests"]
        report["summary"]["passed"] += impl_report["passed"]
        report["summary"]["failed"] += impl_report["failed"]

    return report


def print_report(report: dict[str, Any]):
    """Print human-readable report."""
    summary = report["summary"]

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 60)

    print("\nSummary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(
        f"  Passed: {summary['passed']} ({summary['passed'] / summary['total_tests'] * 100:.1f}%)"
    )
    print(
        f"  Failed: {summary['failed']} ({summary['failed'] / summary['total_tests'] * 100:.1f}%)"
    )
    print(f"  No Baseline: {summary['no_baseline']}")

    if report["regressions"]:
        print("\n⚠️  Performance Regressions Detected:")
        print("-" * 60)
        for reg in report["regressions"]:
            print(f"  {reg['implementation']} - {reg['config']}:")
            print(f"    Baseline: {reg['baseline_ms']:.2f}ms")
            print(f"    Current:  {reg['current_ms']:.2f}ms")
            print(f"    Change:   {reg['regression_pct']:+.1f}%")

    print("\nDetailed Results by Implementation:")
    print("-" * 60)

    for impl_name, impl_data in report["implementations"].items():
        print(f"\n{impl_name}:")
        print(
            f"  Tests: {impl_data['tests']} | Passed: {impl_data['passed']} | Failed: {impl_data['failed']}"
        )

        for detail in impl_data["details"]:
            status_icon = {"passed": "✅", "failed": "❌", "no_baseline": "❓"}.get(
                detail["status"], "?"
            )

            print(f"    {status_icon} {detail['config']}: ", end="")

            if detail["status"] == "no_baseline":
                print(f"{detail['current_ms']:.2f}ms (no baseline)")
            else:
                print(
                    f"{detail['current_ms']:.2f}ms ({detail['regression_pct']:+.1f}%)"
                )


def main():
    parser = argparse.ArgumentParser(description="Compare performance with baseline")
    parser.add_argument("results", help="Path to test results JSON file")
    parser.add_argument(
        "--baseline",
        default="tests/performance_baselines/baselines.json",
        help="Path to baseline JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Regression threshold percentage (default: 15%)",
    )
    parser.add_argument("--output", help="Output JSON report path")

    args = parser.parse_args()

    # Load data
    baseline_path = Path(args.baseline)
    results_path = Path(args.results)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    baselines = load_baseline(baseline_path)
    results = load_results(results_path)

    # Generate report
    report = generate_report(baselines, results, args.threshold)

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

    # Print report
    print_report(report)

    # Exit with error if regressions found
    if report["summary"]["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
