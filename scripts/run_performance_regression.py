#!/usr/bin/env python3
"""
Run performance regression tests and generate reports.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_tests():
    """Run performance regression tests."""
    print("Running performance regression tests...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_performance_regression_all.py",
        "-v",
        "--tb=short",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def generate_html_dashboard():
    """Generate HTML performance dashboard."""
    baseline_dir = Path("tests/performance_baselines")
    history_file = baseline_dir / "history_all.json"

    if not history_file.exists():
        print("No performance history found.")
        return

    with open(history_file) as f:
        history = json.load(f)

    # Group by implementation
    implementations = {}
    for entry in history:
        impl = entry["implementation"]
        if impl not in implementations:
            implementations[impl] = []
        implementations[impl].append(entry)

    # Generate HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .implementation { margin: 20px 0; }
        .pass { color: green; }
        .fail { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Performance Regression Dashboard</h1>
    <p>Generated: {timestamp}</p>
    
    <h2>Recent Results</h2>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for impl, entries in implementations.items():
        recent = entries[-20:]  # Last 20 entries

        html += f"""
    <div class="implementation">
        <h3>{impl}</h3>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Configuration</th>
                <th>Time (ms)</th>
                <th>Change</th>
                <th>Status</th>
            </tr>
"""

        for entry in recent:
            status_class = "pass" if entry["passed"] else "fail"
            status_icon = "✅" if entry["passed"] else "❌"

            html += f"""
            <tr>
                <td>{entry['timestamp'][:19]}</td>
                <td>{entry['config']}</td>
                <td>{entry['metrics']['execution_time_ms']:.2f}</td>
                <td>{entry['regression_pct']:+.1f}%</td>
                <td class="{status_class}">{status_icon}</td>
            </tr>
"""

        html += """
        </table>
    </div>
"""

    html += """
</body>
</html>
"""

    with open("performance_dashboard.html", "w") as f:
        f.write(html)

    print("Generated performance_dashboard.html")


def generate_comparison_report():
    """Generate performance comparison report."""
    baseline_file = Path("tests/performance_baselines/baselines_all.json")

    if not baseline_file.exists():
        print("No baselines found.")
        return

    with open(baseline_file) as f:
        baselines = json.load(f)

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comparison { display: flex; flex-wrap: wrap; gap: 20px; }
        .impl-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .metric { margin: 5px 0; }
        .faster { color: green; }
        .slower { color: red; }
    </style>
</head>
<body>
    <h1>Implementation Performance Comparison</h1>
    
    <div class="comparison">
"""

    # Compare each implementation
    base_impl = "DilatedAttention"

    for impl, configs in baselines.items():
        if impl == base_impl:
            continue

        html += f"""
        <div class="impl-card">
            <h3>{impl}</h3>
"""

        for config, metrics in configs.items():
            base_time = baselines.get(base_impl, {}).get(config, {}).get("execution_time_ms", 0)
            impl_time = metrics.get("execution_time_ms", 0)

            if base_time > 0 and impl_time > 0:
                speedup = (base_time / impl_time - 1) * 100
                speedup_class = "faster" if speedup > 0 else "slower"
                speedup_text = f"{speedup:+.1f}%" if speedup != 0 else "same"

                html += f"""
            <div class="metric">
                <strong>{config}:</strong> {impl_time:.2f}ms 
                <span class="{speedup_class}">({speedup_text} vs baseline)</span>
            </div>
"""

        html += """
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open("performance_comparison.html", "w") as f:
        f.write(html)

    print("Generated performance_comparison.html")


def main():
    """Main entry point."""
    success = run_tests()

    # Generate reports
    generate_html_dashboard()
    generate_comparison_report()

    if not success:
        print("\n❌ Performance regression tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All performance tests passed!")


if __name__ == "__main__":
    main()
