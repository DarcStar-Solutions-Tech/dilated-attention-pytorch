#!/usr/bin/env python
"""Main entry point for running benchmarks.

This script provides a unified interface for running benchmarks from different suites.
Results are saved to the data/benchmarks directory.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def discover_benchmarks(suite: Optional[str] = None) -> Dict[str, List[Path]]:
    """Discover available benchmark scripts in suites."""
    benchmarks_dir = Path(__file__).parent / "suites"
    suites = {}

    if suite:
        # Only look in specified suite
        suite_dirs = [benchmarks_dir / suite]
    else:
        # Look in all suites
        suite_dirs = [d for d in benchmarks_dir.iterdir() if d.is_dir()]

    for suite_dir in suite_dirs:
        if not suite_dir.exists():
            continue

        suite_name = suite_dir.name
        scripts = sorted(suite_dir.glob("*.py"))

        # Filter out __init__.py and other non-benchmark files
        scripts = [s for s in scripts if not s.name.startswith("__")]

        if scripts:
            suites[suite_name] = scripts

    return suites


def run_benchmark_script(script_path: Path, args: List[str] = None) -> Dict:
    """Run a single benchmark script."""
    print(f"\n{'=' * 60}")
    print(f"Running: {script_path.name}")
    print(f"Suite: {script_path.parent.name}")
    print(f"{'=' * 60}\n")

    # Import and run the benchmark
    spec = importlib.util.spec_from_file_location("benchmark", script_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)

        # Look for standard entry points
        if hasattr(module, "main"):
            result = module.main(args)
        elif hasattr(module, "run_benchmark"):
            result = module.run_benchmark(args)
        elif hasattr(module, "benchmark"):
            result = module.benchmark(args)
        else:
            print(f"Warning: No standard entry point found in {script_path.name}")
            result = {"status": "no_entry_point"}

        return {
            "script": str(script_path.relative_to(Path(__file__).parent)),
            "timestamp": datetime.utcnow().isoformat(),
            "result": result,
            "status": "success",
        }

    except Exception as e:
        print(f"Error running {script_path.name}: {e}")
        return {
            "script": str(script_path.relative_to(Path(__file__).parent)),
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "status": "failed",
        }


def save_results(results: List[Dict], output_dir: Path):
    """Save benchmark results to the data directory."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_file = output_dir / f"benchmark-run-{timestamp}.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({"run_timestamp": timestamp, "results": results}, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run dilated attention benchmarks")
    parser.add_argument(
        "--suite",
        choices=["basic", "extreme", "distributed", "specialized"],
        help="Run benchmarks from a specific suite",
    )
    parser.add_argument("--script", type=str, help="Run a specific benchmark script")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "benchmarks",
        help="Directory to save results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List benchmarks that would be run without executing",
    )

    args, unknown_args = parser.parse_known_args()

    # Discover available benchmarks
    suites = discover_benchmarks(args.suite)

    if args.list:
        print("Available benchmarks:\n")
        for suite_name, scripts in suites.items():
            print(f"{suite_name}:")
            for script in scripts:
                print(f"  - {script.name}")
        return

    # Determine which benchmarks to run
    if args.script:
        # Run specific script
        script_path = None
        for suite_scripts in suites.values():
            for script in suite_scripts:
                if script.name == args.script or str(script).endswith(args.script):
                    script_path = script
                    break
            if script_path:
                break

        if not script_path:
            print(f"Error: Script '{args.script}' not found")
            return 1

        scripts_to_run = [script_path]
    else:
        # Run all scripts in selected suite(s)
        scripts_to_run = []
        for scripts in suites.values():
            scripts_to_run.extend(scripts)

    if args.dry_run:
        print("Would run the following benchmarks:\n")
        for script in scripts_to_run:
            print(f"  - {script.parent.name}/{script.name}")
        return

    # Run benchmarks
    results = []
    for script in scripts_to_run:
        result = run_benchmark_script(script, unknown_args)
        results.append(result)

    # Save results
    save_results(results, args.output_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("Benchmark Summary:")
    print(f"{'=' * 60}")
    print(f"Total scripts run: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")


if __name__ == "__main__":
    sys.exit(main() or 0)
