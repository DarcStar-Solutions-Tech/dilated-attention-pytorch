#!/usr/bin/env python3
"""
Script to run performance regression tests and generate reports.

Usage:
    # Run all performance tests
    python scripts/run_performance_regression.py

    # Update baselines (after major changes)
    python scripts/run_performance_regression.py --update-baselines

    # Generate report only
    python scripts/run_performance_regression.py --report-only

    # Run specific implementation
    python scripts/run_performance_regression.py --implementation DilatedAttention
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_performance_regression import generate_performance_report


def run_performance_tests(implementation: str = None, update_baselines: bool = False):
    """Run performance regression tests."""
    cmd = ["pytest", "tests/test_performance_regression.py", "-v", "-s"]

    if implementation:
        cmd.extend(["-k", implementation])

    if update_baselines:
        # This would need special handling in the tests
        print("Updating baselines is not yet implemented.")
        print("Manually delete tests/performance_baselines/baselines.json to reset.")
        return

    # Add performance marker
    cmd.extend(["-m", "performance"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root, check=False)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run performance regression tests")
    parser.add_argument(
        "--implementation",
        help="Run tests for specific implementation only",
        choices=[
            "DilatedAttention",
            "ImprovedDilatedAttention",
            "MultiheadDilatedAttention",
            "ImprovedMultiheadDilatedAttention",
            "RingDilatedAttention",
            "BlockSparseRingDilatedAttention",
        ],
    )
    parser.add_argument(
        "--update-baselines", action="store_true", help="Update performance baselines"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Generate report without running tests"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Regression threshold percentage (default: 15%%)",
    )

    args = parser.parse_args()

    if args.report_only:
        print("Generating performance report...")
        generate_performance_report()
        return 0

    # Run tests
    return_code = run_performance_tests(
        implementation=args.implementation, update_baselines=args.update_baselines
    )

    # Generate report after tests
    if return_code == 0:
        print("\n" + "=" * 80)
        print("Generating performance report...")
        generate_performance_report()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
