#!/usr/bin/env python3
"""
Establish performance baselines for all implementations.

This script runs all performance tests to create initial baselines.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def establish_baselines():
    """Run all performance tests to establish baselines."""
    implementations = [
        "test_dilated_attention_performance",
        "test_improved_dilated_attention_performance",
        "test_multihead_dilated_attention_performance",
    ]

    print("Establishing performance baselines for all implementations...")
    print("=" * 80)

    for impl_test in implementations:
        print(f"\nRunning {impl_test}...")
        cmd = [
            "pytest",
            f"tests/test_performance_regression.py::TestPerformanceRegression::{impl_test}",
            "-v",
            "-s",
        ]

        result = subprocess.run(cmd, cwd=project_root, check=False)
        if result.returncode != 0 and result.returncode != 5:  # 5 = no tests collected
            print(f"Error running {impl_test}")
            return 1

    print("\n" + "=" * 80)
    print("Baselines established successfully!")
    print("\nBaseline files created in: tests/performance_baselines/")
    print("- baselines.json: Current performance baselines")
    print("- history.json: Performance history tracking")

    return 0


def main():
    # Use hatch environment
    return subprocess.run(
        ["hatch", "run", "python", __file__], cwd=project_root, check=False
    ).returncode


if __name__ == "__main__":
    # Check if running in hatch environment
    import os

    if "HATCH_ENV_ACTIVE" in os.environ:
        # We're inside hatch, run directly
        sys.exit(establish_baselines())
    else:
        # Run through hatch
        sys.exit(main())
