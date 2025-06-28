#!/usr/bin/env python3
"""
Lenient pre-commit script that only fixes critical issues.

This script runs ruff with a more lenient configuration to avoid
constant pre-commit failures on minor issues.
"""

import subprocess
import sys
from pathlib import Path


def run_lenient_ruff():
    """Run ruff with lenient configuration."""
    root_dir = Path(__file__).parent.parent
    lenient_config = root_dir / ".ruff-lenient.toml"

    # Only check for critical issues
    critical_codes = [
        "E9",  # Syntax errors
        "F63",  # Invalid print statement
        "F7",  # Syntax errors
        "F82",  # Undefined names
        "W191",  # Indentation contains tabs
        "W291",  # Trailing whitespace
        "W292",  # No newline at end of file
        "W293",  # Blank line contains whitespace
        "W605",  # Invalid escape sequence
        "E999",  # Syntax error
    ]

    # Run ruff check
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--config",
        str(lenient_config),
        "--select",
        ",".join(critical_codes),
        "--fix",
        ".",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("🔍 Critical linting issues found:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False

    print("✅ No critical linting issues found!")
    return True


def run_format():
    """Run ruff format."""
    cmd = [sys.executable, "-m", "ruff", "format", "."]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Formatting failed:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False

    if "reformatted" in result.stdout:
        print("📝 Files were reformatted")
    else:
        print("✅ All files are properly formatted")

    return True


def main():
    """Main function."""
    print("🔍 Running lenient pre-commit checks...")

    # Run formatting first
    if not run_format():
        sys.exit(1)

    # Then check for critical issues
    if not run_lenient_ruff():
        sys.exit(1)

    print("✅ Pre-commit checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
