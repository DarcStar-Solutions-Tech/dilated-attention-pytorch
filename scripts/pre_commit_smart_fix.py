#!/usr/bin/env python3
"""Smart fixes for common linting issues before commit."""

import re
import subprocess
import sys
from pathlib import Path


def fix_unused_variables():
    """Fix unused variable assignments."""
    # Get all F841 errors
    result = subprocess.run(
        ["ruff", "check", ".", "--select", "F841", "--output-format", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return

    try:
        import json

        errors = json.loads(result.stdout)
    except:
        # Fallback to text parsing
        return

    fixes = {}
    for error in errors:
        if error.get("code") == "F841":
            file_path = error["filename"]
            line = error["location"]["row"]
            var_name = (
                error.get("message", "").split("`")[1]
                if "`" in error.get("message", "")
                else None
            )

            if var_name and file_path not in fixes:
                fixes[file_path] = []
            if var_name:
                fixes[file_path].append((line, var_name))

    # Apply fixes
    for file_path, issues in fixes.items():
        try:
            lines = Path(file_path).read_text().splitlines()

            for line_num, var_name in sorted(issues, reverse=True):
                line_idx = line_num - 1
                if line_idx < len(lines):
                    # Replace variable with underscore
                    lines[line_idx] = re.sub(
                        rf"\b{var_name}\b\s*=", "_ =", lines[line_idx]
                    )

            Path(file_path).write_text("\n".join(lines) + "\n")
            print(f"Fixed unused variables in {file_path}")
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")


def fix_invalid_noqa():
    """Fix invalid noqa directives."""
    # Find all invalid noqa warnings
    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)

    invalid_noqa_files = []
    for line in result.stderr.splitlines():
        if "Invalid `# noqa` directive" in line:
            match = re.search(r"on ([^:]+):", line)
            if match:
                invalid_noqa_files.append(match.group(1))

    for file_path in set(invalid_noqa_files):
        try:
            content = Path(file_path).read_text()
            # Fix common invalid noqa patterns
            content = re.sub(r"# noqa: PLC0415", "# noqa: PLC0415", content)
            content = re.sub(r"# noqa: ([A-Z]+\d+)_[a-z]+", r"# noqa: \1", content)
            Path(file_path).write_text(content)
            print(f"Fixed invalid noqa in {file_path}")
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")


def main():
    """Run all smart fixes."""
    print("🔍 Running smart pre-commit fixes...")

    # Fix unused variables
    fix_unused_variables()

    # Fix invalid noqa directives
    fix_invalid_noqa()

    # Run ruff format
    subprocess.run(["ruff", "format", "."], capture_output=True)

    print("✅ Smart fixes complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
