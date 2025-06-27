#!/usr/bin/env python3
"""Smart pre-commit bypass for when you really need to commit."""

import subprocess
import sys


def main():
    # First, try to auto-fix what we can
    print("🔧 Running auto-fixes...")
    subprocess.run(["ruff", "check", ".", "--fix"], capture_output=True)
    subprocess.run(["ruff", "format", "."], capture_output=True)

    # Check remaining issues
    result = subprocess.run(
        ["ruff", "check", ".", "--exit-zero"], capture_output=True, text=True
    )

    if result.stdout:
        print("⚠️  Remaining linting issues:")
        print(result.stdout)

        response = input("\nCommit anyway? [y/N]: ")
        if response.lower() != "y":
            print("❌ Commit cancelled")
            sys.exit(1)

    # Commit with no-verify
    print("✅ Committing with --no-verify...")
    subprocess.run(["git", "commit", "--no-verify"] + sys.argv[1:])


if __name__ == "__main__":
    main()
