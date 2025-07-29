#!/usr/bin/env python3
"""
Validate that Triton kernels compile correctly.

This script checks for common formatting issues that break Triton compilation.
"""

import re
import sys
from pathlib import Path


def check_multiline_pointer_arithmetic(file_path: Path) -> list[str]:
    """Check for problematic multiline pointer arithmetic."""
    errors = []
    content = file_path.read_text()

    # Pattern to find problematic multiline pointer arithmetic
    # Looking for patterns like:
    #   variable_ptrs = (
    #       Variable
    #       + something
    patterns = [
        r"(\w+_ptrs\s*=\s*\(\s*\n\s*)([A-Z]\w*)\s*\n\s*\+",
        r"(\w+_ptrs\s*=\s*\(\s*\n\s*)([A-Z]\w*)\s*$",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[: match.start()].count("\n") + 1
            errors.append(
                f"{file_path}:{line_num}: Multiline pointer arithmetic detected. "
                f"Variable '{match.group(2)}' should be on same line as first operation."
            )

    # Also check for orphaned fmt comments
    fmt_pattern = r"^\s*#\s*fmt:\s*(on|off)\s*$"
    for match in re.finditer(fmt_pattern, content, re.MULTILINE):
        line_num = content[: match.start()].count("\n") + 1
        errors.append(
            f"{file_path}:{line_num}: Orphaned fmt comment found. "
            "These can cause Triton compilation errors."
        )

    return errors


def main():
    """Main validation function."""
    # Find all kernel files
    kernel_dir = (
        Path(__file__).parent.parent / "src" / "dilated_attention_pytorch" / "kernels"
    )

    triton_files = [
        "hilbert_attention_core.py",
        "hilbert_attention_unified.py",
        "hilbert_attention_unified_optimized.py",
        "hilbert_attention_unified_optimized_enhanced.py",
    ]

    all_errors = []

    for filename in triton_files:
        file_path = kernel_dir / filename
        if file_path.exists():
            errors = check_multiline_pointer_arithmetic(file_path)
            all_errors.extend(errors)

    if all_errors:
        print("Triton kernel validation failed!")
        print("\nErrors found:")
        for error in all_errors:
            print(f"  {error}")
        print(f"\nTotal errors: {len(all_errors)}")
        return 1
    else:
        print("All Triton kernels passed validation!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
