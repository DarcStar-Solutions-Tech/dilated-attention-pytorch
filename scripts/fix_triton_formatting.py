#!/usr/bin/env python3
"""
Fix Triton kernel formatting issues automatically.
"""

import re
import sys
from pathlib import Path


def fix_multiline_pointer_arithmetic(content: str) -> str:
    """Fix multiline pointer arithmetic to be Triton-compatible."""

    # Pattern 1: Fix cases where variable is on separate line
    # Example:
    #   q_ptrs = (
    #       Q
    #       + pid_b * stride_qb
    # Becomes:
    #   q_ptrs = (
    #       Q + pid_b * stride_qb
    pattern1 = r"(\w+_ptrs\s*=\s*\(\s*\n\s*)([A-Z]\w*)\s*\n\s*\+"
    replacement1 = r"\1\2 +"
    content = re.sub(pattern1, replacement1, content)

    # Pattern 2: Fix cases with just variable on line
    pattern2 = r"(\w+_ptrs\s*=\s*\(\s*\n\s*)([A-Z]\w*)\s*$"

    def replace_pattern2(match):
        # Find the next non-empty line
        start_pos = match.end()
        lines_after = content[start_pos:].split("\n")
        for i, line in enumerate(lines_after):
            if line.strip():
                if line.strip().startswith("+"):
                    # Merge with next line
                    return match.group(1) + match.group(2) + " "
                break
        return match.group(0)

    content = re.sub(pattern2, replace_pattern2, content, flags=re.MULTILINE)

    # Remove orphaned fmt comments
    content = re.sub(r"^\s*#\s*fmt:\s*off\s*\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*#\s*fmt:\s*on\s*\n", "", content, flags=re.MULTILINE)

    return content


def process_file(file_path: Path) -> bool:
    """Process a single file and return True if changes were made."""
    print(f"Processing {file_path.name}...")

    try:
        original_content = file_path.read_text()
        fixed_content = fix_multiline_pointer_arithmetic(original_content)

        if original_content != fixed_content:
            file_path.write_text(fixed_content)
            print(f"  ✓ Fixed formatting issues in {file_path.name}")
            return True
        else:
            print(f"  - No changes needed in {file_path.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
        return False


def main():
    """Main function."""
    kernel_dir = (
        Path(__file__).parent.parent / "src" / "dilated_attention_pytorch" / "kernels"
    )

    triton_files = [
        "hilbert_attention_core.py",
        "hilbert_attention_unified.py",
        "hilbert_attention_unified_optimized.py",
        "hilbert_attention_unified_optimized_enhanced.py",
    ]

    changes_made = False

    for filename in triton_files:
        file_path = kernel_dir / filename
        if file_path.exists():
            if process_file(file_path):
                changes_made = True

    if changes_made:
        print("\n✓ Formatting fixes applied successfully!")
        print("Run 'python scripts/validate_triton_kernels.py' to verify.")
    else:
        print("\n- No formatting changes were needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
