#!/usr/bin/env python3
"""
Fix common import syntax errors that break the code.

This script specifically targets the following patterns:
1. import torch.distributed as dist  # noqa: PLC0415
2. from flash_attn import flash_attn_func  # noqa: PLC0415

And fixes them to:
1. import torch.distributed as dist  # noqa: PLC0415
2. from flash_attn import flash_attn_func  # noqa: PLC0415
"""

import re
from pathlib import Path


def fix_import_syntax(file_path: Path) -> bool:
    """Fix import syntax in a single file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Pattern 1: Fix "import X # noqa as Y" -> "import X as Y # noqa"
        pattern1 = r"import\s+(\S+)\s+#\s*noqa:\s*\w+\s+as\s+(\w+)"
        replacement1 = r"import \1 as \2  # noqa: PLC0415"
        content = re.sub(pattern1, replacement1, content)

        # Pattern 2: Fix "from X # noqa import Y" -> "from X import Y # noqa"
        pattern2 = r"from\s+(\S+)\s+#\s*noqa:\s*\w+\s+import\s+(.+?)(?=\n|$)"
        replacement2 = r"from \1 import \2  # noqa: PLC0415"
        content = re.sub(pattern2, replacement2, content)

        # Pattern 3: Fix malformed noqa comments (e.g., "# noqa: PLC0415")
        pattern3 = r"#\s*noqa:\s*PLC0415_\w+"
        replacement3 = r"# noqa: PLC0415"
        content = re.sub(pattern3, replacement3, content)

        if content != original_content:
            file_path.write_text(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix import syntax in all Python files."""
    root_dir = Path(__file__).parent.parent
    fixed_files = []

    # Find all Python files
    python_files = list(root_dir.rglob("*.py"))

    for file_path in python_files:
        # Skip virtual environments and cache directories
        if any(
            part in file_path.parts for part in [".venv", "venv", "__pycache__", ".git"]
        ):
            continue

        if fix_import_syntax(file_path):
            fixed_files.append(file_path)

    if fixed_files:
        print(f"✅ Fixed import syntax in {len(fixed_files)} files:")
        for f in fixed_files:
            print(f"  - {f.relative_to(root_dir)}")
    else:
        print("✅ No import syntax issues found!")

    return 0


if __name__ == "__main__":
    exit(main())
