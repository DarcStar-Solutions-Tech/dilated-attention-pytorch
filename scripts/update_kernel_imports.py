#!/usr/bin/env python3
"""
Update imports after removing redundant kernel implementations.
"""

import re
from pathlib import Path

# Mapping of old imports to new ones (regex patterns)
IMPORT_PATTERNS = [
    # hilbert_attention_simple.py removals
    (
        r"from dilated_attention_pytorch\.kernels\.hilbert_attention_simple import (\w+)",
        "from dilated_attention_pytorch.kernels.hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    (
        r"from \.hilbert_attention_simple import (\w+)",
        "from .hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    (
        r"import dilated_attention_pytorch\.kernels\.hilbert_attention_simple",
        "import dilated_attention_pytorch.kernels.hilbert_attention_unified",
    ),
    # hilbert_attention_enhanced.py removals
    (
        r"from dilated_attention_pytorch\.kernels\.hilbert_attention_enhanced import (\w+)",
        "from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced import UnifiedHilbertAttentionOptimizedEnhanced",
    ),
    (
        r"from \.hilbert_attention_enhanced import (\w+)",
        "from .hilbert_attention_unified_optimized_enhanced import UnifiedHilbertAttentionOptimizedEnhanced",
    ),
    # hilbert_attention.py removals (be careful not to match other files)
    (
        r"from dilated_attention_pytorch\.kernels\.hilbert_attention import (\w+)",
        "from dilated_attention_pytorch.kernels.hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    (
        r"from \.hilbert_attention import (\w+)(?!_)",  # Negative lookahead to avoid matching hilbert_attention_*
        "from .hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    # hilbert_attention_core.py removals
    (
        r"from dilated_attention_pytorch\.kernels\.hilbert_attention_core import (\w+)",
        "from dilated_attention_pytorch.kernels.hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    (
        r"from \.hilbert_attention_core import (\w+)",
        "from .hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    (
        r"from \.kernels\.hilbert_attention_core import (\w+)",
        "from .kernels.hilbert_attention_unified import UnifiedHilbertAttention",
    ),
    # Handle specific class imports
    (
        r"HilbertAttentionCore(?!\w)",  # Not followed by word character
        "UnifiedHilbertAttention",
    ),
    (r"HilbertAttentionSimple", "UnifiedHilbertAttention"),
    (r"HilbertAttentionEnhanced", "UnifiedHilbertAttentionOptimizedEnhanced"),
    (
        r"(?<!\w)HilbertAttention(?!\w)",  # Not preceded or followed by word char
        "UnifiedHilbertAttention",
    ),
]

# Also handle relative imports
RELATIVE_MAPPINGS = {
    "from .hilbert_attention_simple": "from .hilbert_attention_unified",
    "from .hilbert_attention_enhanced": "from .hilbert_attention_unified_optimized_enhanced",
    "from .hilbert_attention import": "from .hilbert_attention_unified import",
    "from .hilbert_attention_core": "from .hilbert_attention_unified",
}


def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file. Returns True if changes were made."""
    try:
        content = file_path.read_text()
        original_content = content

        # Apply regex pattern replacements
        for pattern, replacement in IMPORT_PATTERNS:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                print(f"  Updated pattern '{pattern[:30]}...' in {file_path.name}")
                content = new_content

        # Apply relative import mappings
        for old_pattern, new_pattern in RELATIVE_MAPPINGS.items():
            if old_pattern in content:
                content = re.sub(
                    rf"{re.escape(old_pattern)}.*",
                    lambda m: m.group(0).replace(old_pattern, new_pattern),
                    content,
                )
                print(f"  Updated relative import in {file_path.name}")

        # Write back if changed
        if content != original_content:
            file_path.write_text(content)
            return True

        return False

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False


def main():
    print("=== Update Kernel Imports ===")

    # Get project root
    project_root = Path(__file__).parent.parent

    # Directories to search
    search_dirs = [
        project_root / "src",
        project_root / "tests",
        project_root / "benchmarks",
        project_root / "examples",
    ]

    total_updated = 0

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        print(f"\nSearching in {search_dir.relative_to(project_root)}...")

        # Find all Python files
        py_files = list(search_dir.rglob("*.py"))

        for py_file in py_files:
            # Skip the removed kernel files themselves
            if py_file.name in [
                "hilbert_attention_simple.py",
                "hilbert_attention_enhanced.py",
                "hilbert_attention.py",
                "hilbert_attention_core.py",
            ]:
                continue

            if update_imports_in_file(py_file):
                total_updated += 1

    print(f"\n✓ Updated imports in {total_updated} files")

    # Check for any remaining references
    print("\nChecking for remaining references to removed kernels...")

    removed_modules = [
        "hilbert_attention_simple",
        "hilbert_attention_enhanced",
        "hilbert_attention_core",
        "hilbert_attention",  # Be careful with this one as it's a substring of others
    ]

    remaining_refs = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for py_file in search_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                for module in removed_modules:
                    # Check for imports we might have missed
                    if f"kernels.{module}" in content or f"/{module}.py" in content:
                        # Avoid false positives
                        if module == "hilbert_attention" and any(
                            x in content
                            for x in [
                                "hilbert_attention_unified",
                                "hilbert_attention_core",
                            ]
                        ):
                            continue
                        remaining_refs.append((py_file, module))
            except:
                pass

    if remaining_refs:
        print("\n⚠ Found remaining references:")
        for file_path, module in remaining_refs:
            print(f"  {file_path.relative_to(project_root)}: {module}")
        print("\nPlease review these manually.")
    else:
        print("\n✓ No remaining references found")


if __name__ == "__main__":
    main()
