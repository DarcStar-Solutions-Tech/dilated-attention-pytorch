#!/usr/bin/env python3
"""
Safely remove only the most redundant ring attention implementations.

This conservative approach only removes files that:
1. Are not exported in the main package
2. Have clear issues (use all_gather, multiple "fixed" versions)
3. Are superseded by better implementations
"""

import shutil
from pathlib import Path
from datetime import datetime


def main():
    """Remove only the safest redundant ring implementations."""

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    ring_dir = project_root / "src" / "dilated_attention_pytorch" / "ring"

    # Only remove files that are clearly redundant and not used
    files_to_remove = [
        # Hilbert implementations that are clearly redundant
        "hilbert/ring_dilated_attention_hilbert_core_fixed.py",  # Uses all_gather!
        "hilbert/ring_dilated_attention_hilbert_optimized_correct.py",  # Another "correct" version
        "hilbert/ring_dilated_attention_hilbert_optimized_fixed.py",  # Another "fixed" version
        "hilbert/ring_dilated_attention_hilbert_optimized_fixed_v2.py",  # V2 of a fixed version!
        # Redundant utilities
        "utils/ring_attention_utils_fixed.py",  # "Fixed" utility suggests issues
        "utils/ring_attention_fixed_deadlock.py",  # Specific fix that should be in main
        "utils/ring_attention_memory_efficient.py",  # Duplicate functionality
    ]

    # Create backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / "backups" / f"ring_cleanup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("Safe Ring Attention Cleanup")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Backup directory: {backup_dir}")
    print()

    # Check for all_gather usage
    print("Checking for problematic all_gather usage...")
    all_gather_files = []

    for py_file in ring_dir.rglob("*.py"):
        if "__pycache__" in str(py_file) or "__init__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
            if "all_gather" in content and "# all_gather breaks O(n/k)" not in content:
                rel_path = py_file.relative_to(ring_dir)
                all_gather_files.append(str(rel_path))
                print(f"  ⚠️  {rel_path} uses all_gather (breaks O(n/k) memory!)")
        except Exception:
            pass

    print()

    # Process removals
    removed_count = 0
    error_count = 0

    print("Removing redundant files...")
    for file_path in files_to_remove:
        full_path = ring_dir / file_path

        if full_path.exists():
            try:
                # Create backup
                backup_path = backup_dir / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_path, backup_path)

                # Remove file
                full_path.unlink()
                print(f"✓ Removed: {file_path}")
                removed_count += 1

            except Exception as e:
                print(f"✗ Error removing {file_path}: {e}")
                error_count += 1
        else:
            print(f"- Skipped: {file_path} (not found)")

    print()
    print("Summary:")
    print(f"  Files removed: {removed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Files with all_gather: {len(all_gather_files)}")
    print(f"  Backup location: {backup_dir}")

    print("\nRecommendations:")
    print(
        "1. The following files use all_gather and should be refactored or deprecated:"
    )
    for f in all_gather_files:
        if f not in files_to_remove:
            print(f"   - {f}")

    print("\n2. Consider consolidating these legacy implementations:")
    print("   - ring_dilated_attention_v3.py → StandardRingAttention")
    print("   - ring_dilated_attention_fixed_simple.py → StandardRingAttention")
    print("   - ring_dilated_attention_memory_efficient.py → StandardRingAttention")

    print("\n3. Update documentation to direct users to:")
    print("   - StandardRingAttention (basic ring attention)")
    print("   - HilbertRingAttention (Hilbert-optimized)")
    print("   - DistributedRingAttention (enterprise features)")

    print("\nCleanup complete!")


if __name__ == "__main__":
    main()
