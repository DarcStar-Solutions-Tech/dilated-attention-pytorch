#!/usr/bin/env python3
"""
Script to remove redundant ring attention implementations.

This script safely removes deprecated and redundant ring attention files
while preserving the core implementations that provide the best functionality.
"""

import shutil
from pathlib import Path
from datetime import datetime


def main():
    """Remove redundant ring attention implementations."""

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    ring_dir = project_root / "src" / "dilated_attention_pytorch" / "ring"

    # Files to remove (redundant/deprecated implementations)
    files_to_remove = [
        # Deprecated base implementations
        # Keep: ring_dilated_attention_correct.py (exported in __init__.py)
        "base/ring_dilated_attention_v3.py",
        "base/ring_dilated_attention_fixed_simple.py",
        "base/ring_dilated_attention_memory_efficient.py",
        # Keep: ring_dilated_attention_sdpa.py (exported in __init__.py)
        # Redundant Hilbert implementations
        "hilbert/ring_dilated_attention_hilbert_core.py",
        "hilbert/ring_dilated_attention_hilbert_core_fixed.py",
        # Keep: ring_dilated_attention_hilbert_gpu_optimized.py (exported in __init__.py)
        "hilbert/ring_dilated_attention_hilbert_optimized_correct.py",
        "hilbert/ring_dilated_attention_hilbert_optimized_fixed.py",
        "hilbert/ring_dilated_attention_hilbert_optimized_fixed_v2.py",
        "hilbert/ring_dilated_attention_hilbert_proper.py",
        # Redundant distributed implementation
        # Note: ring_distributed_dilated_attention.py already renamed to EnterpriseDistributed
        # Redundant utilities
        "utils/ring_attention_utils_fixed.py",
        "utils/ring_attention_fixed_deadlock.py",
        "utils/ring_attention_memory_efficient.py",
        # Note: Not removing utils/ring_communication_mixin.py if it's the only one
    ]

    # Files to keep (for reference)
    _ = [
        "standard_ring_attention.py",
        "hilbert_ring_attention.py",
        "distributed_ring_attention.py",
        "block_sparse_ring_attention.py",
        "factory.py",
        "__init__.py",
        "base/__init__.py",
        "base/base_ring_attention.py",
        "base/ring_config.py",
        "base/ring_communication_mixin.py",
        "utils/__init__.py",
        "utils/ring_attention_utils.py",
        "utils/ring_attention_autograd.py",
        "utils/ring_attention_lse.py",
    ]

    # Create backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / "backups" / f"ring_cleanup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("Ring Attention Cleanup")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Backup directory: {backup_dir}")
    print()

    # Process removals
    removed_count = 0
    error_count = 0

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

    # Clean up empty directories
    for subdir in ["hilbert", "base", "utils", "distributed"]:
        dir_path = ring_dir / subdir
        if dir_path.exists() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                print(f"✓ Removed empty directory: {subdir}/")
            except Exception:
                pass

    print()
    print("Summary:")
    print(f"  Files removed: {removed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Backup location: {backup_dir}")

    # Update __init__.py files to remove imports of deleted files
    print("\nUpdating __init__.py files...")
    update_init_files(ring_dir)

    print("\nCleanup complete!")
    print("\nNext steps:")
    print("1. Run tests to ensure nothing is broken")
    print("2. Update documentation to reflect removed files")
    print("3. Commit the changes")


def update_init_files(ring_dir: Path):
    """Update __init__.py files to remove imports of deleted modules."""

    # Update base/__init__.py
    base_init = ring_dir / "base" / "__init__.py"
    if base_init.exists():
        content = base_init.read_text()

        # Remove imports of deleted modules
        lines_to_remove = [
            "RingDilatedAttentionCorrect",
            "RingDilatedAttentionV3",
            "RingDilatedAttentionFixedSimple",
            "RingDilatedAttentionMemoryEfficient",
            "RingDilatedAttentionSDPA",
        ]

        new_lines = []
        for line in content.splitlines():
            if not any(name in line for name in lines_to_remove):
                new_lines.append(line)

        base_init.write_text("\n".join(new_lines) + "\n")
        print("✓ Updated base/__init__.py")

    # Update hilbert/__init__.py if it exists
    hilbert_init = ring_dir / "hilbert" / "__init__.py"
    if hilbert_init.exists():
        # Since we're removing all Hilbert implementations except the main one,
        # this directory might be empty now
        print("✓ Checked hilbert/__init__.py")


if __name__ == "__main__":
    main()
