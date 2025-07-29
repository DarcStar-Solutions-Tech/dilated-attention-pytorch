#!/usr/bin/env python3
"""
Remove redundant Hilbert kernel implementations based on performance analysis.
"""

import shutil
from pathlib import Path
from datetime import datetime

# Kernels to remove based on performance analysis
KERNELS_TO_REMOVE = [
    "hilbert_attention_simple.py",  # Redundant with unified_optimized_enhanced
    "hilbert_attention_enhanced.py",  # Features merged into unified_optimized_enhanced
    "hilbert_attention.py",  # Basic implementation superseded
    "hilbert_attention_core.py",  # Slow, features available in other implementations
]

# Kernels to keep
KERNELS_TO_KEEP = [
    "hilbert_attention_unified_optimized_enhanced.py",  # Best overall performance
    "hilbert_attention_unified_optimized.py",  # Best for very large sequences
    "hilbert_attention_unified.py",  # Stable baseline
]


def backup_file(file_path: Path, backup_dir: Path):
    """Create a backup of the file before removal."""
    backup_path = backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    print(f"  Backed up to: {backup_path}")


def main():
    # Get kernel directory
    kernel_dir = (
        Path(__file__).parent.parent / "src" / "dilated_attention_pytorch" / "kernels"
    )

    # Create backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(__file__).parent.parent / f"kernel_backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)

    print("=== Hilbert Kernel Cleanup ===")
    print(f"Backup directory: {backup_dir}")
    print()

    # Check current state
    print("Current kernel files:")
    kernel_files = sorted(kernel_dir.glob("hilbert_*.py"))
    for f in kernel_files:
        status = "REMOVE" if f.name in KERNELS_TO_REMOVE else "KEEP"
        print(f"  {f.name:<45} [{status}]")

    print(f"\nTotal files: {len(kernel_files)}")
    print(f"To remove: {len([f for f in kernel_files if f.name in KERNELS_TO_REMOVE])}")
    print(f"To keep: {len([f for f in kernel_files if f.name in KERNELS_TO_KEEP])}")

    # Confirm with user
    response = input("\nProceed with removal? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    # Remove files
    print("\nRemoving redundant kernels...")
    removed_count = 0

    for kernel_name in KERNELS_TO_REMOVE:
        kernel_path = kernel_dir / kernel_name
        if kernel_path.exists():
            print(f"\nRemoving {kernel_name}...")
            # Backup first
            backup_file(kernel_path, backup_dir)
            # Remove
            kernel_path.unlink()
            print("  ✓ Removed")
            removed_count += 1
        else:
            print(f"\n{kernel_name} not found - skipping")

    # Create summary file
    summary_path = backup_dir / "removal_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Hilbert Kernel Cleanup Summary\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("=================================\n\n")

        f.write("Files Removed:\n")
        for kernel in KERNELS_TO_REMOVE:
            f.write(f"  - {kernel}\n")

        f.write("\nFiles Kept:\n")
        for kernel in KERNELS_TO_KEEP:
            f.write(f"  - {kernel}\n")

        f.write(f"\nTotal removed: {removed_count}\n")

        f.write("\nRationale:\n")
        f.write(
            "- hilbert_attention_simple.py: Redundant with unified_optimized_enhanced\n"
        )
        f.write(
            "- hilbert_attention_enhanced.py: Features merged into unified_optimized_enhanced\n"
        )
        f.write("- hilbert_attention.py: Basic implementation superseded\n")
        f.write(
            "- hilbert_attention_core.py: Poor performance, features available elsewhere\n"
        )

    print(f"\n✓ Removed {removed_count} redundant kernel files")
    print(f"✓ Backups saved to: {backup_dir}")
    print(f"✓ Summary written to: {summary_path}")

    # Final state
    print("\nRemaining kernel files:")
    remaining = sorted(kernel_dir.glob("hilbert_*.py"))
    for f in remaining:
        print(f"  ✓ {f.name}")

    print("\nKernel consolidation complete!")
    print(f"Reduced from {len(kernel_files)} to {len(remaining)} implementations")


if __name__ == "__main__":
    main()
