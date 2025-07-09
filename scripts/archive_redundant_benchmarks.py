#!/usr/bin/env python3
"""
Archive redundant benchmark files after consolidation.

The benchmarks have been consolidated into:
1. Core framework in benchmarks/core/
2. Consolidated suites in benchmarks/suites/consolidated/
"""

import shutil
from pathlib import Path
from datetime import datetime


def main():
    """Archive redundant benchmark files."""
    project_root = Path(__file__).parent.parent

    # Files to keep (new consolidated ones and core framework)
    files_to_keep = {
        # Core framework
        "benchmarks/core/",
        "benchmarks/run_benchmark.py",
        # New consolidated suites
        "benchmarks/suites/consolidated/",
        # Keep a few specialized ones that are unique
        "benchmarks/suites/specialized/benchmark_flash_attention_3.py",
        "benchmarks/suites/specialized/check_sdpa_backends.py",
        "benchmarks/suites/specialized/benchmark_liquid_cfc_routing.py",  # Unique routing test
    }

    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = project_root / f"benchmarks/archive/benchmark_files_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Archiving redundant benchmarks to: {archive_dir}")
    print("=" * 60)

    # Find all benchmark files
    benchmark_files = list((project_root / "benchmarks").rglob("*.py"))

    archived_count = 0
    kept_count = 0

    for file_path in benchmark_files:
        relative_path = file_path.relative_to(project_root)

        # Check if file should be kept
        should_keep = any(str(relative_path).startswith(keep) for keep in files_to_keep)

        if should_keep:
            kept_count += 1
            continue

        # Skip __pycache__ and archive directories
        if "__pycache__" in str(file_path) or "/archive/" in str(file_path):
            continue

        # Archive the file
        archive_path = archive_dir / relative_path.relative_to("benchmarks")
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(file_path), str(archive_path))
        print(f"✓ Archived: {relative_path}")
        archived_count += 1

    # Clean up empty directories
    for dirpath in sorted((project_root / "benchmarks").rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()
            print(f"✓ Removed empty dir: {dirpath.relative_to(project_root)}")

    print("=" * 60)
    print(f"Archived {archived_count} files")
    print(f"Kept {kept_count} files")
    print()
    print("Remaining benchmark structure:")
    print("- benchmarks/core/ - Framework and utilities")
    print("- benchmarks/suites/consolidated/ - Consolidated benchmark suites")
    print("- benchmarks/archive/ - Archived old files")
    print()
    print("Benchmark consolidation complete!")

    # Create a summary file
    summary_path = archive_dir / "ARCHIVE_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(f"# Benchmark Archive - {timestamp}\n\n")
        f.write("## Reason for Archive\n\n")
        f.write("These benchmark files were archived after consolidation into:\n\n")
        f.write("1. **Core Framework** (`benchmarks/core/`)\n")
        f.write("   - `base_benchmark.py` - Base classes\n")
        f.write("   - `config.py` - Configuration system\n")
        f.write("   - `unified_runner.py` - Unified benchmark runner\n")
        f.write("   - `utils/` - Shared utilities\n\n")
        f.write("2. **Consolidated Suites** (`benchmarks/suites/consolidated/`)\n")
        f.write("   - `benchmark_basic_comparison.py` - Basic performance comparison\n")
        f.write("   - `benchmark_extreme_sequences.py` - Extreme sequence lengths\n")
        f.write("   - `benchmark_block_sparse.py` - Block sparse variants\n")
        f.write("   - (more to come)\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Files archived: {archived_count}\n")
        f.write(f"- Files kept: {kept_count}\n")
        f.write(
            f"- Reduction: ~{archived_count / (archived_count + kept_count) * 100:.0f}%\n\n"
        )
        f.write("## Benefits\n\n")
        f.write("- Eliminated duplicate timing/memory/setup code\n")
        f.write("- Unified configuration system\n")
        f.write("- Consistent output formatting\n")
        f.write("- Easier to add new benchmarks\n")
        f.write("- Better maintainability\n")


if __name__ == "__main__":
    main()
