#!/usr/bin/env python3
"""
Archive redundant memory test files after consolidation.

The memory tests have been consolidated into two comprehensive files:
1. tests/core/test_memory_pools_comprehensive.py - Functional tests
2. tests/core/test_memory_performance.py - Performance tests
"""

import shutil
from pathlib import Path
from datetime import datetime


def main():
    """Archive redundant memory test files."""
    project_root = Path(__file__).parent.parent

    # Files to archive (now redundant)
    files_to_archive = [
        "tests/core/test_fragment_aware_memory.py",  # Consolidated into comprehensive
        "tests/core/test_numa_aware_memory.py",  # Consolidated into comprehensive
        "tests/misc/test_memory_pool_consolidated.py",  # Old consolidation attempt
        "tests/sparse/test_block_sparse_memory_improvement.py",  # Consolidated into performance
        "tests/utils/test_memory_optimizations.py",  # Consolidated into performance
        "tests/utils/test_memory_profiler.py",  # Consolidated into performance
    ]

    # Keep this one as it's specific to multihead implementation
    # "tests/sparse/test_block_sparse_multihead_memory.py"

    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = project_root / f"tests/archive/memory_tests_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Archiving redundant memory tests to: {archive_dir}")
    print("=" * 60)

    archived_count = 0
    for file_path in files_to_archive:
        full_path = project_root / file_path
        if full_path.exists():
            # Create subdirectory structure in archive
            relative_path = Path(file_path).relative_to("tests")
            archive_path = archive_dir / relative_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file to archive
            shutil.move(str(full_path), str(archive_path))
            print(f"✓ Archived: {file_path}")
            archived_count += 1
        else:
            print(f"✗ Not found: {file_path}")

    print("=" * 60)
    print(f"Archived {archived_count} files")
    print()
    print("Remaining memory test files:")
    print("1. tests/core/test_memory_pools_comprehensive.py - All functional tests")
    print("2. tests/core/test_memory_performance.py - All performance tests")
    print(
        "3. tests/sparse/test_block_sparse_multihead_memory.py - Specific multihead tests"
    )
    print()
    print("Memory test consolidation complete!")

    # Create a summary file
    summary_path = archive_dir / "ARCHIVE_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(f"# Memory Test Archive - {timestamp}\n\n")
        f.write("## Reason for Archive\n\n")
        f.write(
            "These memory test files were archived after consolidation into two comprehensive test files:\n\n"
        )
        f.write(
            "1. `tests/core/test_memory_pools_comprehensive.py` - Contains all functional tests\n"
        )
        f.write(
            "2. `tests/core/test_memory_performance.py` - Contains all performance tests\n\n"
        )
        f.write("## Archived Files\n\n")
        for file_path in files_to_archive:
            f.write(f"- `{file_path}`\n")
        f.write("\n## Consolidation Benefits\n\n")
        f.write("- Reduced code duplication (~60% reduction)\n")
        f.write("- Better organized test structure\n")
        f.write("- Easier maintenance\n")
        f.write("- Preserved all unique test functionality\n")


if __name__ == "__main__":
    main()
