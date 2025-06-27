#!/usr/bin/env python3
"""
Archive obsolete documentation files.

This script moves completed/historical documentation to the archive folder.
"""

import shutil
from datetime import datetime
from pathlib import Path


def archive_files():
    """Archive obsolete documentation files."""
    # Files to archive - organized by category
    files_to_archive = {
        "completed_refactoring": [
            "docs/DOCUMENTATION_RENAME_SUMMARY.md",
            "docs/OBSOLETE_DOCS_CLEANUP.md",
            "docs/core-testing-summary.md",
            "docs/defect-report.md",
            "docs/advanced-distributed-summary.md",
        ],
        "completed_reports": [
            "docs/reports/REFACTORING_FINAL_SUMMARY.md",
            "docs/reports/TEST_FIXES_SUMMARY.md",
            "docs/reports/INDEX_SELECT_OPTIMIZATION_SUMMARY.md",
            "docs/reports/UNFOLD_OPTIMIZATION_SUMMARY.md",
            "docs/reports/DOCUMENTATION_UPDATE_SUMMARY.md",
            "docs/reports/BENCHMARK_UPDATE_SUMMARY.md",
            "docs/reports/comprehensive_defect_report.md",
            "docs/reports/phase1-progress-2025-06-26-1542-UTC.md",
            "docs/reports/BLOCK_SPARSE_OPTIMIZATION_REPORT.md",
            "docs/reports/IMPLEMENTATION_COMPARISON_REPORT.md",
        ],
        "historical_analysis": [
            "docs/reports/defect-analysis-2025-06-26-1456-UTC.md",
            "docs/reports/ring_attention_implementation_analysis.md",
            "docs/reports/ring_vs_block_sparse_comparison.md",
            "docs/reports/maximum_chunk_analysis_results.md",
        ],
        "timestamped_feasibility": [
            "docs/feasibility/1t-parameter-training-feasibility-2025-06-26-1136-UTC.md",
            "docs/feasibility/1t-parameter-training-feasibility-block-sparse-2025-06-26-1136-UTC.md",
        ],
    }

    # Create archive directory
    archive_dir = Path("docs/archive")
    archive_dir.mkdir(exist_ok=True)

    # Create subdirectories for better organization
    subdirs = {
        "refactoring": archive_dir / "refactoring",
        "reports": archive_dir / "reports",
        "analysis": archive_dir / "analysis",
        "feasibility": archive_dir / "feasibility",
    }

    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)

    # Track what we archive
    archived_files = []
    not_found_files = []

    # Archive files by category
    category_mapping = {
        "completed_refactoring": "refactoring",
        "completed_reports": "reports",
        "historical_analysis": "analysis",
        "timestamped_feasibility": "feasibility",
    }

    for category, files in files_to_archive.items():
        target_subdir = subdirs[category_mapping[category]]

        for file_path in files:
            source = Path(file_path)
            if source.exists():
                target = target_subdir / source.name
                shutil.move(str(source), str(target))
                archived_files.append((str(source), str(target)))
                print(f"✓ Archived: {file_path} → {target}")
            else:
                not_found_files.append(file_path)

    # Generate archive summary
    summary_path = (
        archive_dir
        / f"archive-summary-{datetime.utcnow().strftime('%Y-%m-%d-%H%M-UTC')}.md"
    )
    with open(summary_path, "w") as f:
        f.write("# Documentation Archive Summary\n\n")
        f.write(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("## Purpose\n\n")
        f.write("Archived obsolete documentation files that describe completed work, ")
        f.write("historical analyses, and fixed defects. These files are preserved ")
        f.write(
            "for historical reference but are no longer part of active documentation.\n\n"
        )
        f.write("## Archived Files\n\n")

        for category, target_dir in category_mapping.items():
            files_in_category = [f for f in archived_files if target_dir in f[1]]
            if files_in_category:
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for source, target in files_in_category:
                    f.write(f"- `{source}` → `{target}`\n")
                f.write("\n")

        if not_found_files:
            f.write("## Files Not Found\n\n")
            for file_path in not_found_files:
                f.write(f"- `{file_path}`\n")

    print(f"\n✅ Archived {len(archived_files)} files")
    if not_found_files:
        print(f"⚠️  {len(not_found_files)} files not found")
    print(f"📄 Archive summary: {summary_path}")


if __name__ == "__main__":
    archive_files()
