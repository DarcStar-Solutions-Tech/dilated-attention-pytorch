#!/usr/bin/env python3
"""
Preview which documentation files would be archived.
This is a dry-run version that shows what would happen without making changes.
"""

from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


def parse_timestamp(filename):
    """Extract timestamp from filename if present."""
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{4}-UTC)", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d-%H%M-UTC")
    return None


def get_file_type(filepath):
    """Categorize file based on path and name."""
    path_str = str(filepath)
    name = filepath.name

    if "benchmarks" in path_str:
        if "ring-attention" in name:
            return "ring-attention-benchmark"
        elif "benchmark-report" in name:
            return "benchmark-report"
        elif "performance-report" in name:
            return "performance-report"
        elif "trend-report" in name:
            return "trend-report"
        else:
            return "benchmark"
    elif "reports" in path_str:
        if "ring-attention" in name:
            return "ring-attention-report"
        elif "benchmark" in name:
            return "benchmark-report"
        elif "performance" in name:
            return "performance-report"
        else:
            return "report"
    elif "plans" in path_str:
        return "plan"
    elif "feasibility" in path_str:
        return "feasibility"
    else:
        return "other"


def should_archive(filepath, all_files):
    """Determine if a file should be archived."""
    name = filepath.name

    # Never archive non-timestamped files (permanent docs)
    if not parse_timestamp(name):
        return False, "Permanent documentation"

    # Never archive files in archive directory
    if "archive" in str(filepath):
        return False, "Already archived"

    # Never archive index or readme files
    if name.lower() in ["index.md", "readme.md", "latest_report.md"]:
        return False, "System file"

    # Check for specific files to archive

    # Empty or placeholder files
    if name in [
        "benchmark-report-2025-06-27-1555-UTC.md",
        "trend-report-2025-06-27-1555-UTC.md",
    ]:
        return True, "Empty placeholder file"

    # Superseded benchmarks
    if name == "ring-dilated-integration-benchmark-2025-06-27-2050-UTC.md":
        # Check if newer version exists
        newer = Path(str(filepath).replace("2050", "2130"))
        if newer.exists():
            return True, "Superseded by newer version"

    if name in [
        "performance-report-2025-06-27-2001-UTC.md",
        "ring-production-benchmark-2025-06-27-2138-UTC.md",
    ]:
        return True, "Superseded by comprehensive benchmarks"

    # Superseded benchmark images
    if name == "ring-attention-v2-benchmark-2025-06-27-1934-UTC.png":
        # Check if newer version exists (2001 version)
        newer = Path(str(filepath).replace("1934", "2001"))
        if newer.exists():
            return True, "Superseded by newer version"

    # Old reports that have been resolved or superseded
    reports_to_archive = [
        "benchmarking-improvements-2025-06-27-0620-UTC.md",
        "benchmark-system-implementation-2025-06-27-1722-UTC.md",
        "benchmark-tracking-status-2025-06-27-1716-UTC.md",
        "comprehensive-sequence-length-benchmarks-2025-06-27-1733-UTC.md",
        "extreme-sequence-benchmark-2025-06-27-0950-UTC.md",
        "fixed-ring-attention-benchmark-analysis-2025-06-27-1848-UTC.md",
        "performance-analysis-post-fa3-2025-06-27-1023-UTC.md",
        "performance-comparison-2025-06-27-1721-UTC.md",
        "performance-comparison-2025-06-27-1729-UTC.md",
        "ring-attention-analysis-2025-06-27-1742-UTC.md",
        "ring-attention-cleanup-summary-2025-06-27-1946-UTC.md",
        "ring-attention-complete-progress-2025-06-27-2128-UTC.md",
        "ring-attention-fixed-summary-2025-06-27-2125-UTC.md",
        "ring-attention-implementation-summary-2025-06-27-1915-UTC.md",
        "ring-attention-multi-gpu-analysis-2025-06-27-1856-UTC.md",
        "ring-attention-normalization-issue-2025-06-27-2116-UTC.md",
        "ring-attention-v2-benchmark-summary-2025-06-27-1935-UTC.md",
    ]

    if name in reports_to_archive:
        return True, "Old report - work completed or issue resolved"

    # Completed plans
    if "plans" in str(filepath):
        return True, "Completed implementation plan"

    # Check if this is an older version of a file type
    file_type = get_file_type(filepath)
    timestamp = parse_timestamp(name)

    if timestamp and file_type != "other":
        # Find all files of the same type
        same_type_files = []
        for f in all_files:
            if get_file_type(f) == file_type:
                ts = parse_timestamp(f.name)
                if ts:
                    same_type_files.append((f, ts))

        # Sort by timestamp
        same_type_files.sort(key=lambda x: x[1], reverse=True)

        # If there's a newer file of the same type, archive this one
        if len(same_type_files) > 1:
            newest = same_type_files[0][0]
            if filepath != newest:
                # Check if files are in same category (e.g., both about ring attention)
                base_name1 = re.sub(r"-\d{4}-\d{2}-\d{2}-\d{4}-UTC", "", name)
                base_name2 = re.sub(r"-\d{4}-\d{2}-\d{2}-\d{4}-UTC", "", newest.name)
                if base_name1 == base_name2:
                    return True, f"Older version (newest: {newest.name})"

    return False, "Current/latest version"


def main():
    """Preview files that would be archived."""
    docs_dir = Path("docs")

    # Collect all markdown and image files
    all_files = []
    for ext in ["*.md", "*.png", "*.csv"]:
        all_files.extend(docs_dir.rglob(ext))

    # Filter out archive directory
    all_files = [f for f in all_files if "archive" not in str(f)]

    # Categorize files
    to_archive = []
    to_keep = []

    for filepath in sorted(all_files):
        should_arch, reason = should_archive(filepath, all_files)
        if should_arch:
            to_archive.append((filepath, reason))
        else:
            to_keep.append((filepath, reason))

    # Display results
    print("=" * 80)
    print("DOCUMENTATION ARCHIVE PREVIEW")
    print("=" * 80)
    print(f"\nTotal files analyzed: {len(all_files)}")
    print(f"Files to archive: {len(to_archive)}")
    print(f"Files to keep: {len(to_keep)}")

    if to_archive:
        print("\n" + "=" * 80)
        print("FILES TO BE ARCHIVED")
        print("=" * 80)

        # Group by type
        by_type = defaultdict(list)
        for filepath, reason in to_archive:
            file_type = get_file_type(filepath)
            by_type[file_type].append((filepath, reason))

        for file_type, files in sorted(by_type.items()):
            print(f"\n{file_type.upper().replace('-', ' ')} ({len(files)} files):")
            print("-" * 40)
            for filepath, reason in sorted(files):
                try:
                    rel_path = filepath.relative_to(Path.cwd())
                except ValueError:
                    rel_path = filepath
                print(f"  {rel_path}")
                print(f"    → Reason: {reason}")

        # Calculate space
        total_size = sum(f.stat().st_size for f, _ in to_archive)
        print("\n" + "=" * 80)
        print(f"Total space to be freed: {total_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 80)
    print("FILES TO KEEP (showing first 10)")
    print("=" * 80)

    # Show sample of files to keep
    for filepath, reason in sorted(to_keep)[:10]:
        try:
            rel_path = filepath.relative_to(Path.cwd())
        except ValueError:
            rel_path = filepath
        print(f"  ✓ {rel_path} - {reason}")

    if len(to_keep) > 10:
        print(f"  ... and {len(to_keep) - 10} more files")

    print("\n" + "=" * 80)
    print("This is a PREVIEW. No files have been moved.")
    print("To actually archive files, run: python scripts/archive_old_docs.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
