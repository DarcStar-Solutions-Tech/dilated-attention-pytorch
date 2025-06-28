#!/usr/bin/env python3
"""
Archive old documentation files to clean up the docs directory.
This version runs automatically without prompting.
"""

import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import functions from preview script
from preview_archive import get_file_type, should_archive


def get_archive_path(filepath):
    """Determine the appropriate archive path for a file."""
    rel_path = filepath.relative_to(Path("docs"))

    # Determine subdirectory based on file type and location
    if "benchmarks" in str(rel_path):
        if "by-type" in str(rel_path) or "by-date" in str(rel_path):
            # Preserve existing structure
            archive_subdir = Path("benchmarks") / rel_path.parent.relative_to(
                Path("benchmarks")
            )
        else:
            # Direct benchmarks
            archive_subdir = Path("benchmarks/archived")
    elif "reports" in str(rel_path):
        archive_subdir = Path("reports/archived")
    elif "plans" in str(rel_path):
        archive_subdir = Path("plans/archived")
    else:
        # Other files
        archive_subdir = Path("other")

    archive_path = Path("docs/archive") / archive_subdir / filepath.name
    return archive_path


def archive_files(files_to_archive, dry_run=False):
    """Archive the specified files."""
    archived = []
    failed = []

    for filepath, reason in files_to_archive:
        try:
            archive_path = get_archive_path(filepath)

            if dry_run:
                print(f"Would move: {filepath} → {archive_path}")
            else:
                # Create archive directory if needed
                archive_path.parent.mkdir(parents=True, exist_ok=True)

                # Move the file
                shutil.move(str(filepath), str(archive_path))
                print(f"Archived: {filepath} → {archive_path}")

            archived.append((filepath, archive_path, reason))

        except Exception as e:
            print(f"ERROR archiving {filepath}: {e}")
            failed.append((filepath, str(e)))

    return archived, failed


def create_archive_summary(archived_files):
    """Create a summary of the archiving operation."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    summary_path = Path(f"docs/archive/archive-summary-{timestamp}.md")

    with open(summary_path, "w") as f:
        f.write("# Documentation Archive Summary\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total files archived: {len(archived_files)}\n")

        # Group by type
        by_type = defaultdict(list)
        for orig, arch, reason in archived_files:
            file_type = get_file_type(orig)
            by_type[file_type].append((orig, arch, reason))

        f.write(f"- File types: {', '.join(sorted(by_type.keys()))}\n\n")

        f.write("## Archived Files\n\n")

        for file_type, files in sorted(by_type.items()):
            f.write(
                f"### {file_type.title().replace('-', ' ')} ({len(files)} files)\n\n"
            )

            for orig, arch, reason in sorted(files):
                orig_rel = orig.relative_to(Path.cwd())
                arch_rel = arch.relative_to(Path.cwd())
                f.write(f"- `{orig_rel}`\n")
                f.write(f"  - Moved to: `{arch_rel}`\n")
                f.write(f"  - Reason: {reason}\n\n")

    return summary_path


def main():
    """Main archiving function."""
    docs_dir = Path("docs")

    # Collect all markdown and image files
    all_files = []
    for ext in ["*.md", "*.png", "*.csv"]:
        all_files.extend(docs_dir.rglob(ext))

    # Filter out archive directory
    all_files = [f for f in all_files if "archive" not in str(f)]

    # Determine which files to archive
    to_archive = []

    for filepath in sorted(all_files):
        should_arch, reason = should_archive(filepath, all_files)
        if should_arch:
            to_archive.append((filepath, reason))

    if not to_archive:
        print("No files need to be archived.")
        return

    # Display what will be archived
    print("=" * 80)
    print("DOCUMENTATION ARCHIVING")
    print("=" * 80)
    print(f"\nArchiving {len(to_archive)} files...")

    # Calculate space
    total_size = sum(f.stat().st_size for f, _ in to_archive)
    print(f"Total space to be freed: {total_size / 1024 / 1024:.2f} MB\n")

    # Archive the files
    archived, failed = archive_files(to_archive)

    if archived:
        print(f"\n✓ Successfully archived {len(archived)} files")

        # Create summary
        summary_path = create_archive_summary(archived)
        print(f"\n✓ Archive summary created: {summary_path}")

    if failed:
        print(f"\n✗ Failed to archive {len(failed)} files:")
        for filepath, error in failed:
            print(f"  - {filepath}: {error}")

    print("\n" + "=" * 80)
    print("Archiving complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
