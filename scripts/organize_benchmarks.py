#!/usr/bin/env python3
"""
Organize benchmark files into a structured directory layout.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import ClassVar


class BenchmarkOrganizer:
    """Organizes benchmark files into a structured layout."""

    # Patterns to identify benchmark types
    TYPE_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "comprehensive": re.compile(
            r"benchmark-(all-implementations|comprehensive)", re.I
        ),
        "long-sequences": re.compile(r"benchmark-long-sequences", re.I),
        "distributed": re.compile(r"benchmark-distributed", re.I),
        "regression": re.compile(r"(regression|performance-regression)", re.I),
        "memory": re.compile(r"(memory|memory-analysis)", re.I),
        "billion-token": re.compile(r"billion-token|1B-tokens", re.I),
        "comparison": re.compile(r"comparison|bugfix|phase\d+", re.I),
    }

    # Pattern to extract timestamp
    TIMESTAMP_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{4})-UTC"
    )

    def __init__(self, base_dir: Path):
        """Initialize organizer with base directory."""
        self.base_dir = Path(base_dir)
        self.by_date_dir = self.base_dir / "by-date"
        self.by_type_dir = self.base_dir / "by-type"
        self.latest_dir = self.base_dir / "latest"
        self.comparisons_dir = self.base_dir / "comparisons"
        self.archive_dir = self.base_dir / "archive"

    def setup_directories(self):
        """Create directory structure."""
        for dir_path in [
            self.by_date_dir,
            self.by_type_dir,
            self.latest_dir,
            self.comparisons_dir,
            self.archive_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

    def detect_type(self, filename: str) -> str:
        """Detect benchmark type from filename."""
        for bench_type, pattern in self.TYPE_PATTERNS.items():
            if pattern.search(filename):
                return bench_type
        return "other"

    def extract_timestamp(self, filename: str) -> tuple[str | None, datetime | None]:
        """Extract timestamp from filename."""
        match = self.TIMESTAMP_PATTERN.search(filename)
        if match:
            year, month, day, time = match.groups()
            timestamp_str = f"{year}-{month}-{day}-{time}-UTC"
            try:
                # Parse for datetime object
                dt = datetime.strptime(f"{year}-{month}-{day}-{time}", "%Y-%m-%d-%H%M")
            except ValueError:
                pass
            else:
                return timestamp_str, dt
        return None, None

    def organize_file(self, filepath: Path) -> dict[str, str]:
        """Organize a single file."""
        filename = filepath.name
        result = {"original": str(filepath), "status": "skipped", "reason": ""}

        # Skip if already organized or special files
        if any(
            parent.name in ["by-date", "by-type", "latest", "archive"]
            for parent in filepath.parents
        ):
            result["reason"] = "Already organized"
            return result

        if filename in ["README.md", "ORGANIZATION_PROPOSAL.md", ".gitkeep"]:
            result["reason"] = "Special file"
            return result

        # Detect type and timestamp
        bench_type = self.detect_type(filename)
        timestamp_str, timestamp_dt = self.extract_timestamp(filename)

        if not timestamp_str:
            # Try to get from file modification time
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            timestamp_str = mtime.strftime("%Y-%m-%d-%H%M-UTC")
            timestamp_dt = mtime

        # Determine target paths
        date_path = (
            self.by_date_dir
            / timestamp_dt.strftime("%Y-%m")
            / timestamp_dt.strftime("%d")
            / bench_type
        )
        type_path = self.by_type_dir / bench_type / timestamp_str

        # Create directories
        date_path.mkdir(parents=True, exist_ok=True)
        type_path.mkdir(parents=True, exist_ok=True)

        # Move file to type directory
        new_path = type_path / filename
        if not new_path.exists():
            shutil.move(str(filepath), str(new_path))

            # Create symlink in date directory
            date_link = date_path / filename
            if not date_link.exists():
                date_link.symlink_to(new_path)

            result["status"] = "organized"
            result["new_path"] = str(new_path)
            result["type"] = bench_type
            result["timestamp"] = timestamp_str
        else:
            result["status"] = "exists"
            result["reason"] = "File already exists at destination"

        return result

    def update_latest_links(self):
        """Update symlinks to latest results."""
        # Remove old links
        for link in self.latest_dir.glob("*"):
            if link.is_symlink():
                link.unlink()

        # Create new links for each type
        for bench_type in self.TYPE_PATTERNS:
            type_dir = self.by_type_dir / bench_type
            if not type_dir.exists():
                continue

            # Find latest timestamp directory
            timestamp_dirs = sorted(
                [d for d in type_dir.iterdir() if d.is_dir()], reverse=True
            )
            if timestamp_dirs:
                latest_dir = timestamp_dirs[0]

                # Link JSON and image files
                for ext in ["*.json", "*.png", "*.md"]:
                    for file in latest_dir.glob(ext):
                        link_name = f"{bench_type}{file.suffix}"
                        link_path = self.latest_dir / link_name
                        link_path.symlink_to(file)

    def generate_index(self):
        """Generate README.md index."""
        readme_path = self.base_dir / "README.md"

        content = ["# Benchmark Results\n\n"]
        content.append("## Latest Results\n\n")

        # List latest results
        for link in sorted(self.latest_dir.glob("*")):
            if link.is_symlink():
                target = link.readlink()
                content.append(f"- [{link.name}](latest/{link.name}) → {target.name}\n")

        content.append("\n## All Results by Type\n\n")

        # List all results by type
        for bench_type in sorted(self.TYPE_PATTERNS):
            type_dir = self.by_type_dir / bench_type
            if not type_dir.exists() or not any(type_dir.iterdir()):
                continue

            content.append(f"### {bench_type.replace('-', ' ').title()}\n\n")

            for timestamp_dir in sorted(type_dir.iterdir(), reverse=True):
                if timestamp_dir.is_dir():
                    files = list(timestamp_dir.glob("*"))
                    if files:
                        content.append(f"- **{timestamp_dir.name}**\n")
                        for file in sorted(files):
                            rel_path = file.relative_to(self.base_dir)
                            content.append(f"  - [{file.name}]({rel_path})\n")
            content.append("\n")

        # Write README
        with open(readme_path, "w") as f:
            f.writelines(content)

    def organize_all(self, dry_run: bool = False) -> list[dict[str, str]]:
        """Organize all benchmark files."""
        results = []

        if not dry_run:
            self.setup_directories()

        # Find all files to organize
        for file in self.base_dir.glob("*"):
            if file.is_file():
                if dry_run:
                    result = {
                        "original": str(file),
                        "status": "would organize",
                        "type": self.detect_type(file.name),
                        "timestamp": self.extract_timestamp(file.name)[0]
                        or "no timestamp",
                    }
                    results.append(result)
                else:
                    result = self.organize_file(file)
                    results.append(result)

        if not dry_run:
            self.update_latest_links()
            self.generate_index()

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Organize benchmark files")
    parser.add_argument(
        "--benchmarks-dir",
        default="docs/benchmarks",
        help="Path to benchmarks directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    organizer = BenchmarkOrganizer(args.benchmarks_dir)
    results = organizer.organize_all(dry_run=args.dry_run)

    # Print results
    print(f"\n{'DRY RUN' if args.dry_run else 'ORGANIZING'} RESULTS")
    print("=" * 60)

    organized = 0
    skipped = 0
    errors = 0

    for result in results:
        if result["status"] == "organized" or result["status"] == "would organize":
            organized += 1
            print(f"✓ {Path(result['original']).name}")
            if "type" in result:
                print(f"  Type: {result['type']}")
            if "timestamp" in result:
                print(f"  Timestamp: {result['timestamp']}")
            if "new_path" in result:
                print(f"  New path: {result['new_path']}")
        elif result["status"] == "skipped":
            skipped += 1
            if args.dry_run or result["reason"] != "Already organized":
                print(f"- {Path(result['original']).name}: {result['reason']}")
        else:
            errors += 1
            print(
                f"✗ {Path(result['original']).name}: {result.get('reason', 'Unknown error')}"
            )

    print(f"\nSummary: {organized} organized, {skipped} skipped, {errors} errors")

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
