"""Centralized benchmark storage management."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


class BenchmarkStorage:
    """Manages benchmark file storage with consistent organization."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize storage manager."""
        if base_dir is None:
            # Find project root
            current = Path(__file__).resolve()
            while (
                not (current / "pyproject.toml").exists() and current.parent != current
            ):
                current = current.parent
            base_dir = current / "docs" / "benchmarks"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create organized structure
        self.by_date_dir = self.base_dir / "by-date"
        self.by_type_dir = self.base_dir / "by-type"
        self.latest_dir = self.base_dir / "latest"
        self.archive_dir = self.base_dir / "archive"

        for dir_path in [
            self.by_date_dir,
            self.by_type_dir,
            self.latest_dir,
            self.archive_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

    def get_output_path(
        self, benchmark_type: str, timestamp: str, extension: str
    ) -> Path:
        """Get standardized output path for a benchmark file."""
        filename = f"{benchmark_type}-{timestamp}.{extension}"

        # Store in by-type directory
        type_dir = self.by_type_dir / benchmark_type / timestamp
        type_dir.mkdir(parents=True, exist_ok=True)

        # Also create by-date symlink
        date_parts = timestamp.split("-")
        if len(date_parts) >= 3:
            year, month, day = date_parts[:3]
            date_dir = self.by_date_dir / year / month / day / benchmark_type
            date_dir.mkdir(parents=True, exist_ok=True)

            # Create symlink in by-date
            date_link = date_dir / filename
            type_file = type_dir / filename
            if not date_link.exists():
                try:
                    # Calculate relative path from date_link to type_file
                    rel_path = os.path.relpath(type_file, date_link.parent)
                    date_link.symlink_to(rel_path)
                except (OSError, ValueError):
                    # Fallback to copy if symlinks not supported or path issues
                    pass

        return type_dir / filename

    def save_json(
        self, data: dict[str, Any], benchmark_type: str, timestamp: str
    ) -> Path:
        """Save benchmark results as JSON."""
        path = self.get_output_path(benchmark_type, timestamp, "json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_text(
        self, content: str, benchmark_type: str, timestamp: str, extension: str = "md"
    ) -> Path:
        """Save text content (markdown, txt, etc)."""
        path = self.get_output_path(benchmark_type, timestamp, extension)
        with open(path, "w") as f:
            f.write(content)
        return path

    def save_plot(self, plot_path: Path, benchmark_type: str, timestamp: str) -> Path:
        """Copy plot to organized location."""
        extension = plot_path.suffix.lstrip(".")
        dest = self.get_output_path(benchmark_type, timestamp, extension)
        shutil.copy2(plot_path, dest)
        return dest

    def update_latest_links(self, benchmark_type: str, timestamp: str):
        """Update 'latest' symlinks for a benchmark type."""
        type_dir = self.by_type_dir / benchmark_type / timestamp

        # Remove old latest links for this type
        for old_link in self.latest_dir.glob(f"{benchmark_type}.*"):
            old_link.unlink(missing_ok=True)

        # Create new latest links
        for file_path in type_dir.glob("*"):
            if file_path.is_file():
                extension = file_path.suffix
                latest_link = self.latest_dir / f"{benchmark_type}{extension}"
                try:
                    # Calculate relative path from latest_link to file_path
                    rel_path = os.path.relpath(file_path, latest_link.parent)
                    latest_link.symlink_to(rel_path)
                except (OSError, ValueError):
                    # Fallback to copy
                    shutil.copy2(file_path, latest_link)

    def list_benchmarks(
        self, benchmark_type: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """List available benchmark results."""
        results = []

        search_dir = self.by_type_dir
        if benchmark_type:
            search_dir = search_dir / benchmark_type

        for json_path in sorted(search_dir.rglob("*.json"), reverse=True):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                    results.append(
                        {
                            "path": json_path,
                            "type": benchmark_type or json_path.parent.parent.name,
                            "timestamp": json_path.stem.split("-", 1)[-1],
                            "metadata": data.get("metadata", {}),
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                continue

            if limit and len(results) >= limit:
                break

        return results

    def archive_old_results(self, days: int = 30):
        """Archive results older than specified days."""
        cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)

        for type_dir in self.by_type_dir.iterdir():
            if not type_dir.is_dir():
                continue

            for timestamp_dir in type_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                # Parse timestamp from directory name
                try:
                    timestamp_str = timestamp_dir.name
                    # Convert YYYY-MM-DD-HHMM-UTC to datetime
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d-%H%M-UTC")
                    if dt.timestamp() < cutoff:
                        # Move to archive
                        archive_path = (
                            self.archive_dir / type_dir.name / timestamp_dir.name
                        )
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(timestamp_dir), str(archive_path))
                except ValueError:
                    # Skip if can't parse timestamp
                    continue
