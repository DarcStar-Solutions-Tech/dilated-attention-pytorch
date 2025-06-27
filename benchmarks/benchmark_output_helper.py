"""
Helper module for organized benchmark output.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import ClassVar


class BenchmarkOutputHelper:
    """Helper for organizing benchmark outputs."""

    BENCHMARK_TYPES: ClassVar[dict[str, str]] = {
        "benchmark_all_implementations.py": "comprehensive",
        "benchmark_long_sequences.py": "long-sequences",
        "benchmark_distributed.py": "distributed",
        "benchmark.py": "standard",
        "benchmark_sequence_limits.py": "sequence-limits",
        "benchmark_ring_billion_tokens.py": "billion-token",
    }

    def __init__(self, script_name: str, output_dir: str = "docs/benchmarks"):
        """Initialize helper with script name and output directory."""
        self.script_name = Path(script_name).name
        self.base_dir = Path(output_dir)
        self.benchmark_type = self.BENCHMARK_TYPES.get(self.script_name, "other")
        self.timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")

    def get_output_dir(self) -> Path:
        """Get organized output directory for this benchmark run."""
        # Parse timestamp for date hierarchy
        parts = self.timestamp.split("-")
        year_month = f"{parts[0]}-{parts[1]}"
        day = parts[2]

        # Create type-based path
        type_path = self.base_dir / "by-type" / self.benchmark_type / self.timestamp
        type_path.mkdir(parents=True, exist_ok=True)

        # Create date-based symlink directory
        date_path = self.base_dir / "by-date" / year_month / day / self.benchmark_type
        date_path.mkdir(parents=True, exist_ok=True)

        return type_path

    def get_output_path(self, filename: str) -> Path:
        """Get full output path for a file."""
        return self.get_output_dir() / filename

    def update_latest_link(self, file_path: Path):
        """Update the 'latest' symlink for this file."""
        latest_dir = self.base_dir / "latest"
        latest_dir.mkdir(exist_ok=True)

        # Determine link name
        link_name = f"{self.benchmark_type}{file_path.suffix}"
        link_path = latest_dir / link_name

        # Remove old link if exists
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        # Create new link (relative to make it portable)
        try:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, latest_dir)
            link_path.symlink_to(rel_path)
        except Exception:
            # Fallback to absolute if relative fails
            link_path.symlink_to(file_path.absolute())

    def create_date_symlink(self, file_path: Path):
        """Create symlink in date-based directory."""
        parts = self.timestamp.split("-")
        year_month = f"{parts[0]}-{parts[1]}"
        day = parts[2]

        date_dir = self.base_dir / "by-date" / year_month / day / self.benchmark_type
        date_link = date_dir / file_path.name

        if not date_link.exists():
            try:
                # Calculate relative path
                rel_path = os.path.relpath(file_path, date_dir)
                date_link.symlink_to(rel_path)
            except Exception:
                # Fallback to absolute
                date_link.symlink_to(file_path.absolute())

    def finalize_output(self, *file_paths: Path):
        """Finalize output by creating all necessary links."""
        for file_path in file_paths:
            if file_path.exists():
                self.update_latest_link(file_path)
                self.create_date_symlink(file_path)

    @property
    def output_dir_str(self) -> str:
        """Get output directory as string (for compatibility)."""
        return str(self.get_output_dir())


# Example usage:
if __name__ == "__main__":
    # In a benchmark script:
    helper = BenchmarkOutputHelper(__file__)

    # Get paths
    json_path = helper.get_output_path(f"benchmark-{helper.benchmark_type}-{helper.timestamp}.json")
    png_path = helper.get_output_path(f"benchmark-{helper.benchmark_type}-{helper.timestamp}.png")

    print(f"Benchmark type: {helper.benchmark_type}")
    print(f"Output directory: {helper.output_dir_str}")
    print(f"JSON path: {json_path}")
    print(f"PNG path: {png_path}")

    # After saving files:
    # helper.finalize_output(json_path, png_path)
