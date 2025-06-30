"""
Utilities for benchmark scripts.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional


class BenchmarkOutputManager:
    """Manages output paths for benchmark results."""

    def __init__(self, base_dir: str = "benchmark_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, filename: str, extension: str) -> Path:
        """Get output path with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Remove extension if provided in filename
        base_name = filename.rsplit(".", 1)[0]

        # Create filename with timestamp
        output_name = f"{base_name}_{timestamp}.{extension}"

        return self.base_dir / output_name

    def get_latest_output(self, pattern: str) -> Optional[Path]:
        """Get the most recent output file matching pattern."""
        files = list(self.base_dir.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        return None
