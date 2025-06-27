#!/usr/bin/env python3
"""
Update benchmark scripts to save outputs in organized structure.
"""

from pathlib import Path


def get_organized_output_path(
    benchmark_type: str, timestamp: str, base_dir: str = "docs/benchmarks"
) -> str:
    """
    Get the organized output path for a benchmark.

    Args:
        benchmark_type: Type of benchmark (e.g., 'long-sequences', 'comprehensive')
        timestamp: Timestamp string in format 'YYYY-MM-DD-HHMM-UTC'
        base_dir: Base directory for benchmarks

    Returns:
        Path string for organized output
    """
    # Parse timestamp
    parts = timestamp.split("-")
    year_month = f"{parts[0]}-{parts[1]}"
    day = parts[2]

    # Create paths
    by_type_path = Path(base_dir) / "by-type" / benchmark_type / timestamp
    by_date_path = Path(base_dir) / "by-date" / year_month / day / benchmark_type

    # Ensure directories exist
    by_type_path.mkdir(parents=True, exist_ok=True)
    by_date_path.mkdir(parents=True, exist_ok=True)

    return str(by_type_path)


def update_latest_symlink(benchmark_type: str, file_path: Path, base_dir: str = "docs/benchmarks"):
    """Update the 'latest' symlink for a benchmark type."""
    latest_dir = Path(base_dir) / "latest"
    latest_dir.mkdir(exist_ok=True)

    # Determine link name
    link_name = f"{benchmark_type}{file_path.suffix}"
    link_path = latest_dir / link_name

    # Remove old link if exists
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    # Create new link
    link_path.symlink_to(file_path.absolute())


# Example usage in benchmark scripts:
if __name__ == "__main__":
    # Example: In benchmark_long_sequences.py
    timestamp = "2025-06-27-0700-UTC"
    output_dir = get_organized_output_path("long-sequences", timestamp)

    # Save files
    json_path = Path(output_dir) / f"benchmark-long-sequences-{timestamp}.json"
    png_path = Path(output_dir) / f"benchmark-long-sequences-{timestamp}.png"

    print(f"JSON output: {json_path}")
    print(f"PNG output: {png_path}")

    # After saving, update latest links
    # update_latest_symlink("long-sequences", json_path)
    # update_latest_symlink("long-sequences", png_path)
