"""Unified output management for all benchmark scripts."""

from pathlib import Path
from typing import Any

from .metadata import create_metadata
from .storage import BenchmarkStorage


class BenchmarkOutputManager:
    """Manages all benchmark outputs with consistent organization."""

    def __init__(
        self,
        benchmark_type: str,
        parameters: dict[str, Any] | None = None,
        storage: BenchmarkStorage | None = None,
    ):
        """Initialize output manager for a benchmark run."""
        self.benchmark_type = benchmark_type
        self.metadata = create_metadata(benchmark_type, parameters)
        self.storage = storage or BenchmarkStorage()
        self.results: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}

    def add_result(self, key: str, value: Any):
        """Add a result to the benchmark data."""
        self.results[key] = value

    def set_summary(self, summary: dict[str, Any]):
        """Set the benchmark summary."""
        self.summary = summary

    def save_results(self) -> dict[str, Path]:
        """Save all benchmark results and return paths."""
        # Prepare complete data structure
        data = {
            "metadata": self.metadata.to_dict(),
            "results": self.results,
            "summary": self.summary,
        }

        # Save JSON
        json_path = self.storage.save_json(
            data, self.benchmark_type, self.metadata.timestamp
        )

        paths = {"json": json_path}

        # Generate and save markdown summary if we have results
        if self.results:
            markdown = self._generate_markdown_summary(data)
            md_path = self.storage.save_text(
                markdown, self.benchmark_type, self.metadata.timestamp, "md"
            )
            paths["markdown"] = md_path

        # Update latest links
        self.storage.update_latest_links(self.benchmark_type, self.metadata.timestamp)

        return paths

    def save_plot(self, plot_path: Path) -> Path:
        """Save a plot file to organized location."""
        return self.storage.save_plot(
            plot_path, self.benchmark_type, self.metadata.timestamp
        )

    def _generate_markdown_summary(self, data: dict[str, Any]) -> str:
        """Generate markdown summary of benchmark results."""
        md_lines = [
            f"# {self.benchmark_type.replace('-', ' ').title()} Benchmark Results",
            "",
            f"**Date**: {self.metadata.timestamp}  ",
            f"**Git Commit**: {self.metadata.git_commit[:8] if self.metadata.git_commit else 'N/A'}  ",
            f"**Hardware**: {self.metadata.hardware.get('gpu_names', ['CPU'])[0]}  ",
            "",
        ]

        # Add parameters if present
        if self.metadata.parameters:
            md_lines.extend(["## Parameters", ""])
            for key, value in self.metadata.parameters.items():
                md_lines.append(f"- **{key}**: {value}")
            md_lines.append("")

        # Add summary if present
        if self.summary:
            md_lines.extend(["## Summary", ""])
            for key, value in self.summary.items():
                if isinstance(value, (int, float)):
                    md_lines.append(
                        f"- **{key.replace('_', ' ').title()}**: {value:,.2f}"
                    )
                else:
                    md_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md_lines.append("")

        # Add detailed results
        if self.results:
            md_lines.extend(["## Detailed Results", ""])

            # Try to format as table if results have consistent structure
            if isinstance(next(iter(self.results.values())), dict):
                # Collect all keys for table headers
                all_keys = set()
                for result in self.results.values():
                    if isinstance(result, dict):
                        all_keys.update(result.keys())

                if all_keys:
                    headers = ["Implementation"] + sorted(all_keys)
                    md_lines.append("| " + " | ".join(headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                    for impl, metrics in self.results.items():
                        if isinstance(metrics, dict):
                            row = [impl]
                            for key in sorted(all_keys):
                                value = metrics.get(key, "N/A")
                                if isinstance(value, float):
                                    row.append(f"{value:.2f}")
                                else:
                                    row.append(str(value))
                            md_lines.append("| " + " | ".join(row) + " |")
            else:
                # Fallback to simple list
                for key, value in self.results.items():
                    md_lines.append(f"- **{key}**: {value}")

        return "\n".join(md_lines)

    @classmethod
    def from_existing_script(
        cls, script_name: str, **kwargs
    ) -> "BenchmarkOutputManager":
        """Create manager for existing benchmark script (backward compatibility)."""
        # Extract benchmark type from script name
        script_path = Path(script_name)
        benchmark_type = script_path.stem.replace("benchmark_", "").replace("_", "-")

        return cls(benchmark_type=benchmark_type, **kwargs)
