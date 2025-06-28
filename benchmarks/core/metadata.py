"""Benchmark metadata collection and standardization."""

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch


@dataclass
class BenchmarkMetadata:
    """Standard metadata for all benchmark runs."""

    benchmark_type: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool
    hardware: dict[str, Any]
    python_version: str
    torch_version: str
    cuda_version: str | None
    command_line: str | None = None
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_git_info() -> tuple[str | None, bool]:
    """Get current git commit and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()

        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            != ""
        )

        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def get_system_info() -> dict[str, Any]:
    """Collect comprehensive system information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": torch.get_num_threads(),
    }

    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(info["gpu_count"])
        ]
        info["gpu_memory_gb"] = [
            torch.cuda.get_device_properties(i).total_memory / 1024**3
            for i in range(info["gpu_count"])
        ]
        info["cuda_capability"] = [
            f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            for i in range(info["gpu_count"])
        ]
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []

    return info


def create_metadata(
    benchmark_type: str,
    parameters: dict[str, Any] | None = None,
    command_line: str | None = None,
) -> BenchmarkMetadata:
    """Create standard metadata for a benchmark run."""
    git_commit, git_dirty = get_git_info()

    return BenchmarkMetadata(
        benchmark_type=benchmark_type,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC"),
        git_commit=git_commit,
        git_dirty=git_dirty,
        hardware=get_system_info(),
        python_version=sys.version,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        command_line=command_line or " ".join(sys.argv),
        parameters=parameters or {},
    )
