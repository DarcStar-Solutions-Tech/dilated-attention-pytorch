"""Memory management utilities for benchmarks."""

import gc
from typing import Any, Callable, Dict, Optional, Tuple
import torch


def cleanup_memory(device: Optional[torch.device] = None) -> None:
    """Clean up GPU memory.

    Args:
        device: Device to clean up (default: current device)
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if device and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elif torch.cuda.is_available():
            # Reset for all devices
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)


def measure_peak_memory(
    func: Callable,
    *args,
    device: Optional[torch.device] = None,
    cleanup_before: bool = True,
    cleanup_after: bool = False,
    **kwargs,
) -> Tuple[Any, float]:
    """Measure peak memory usage of a function.

    Args:
        func: Function to measure
        *args: Positional arguments for func
        device: Device to measure (default: current device)
        cleanup_before: Whether to clean memory before measurement
        cleanup_after: Whether to clean memory after measurement
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (function result, peak memory in MB)
    """
    if cleanup_before:
        cleanup_memory(device)

    if device and device.type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

    # Run function
    result = func(*args, **kwargs)

    # Measure memory
    if device and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_mb = 0.0

    if cleanup_after:
        cleanup_memory(device)

    return result, peak_mb


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get current memory statistics.

    Args:
        device: Device to query (default: current device)

    Returns:
        Dictionary with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0,
        }

    if device and device.type == "cuda":
        device_id = device.index or 0
    else:
        device_id = torch.cuda.current_device()

    return {
        "allocated": torch.cuda.memory_allocated(device_id) / 1024**2,
        "reserved": torch.cuda.memory_reserved(device_id) / 1024**2,
        "free": (
            torch.cuda.memory_reserved(device_id)
            - torch.cuda.memory_allocated(device_id)
        )
        / 1024**2,
        "total": torch.cuda.get_device_properties(device_id).total_memory / 1024**2,
    }


def check_memory_available(
    required_mb: float,
    device: Optional[torch.device] = None,
    safety_factor: float = 1.2,
) -> bool:
    """Check if enough memory is available.

    Args:
        required_mb: Required memory in MB
        device: Device to check (default: current device)
        safety_factor: Safety multiplier for required memory

    Returns:
        True if enough memory is available
    """
    stats = get_memory_stats(device)
    available = stats["total"] - stats["allocated"]
    return available >= (required_mb * safety_factor)


def estimate_tensor_memory(*shapes, dtype: torch.dtype = torch.float32) -> float:
    """Estimate memory required for tensors.

    Args:
        *shapes: Tensor shapes
        dtype: Data type

    Returns:
        Estimated memory in MB
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_elements = sum(torch.tensor(shape).prod().item() for shape in shapes)
    return (total_elements * bytes_per_element) / 1024**2


class MemoryMonitor:
    """Context manager for monitoring memory usage."""

    def __init__(self, device: Optional[torch.device] = None, tag: str = ""):
        """Initialize memory monitor.

        Args:
            device: Device to monitor
            tag: Tag for identification
        """
        self.device = device
        self.tag = tag
        self.start_memory = 0.0
        self.peak_memory = 0.0

    def __enter__(self):
        """Enter context and record starting memory."""
        cleanup_memory(self.device)

        if self.device and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.start_memory = torch.cuda.memory_allocated(self.device) / 1024**2
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated() / 1024**2

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record peak memory."""
        if self.device and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated() / 1024**2

        if self.tag:
            print(f"[{self.tag}] Memory: {self.peak_memory - self.start_memory:.2f} MB")

    @property
    def memory_used(self) -> float:
        """Get memory used during context."""
        return self.peak_memory - self.start_memory
