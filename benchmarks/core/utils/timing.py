"""Timing utilities for benchmarks."""

import time
from typing import Any, Callable, Dict, Optional
import torch


def time_cuda_operation(
    func: Callable,
    *args,
    warmup: int = 3,
    iterations: int = 10,
    device: Optional[torch.device] = None,
    return_values: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Time a CUDA operation with proper synchronization.

    Args:
        func: Function to time
        *args: Positional arguments for func
        warmup: Number of warmup iterations
        iterations: Number of timing iterations
        device: Device to synchronize (default: current device)
        return_values: Whether to return function values
        **kwargs: Keyword arguments for func

    Returns:
        Dictionary with timing statistics and optionally values
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Synchronize before timing
    if device and device.type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time iterations
    times = []
    values = [] if return_values else None

    for _ in range(iterations):
        # Synchronize before
        if device and device.type == "cuda":
            torch.cuda.synchronize(device)
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = func(*args, **kwargs)

        # Synchronize after
        if device and device.type == "cuda":
            torch.cuda.synchronize(device)
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

        if return_values:
            values.append(result)

    # Calculate statistics
    times_tensor = torch.tensor(times)
    stats = {
        "mean": times_tensor.mean().item(),
        "std": times_tensor.std().item(),
        "min": times_tensor.min().item(),
        "max": times_tensor.max().item(),
        "median": times_tensor.median().item(),
        "total": sum(times),
        "iterations": iterations,
    }

    if return_values:
        stats["values"] = values

    return stats


def time_with_events(
    func: Callable,
    *args,
    warmup: int = 3,
    iterations: int = 10,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, float]:
    """Time operation using CUDA events for higher precision.

    Args:
        func: Function to time
        *args: Positional arguments for func
        warmup: Number of warmup iterations
        iterations: Number of timing iterations
        device: Device to use for events
        **kwargs: Keyword arguments for func

    Returns:
        Dictionary with timing statistics in milliseconds
    """
    if not torch.cuda.is_available():
        # Fallback to CPU timing
        result = time_cuda_operation(
            func, *args, warmup=warmup, iterations=iterations, **kwargs
        )
        # Convert to milliseconds
        return {k: v * 1000 if k != "iterations" else v for k, v in result.items()}

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Create events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    # Time iterations
    for i in range(iterations):
        start_events[i].record()
        _ = func(*args, **kwargs)
        end_events[i].record()

    # Wait for all operations to complete
    torch.cuda.synchronize()

    # Calculate elapsed times
    times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    times_tensor = torch.tensor(times)

    return {
        "mean": times_tensor.mean().item(),
        "std": times_tensor.std().item(),
        "min": times_tensor.min().item(),
        "max": times_tensor.max().item(),
        "median": times_tensor.median().item(),
        "total": sum(times),
        "iterations": iterations,
    }


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "", verbose: bool = True):
        """Initialize timer.

        Args:
            name: Timer name for display
            verbose: Whether to print timing
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and optionally print."""
        self.end_time = time.perf_counter()
        if self.verbose:
            prefix = f"[{self.name}] " if self.name else ""
            print(f"{prefix}Time: {self.elapsed:.4f} seconds")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


class CUDATimer:
    """CUDA event-based timer context manager."""

    def __init__(
        self,
        name: str = "",
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        """Initialize CUDA timer.

        Args:
            name: Timer name
            device: CUDA device
            verbose: Whether to print timing
        """
        self.name = name
        self.device = device or torch.cuda.current_device()
        self.verbose = verbose
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        """Start timing."""
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and optionally print."""
        if torch.cuda.is_available() and self.start_event:
            self.end_event.record()
            torch.cuda.synchronize()
            if self.verbose:
                elapsed_ms = self.start_event.elapsed_time(self.end_event)
                prefix = f"[{self.name}] " if self.name else ""
                print(f"{prefix}Time: {elapsed_ms:.2f} ms")
        else:
            elapsed = time.perf_counter() - self.start_time
            if self.verbose:
                prefix = f"[{self.name}] " if self.name else ""
                print(f"{prefix}Time: {elapsed * 1000:.2f} ms")

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if torch.cuda.is_available() and self.start_event and self.end_event:
            return self.start_event.elapsed_time(self.end_event)
        return 0.0
