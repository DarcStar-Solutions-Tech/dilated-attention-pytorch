"""
Real-time memory profiler for Phase 1.4 memory management.

This module provides comprehensive memory profiling capabilities including
allocation tracking, timeline analysis, pattern detection, and visualization.
"""

import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor

logger = logging.getLogger("dilated_attention_pytorch.memory_profiler")


@dataclass
class AllocationEvent:
    """Represents a memory allocation event."""

    timestamp: float
    operation: str  # Operation name (e.g., "forward", "backward", "optimizer")
    size_bytes: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    tensor_id: int
    stack_trace: str
    pool_type: str = "unknown"  # Which pool handled the allocation
    numa_node: Optional[int] = None

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""

    timestamp: float
    total_allocated: int  # Total allocated memory in bytes
    total_cached: int  # Total cached memory in bytes
    peak_allocated: int  # Peak allocated memory in bytes
    num_allocations: int  # Number of active allocations
    num_deallocations: int  # Number of deallocations so far
    fragmentation_score: float = 0.0
    numa_stats: Dict[str, Any] = field(default_factory=dict)
    pool_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in MB."""
        return self.total_allocated / (1024 * 1024)

    @property
    def allocated_gb(self) -> float:
        """Allocated memory in GB."""
        return self.total_allocated / (1024 * 1024 * 1024)


@dataclass
class AllocationPattern:
    """Represents a detected allocation pattern."""

    pattern_type: str  # "periodic", "burst", "linear_growth", "stable"
    frequency: float  # Events per second
    average_size: float  # Average allocation size
    size_variance: float  # Size variance
    confidence: float  # Pattern confidence (0-1)
    description: str  # Human-readable description
    recommendations: List[str] = field(default_factory=list)


class MemoryProfiler:
    """
    Real-time memory profiler with comprehensive monitoring.

    Features:
    - Real-time allocation and deallocation tracking
    - Memory timeline with snapshots
    - Allocation pattern detection and analysis
    - Stack trace capture for debugging
    - Integration with memory pools
    - NUMA-aware profiling
    - Automatic recommendations
    """

    def __init__(
        self,
        enable_stack_traces: bool = True,
        max_events: int = 10000,
        snapshot_interval: float = 1.0,
        pattern_analysis_window: int = 100,
        enable_numa_tracking: bool = True,
    ):
        """
        Initialize the memory profiler.

        Args:
            enable_stack_traces: Capture stack traces for allocations
            max_events: Maximum number of events to store
            snapshot_interval: Interval between memory snapshots (seconds)
            pattern_analysis_window: Window size for pattern analysis
            enable_numa_tracking: Enable NUMA-aware tracking
        """
        self.enable_stack_traces = enable_stack_traces
        self.max_events = max_events
        self.snapshot_interval = snapshot_interval
        self.pattern_analysis_window = pattern_analysis_window
        self.enable_numa_tracking = enable_numa_tracking

        # Event tracking
        self.allocation_events: deque = deque(maxlen=max_events)
        self.deallocation_events: deque = deque(maxlen=max_events)
        self.memory_snapshots: deque = deque(maxlen=max_events // 10)

        # Active allocations tracking
        self.active_allocations: Dict[int, AllocationEvent] = {}

        # Pattern analysis
        self.detected_patterns: List[AllocationPattern] = []
        self.pattern_last_analysis: float = 0.0

        # Statistics
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "peak_memory": 0,
            "total_memory_allocated": 0,
            "average_allocation_size": 0.0,
            "allocation_rate": 0.0,  # Allocations per second
            "profiling_overhead": 0.0,  # Profiling overhead in seconds
        }

        # Current operation context
        self.current_operation = "unknown"
        self.operation_stack: List[str] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background snapshot thread
        self._snapshot_thread: Optional[threading.Thread] = None
        self._stop_profiling = threading.Event()
        self._profiling_active = False

        logger.info(f"Memory profiler initialized with {max_events} max events")

    def start_profiling(self) -> None:
        """Start background profiling."""
        if self._profiling_active:
            return

        self._profiling_active = True
        self._stop_profiling.clear()

        # Start snapshot thread
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_worker, daemon=True
        )
        self._snapshot_thread.start()

        logger.info("Memory profiling started")

    def stop_profiling(self) -> None:
        """Stop background profiling."""
        if not self._profiling_active:
            return

        self._profiling_active = False
        self._stop_profiling.set()

        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=1.0)

        logger.info("Memory profiling stopped")

    def _snapshot_worker(self) -> None:
        """Background worker for taking memory snapshots."""
        while not self._stop_profiling.wait(self.snapshot_interval):
            try:
                self._take_snapshot()
            except Exception as e:
                logger.debug(f"Snapshot failed: {e}")

    def _take_snapshot(self) -> None:
        """Take a memory snapshot."""
        with self._lock:
            current_time = time.perf_counter()

            # Get memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                peak = torch.cuda.max_memory_allocated()
            else:
                # For CPU, estimate from active allocations
                allocated = sum(
                    event.size_bytes for event in self.active_allocations.values()
                )
                cached = 0
                peak = self.stats["peak_memory"]

            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=current_time,
                total_allocated=allocated,
                total_cached=cached,
                peak_allocated=peak,
                num_allocations=len(self.active_allocations),
                num_deallocations=self.stats["total_deallocations"],
            )

            self.memory_snapshots.append(snapshot)

            # Update peak memory
            self.stats["peak_memory"] = max(self.stats["peak_memory"], allocated)

            # Analyze patterns periodically
            if current_time - self.pattern_last_analysis > 10.0:  # Every 10 seconds
                self._analyze_patterns()
                self.pattern_last_analysis = current_time

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling a specific operation."""
        start_time = time.perf_counter()

        with self._lock:
            self.operation_stack.append(self.current_operation)
            self.current_operation = operation_name

        try:
            yield
        finally:
            end_time = time.perf_counter()

            with self._lock:
                if self.operation_stack:
                    self.current_operation = self.operation_stack.pop()
                else:
                    self.current_operation = "unknown"

                # Track operation time
                operation_time = end_time - start_time
                self.stats["profiling_overhead"] += (
                    operation_time * 0.01
                )  # Estimate 1% overhead

    def record_allocation(
        self,
        tensor: Tensor,
        pool_type: str = "unknown",
        numa_node: Optional[int] = None,
    ) -> None:
        """
        Record a memory allocation event.

        Args:
            tensor: Allocated tensor
            pool_type: Type of pool that handled allocation
            numa_node: NUMA node where allocation occurred
        """
        if not self._profiling_active:
            return

        start_time = time.perf_counter()

        try:
            with self._lock:
                # Calculate allocation size
                size_bytes = tensor.element_size() * tensor.numel()

                # Get stack trace if enabled
                stack_trace = ""
                if self.enable_stack_traces:
                    stack_trace = "".join(
                        traceback.format_stack()[-5:]
                    )  # Last 5 frames

                # Create allocation event
                event = AllocationEvent(
                    timestamp=time.perf_counter(),
                    operation=self.current_operation,
                    size_bytes=size_bytes,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    tensor_id=id(tensor),
                    stack_trace=stack_trace,
                    pool_type=pool_type,
                    numa_node=numa_node,
                )

                # Store event
                self.allocation_events.append(event)
                self.active_allocations[event.tensor_id] = event

                # Update statistics
                self.stats["total_allocations"] += 1
                self.stats["total_memory_allocated"] += size_bytes

                # Update average allocation size
                total_allocs = self.stats["total_allocations"]
                total_memory = self.stats["total_memory_allocated"]
                self.stats["average_allocation_size"] = total_memory / total_allocs

        except Exception as e:
            logger.debug(f"Failed to record allocation: {e}")

        finally:
            # Track profiling overhead
            overhead = time.perf_counter() - start_time
            self.stats["profiling_overhead"] += overhead

    def record_deallocation(self, tensor: Tensor) -> None:
        """
        Record a memory deallocation event.

        Args:
            tensor: Tensor being deallocated
        """
        if not self._profiling_active:
            return

        try:
            with self._lock:
                tensor_id = id(tensor)

                # Find original allocation
                if tensor_id in self.active_allocations:
                    original_event = self.active_allocations.pop(tensor_id)

                    # Record deallocation
                    dealloc_event = {
                        "timestamp": time.perf_counter(),
                        "operation": self.current_operation,
                        "tensor_id": tensor_id,
                        "original_event": original_event,
                        "lifetime": time.perf_counter() - original_event.timestamp,
                    }

                    self.deallocation_events.append(dealloc_event)
                    self.stats["total_deallocations"] += 1

        except Exception as e:
            logger.debug(f"Failed to record deallocation: {e}")

    def _analyze_patterns(self) -> None:
        """Analyze allocation patterns and generate recommendations."""
        if len(self.allocation_events) < self.pattern_analysis_window:
            return

        try:
            # Get recent events
            recent_events = list(self.allocation_events)[
                -self.pattern_analysis_window :
            ]

            # Analyze allocation rate
            time_span = recent_events[-1].timestamp - recent_events[0].timestamp
            if time_span > 0:
                allocation_rate = len(recent_events) / time_span
                self.stats["allocation_rate"] = allocation_rate

            # Analyze size patterns
            sizes = [event.size_bytes for event in recent_events]
            avg_size = sum(sizes) / len(sizes)
            size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)

            # Detect burst pattern
            if allocation_rate > 100:  # More than 100 allocations per second
                pattern = AllocationPattern(
                    pattern_type="burst",
                    frequency=allocation_rate,
                    average_size=avg_size,
                    size_variance=size_variance,
                    confidence=0.8,
                    description=f"High allocation rate: {allocation_rate:.1f}/sec",
                    recommendations=[
                        "Consider using memory pools for frequent allocations",
                        "Batch operations to reduce allocation overhead",
                        "Pre-allocate buffers for known patterns",
                    ],
                )
                self.detected_patterns.append(pattern)

            # Detect large allocation pattern
            large_allocations = [s for s in sizes if s > 100 * 1024 * 1024]  # > 100MB
            if len(large_allocations) / len(sizes) > 0.3:  # > 30% large allocations
                pattern = AllocationPattern(
                    pattern_type="large_allocation",
                    frequency=allocation_rate,
                    average_size=avg_size,
                    size_variance=size_variance,
                    confidence=0.9,
                    description=f"High ratio of large allocations: {len(large_allocations)}/{len(sizes)}",
                    recommendations=[
                        "Enable NUMA-aware allocation for large tensors",
                        "Consider gradient checkpointing to reduce memory usage",
                        "Use mixed precision training if possible",
                    ],
                )
                self.detected_patterns.append(pattern)

            # Keep only recent patterns
            cutoff_time = time.perf_counter() - 60.0  # Last minute
            self.detected_patterns = [
                p
                for p in self.detected_patterns
                if hasattr(p, "timestamp") and getattr(p, "timestamp", 0) > cutoff_time
            ]

        except Exception as e:
            logger.debug(f"Pattern analysis failed: {e}")

    def get_memory_timeline(self, duration: float = 60.0) -> List[MemorySnapshot]:
        """
        Get memory timeline for the last N seconds.

        Args:
            duration: Duration in seconds

        Returns:
            List of memory snapshots
        """
        cutoff_time = time.perf_counter() - duration

        with self._lock:
            return [
                snapshot
                for snapshot in self.memory_snapshots
                if snapshot.timestamp > cutoff_time
            ]

    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get allocation summary statistics."""
        with self._lock:
            # Analyze allocations by pool type
            pool_stats = defaultdict(lambda: {"count": 0, "total_size": 0})
            for event in self.allocation_events:
                pool_stats[event.pool_type]["count"] += 1
                pool_stats[event.pool_type]["total_size"] += event.size_bytes

            # Analyze allocations by operation
            operation_stats = defaultdict(lambda: {"count": 0, "total_size": 0})
            for event in self.allocation_events:
                operation_stats[event.operation]["count"] += 1
                operation_stats[event.operation]["total_size"] += event.size_bytes

            # Analyze allocations by device
            device_stats = defaultdict(lambda: {"count": 0, "total_size": 0})
            for event in self.allocation_events:
                device_key = str(event.device)
                device_stats[device_key]["count"] += 1
                device_stats[device_key]["total_size"] += event.size_bytes

            return {
                "basic_stats": self.stats.copy(),
                "pool_breakdown": dict(pool_stats),
                "operation_breakdown": dict(operation_stats),
                "device_breakdown": dict(device_stats),
                "active_allocations": len(self.active_allocations),
                "detected_patterns": len(self.detected_patterns),
                "profiling_active": self._profiling_active,
            }

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on profiling data."""
        recommendations = []

        with self._lock:
            stats = self.stats

            # High allocation rate
            if stats["allocation_rate"] > 50:
                recommendations.append(
                    f"High allocation rate ({stats['allocation_rate']:.1f}/sec) - "
                    "consider memory pooling"
                )

            # Large average allocation size
            if stats["average_allocation_size"] > 50 * 1024 * 1024:  # > 50MB
                recommendations.append(
                    f"Large average allocation ({stats['average_allocation_size'] / (1024 * 1024):.1f}MB) - "
                    "enable NUMA-aware allocation"
                )

            # High number of active allocations
            if len(self.active_allocations) > 1000:
                recommendations.append(
                    f"High number of active allocations ({len(self.active_allocations)}) - "
                    "consider tensor reuse strategies"
                )

            # Add pattern-specific recommendations
            for pattern in self.detected_patterns:
                recommendations.extend(pattern.recommendations)

        return recommendations

    def generate_report(self, include_timeline: bool = True) -> str:
        """Generate comprehensive profiling report."""
        lines = ["Memory Profiling Report", "=" * 25, ""]

        # Basic statistics
        summary = self.get_allocation_summary()
        basic_stats = summary["basic_stats"]

        lines.extend(
            [
                "Basic Statistics:",
                f"  Total Allocations: {basic_stats['total_allocations']:,}",
                f"  Total Deallocations: {basic_stats['total_deallocations']:,}",
                f"  Active Allocations: {summary['active_allocations']:,}",
                f"  Peak Memory: {basic_stats['peak_memory'] / (1024**3):.2f} GB",
                f"  Total Memory Allocated: {basic_stats['total_memory_allocated'] / (1024**3):.2f} GB",
                f"  Average Allocation Size: {basic_stats['average_allocation_size'] / (1024**2):.2f} MB",
                f"  Allocation Rate: {basic_stats['allocation_rate']:.1f} allocs/sec",
                f"  Profiling Overhead: {basic_stats['profiling_overhead']:.3f} sec",
                "",
            ]
        )

        # Pool breakdown
        if summary["pool_breakdown"]:
            lines.append("Pool Breakdown:")
            for pool_type, stats in summary["pool_breakdown"].items():
                size_gb = stats["total_size"] / (1024**3)
                lines.append(
                    f"  {pool_type}: {stats['count']:,} allocs, {size_gb:.2f} GB"
                )
            lines.append("")

        # Operation breakdown
        if summary["operation_breakdown"]:
            lines.append("Operation Breakdown:")
            sorted_ops = sorted(
                summary["operation_breakdown"].items(),
                key=lambda x: x[1]["total_size"],
                reverse=True,
            )
            for op_name, stats in sorted_ops[:10]:  # Top 10
                size_gb = stats["total_size"] / (1024**3)
                lines.append(
                    f"  {op_name}: {stats['count']:,} allocs, {size_gb:.2f} GB"
                )
            lines.append("")

        # Detected patterns
        if self.detected_patterns:
            lines.append("Detected Patterns:")
            for pattern in self.detected_patterns[-5:]:  # Last 5 patterns
                lines.append(f"  {pattern.pattern_type}: {pattern.description}")
            lines.append("")

        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            lines.append("Recommendations:")
            for rec in recommendations[:10]:  # Top 10
                lines.append(f"  • {rec}")
            lines.append("")

        # Memory timeline
        if include_timeline:
            timeline = self.get_memory_timeline(60.0)  # Last minute
            if timeline:
                lines.append("Memory Timeline (last 60s):")
                lines.append("  Time    | Allocated | Cached   | Active")
                lines.append("  --------|-----------|----------|-------")

                for snapshot in timeline[-10:]:  # Last 10 snapshots
                    relative_time = snapshot.timestamp - timeline[0].timestamp
                    lines.append(
                        f"  {relative_time:6.1f}s | "
                        f"{snapshot.allocated_gb:8.2f}G | "
                        f"{snapshot.total_cached / (1024**3):7.2f}G | "
                        f"{snapshot.num_allocations:6,}"
                    )

        return "\n".join(lines)

    def export_data(self, filepath: Path) -> None:
        """Export profiling data to file."""
        try:
            import json

            data = {
                "stats": self.stats,
                "allocation_events": [
                    {
                        "timestamp": event.timestamp,
                        "operation": event.operation,
                        "size_bytes": event.size_bytes,
                        "shape": list(event.shape),
                        "dtype": str(event.dtype),
                        "device": str(event.device),
                        "pool_type": event.pool_type,
                        "numa_node": event.numa_node,
                    }
                    for event in self.allocation_events
                ],
                "memory_snapshots": [
                    {
                        "timestamp": snapshot.timestamp,
                        "total_allocated": snapshot.total_allocated,
                        "total_cached": snapshot.total_cached,
                        "peak_allocated": snapshot.peak_allocated,
                        "num_allocations": snapshot.num_allocations,
                        "num_deallocations": snapshot.num_deallocations,
                    }
                    for snapshot in self.memory_snapshots
                ],
                "detected_patterns": [
                    {
                        "pattern_type": pattern.pattern_type,
                        "frequency": pattern.frequency,
                        "average_size": pattern.average_size,
                        "confidence": pattern.confidence,
                        "description": pattern.description,
                        "recommendations": pattern.recommendations,
                    }
                    for pattern in self.detected_patterns
                ],
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Profiling data exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export profiling data: {e}")

    def clear_data(self) -> None:
        """Clear all profiling data."""
        with self._lock:
            self.allocation_events.clear()
            self.deallocation_events.clear()
            self.memory_snapshots.clear()
            self.active_allocations.clear()
            self.detected_patterns.clear()

            # Reset stats
            self.stats = {
                "total_allocations": 0,
                "total_deallocations": 0,
                "peak_memory": 0,
                "total_memory_allocated": 0,
                "average_allocation_size": 0.0,
                "allocation_rate": 0.0,
                "profiling_overhead": 0.0,
            }

        logger.info("Profiling data cleared")


# Global profiler instance
_MEMORY_PROFILER: Optional[MemoryProfiler] = None
_PROFILER_LOCK = threading.Lock()


def get_memory_profiler(**kwargs) -> MemoryProfiler:
    """
    Get the global memory profiler instance.

    Args:
        **kwargs: Configuration for the profiler (only used on first call)

    Returns:
        Global memory profiler instance
    """
    global _MEMORY_PROFILER

    if _MEMORY_PROFILER is None:
        with _PROFILER_LOCK:
            if _MEMORY_PROFILER is None:
                _MEMORY_PROFILER = MemoryProfiler(**kwargs)

    return _MEMORY_PROFILER


def reset_memory_profiler() -> None:
    """Reset the global memory profiler."""
    global _MEMORY_PROFILER

    with _PROFILER_LOCK:
        if _MEMORY_PROFILER is not None:
            _MEMORY_PROFILER.stop_profiling()
            _MEMORY_PROFILER = None


@contextmanager
def profile_memory(operation_name: str = "operation"):
    """
    Context manager for profiling memory usage of an operation.

    Args:
        operation_name: Name of the operation being profiled
    """
    profiler = get_memory_profiler()

    if not profiler._profiling_active:
        profiler.start_profiling()

    with profiler.profile_operation(operation_name):
        yield profiler
