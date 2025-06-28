"""
Enhanced memory pool integrating fragment-aware and bucketed allocation.

This module combines the fragment-aware memory pool with the bucketed memory pool
to provide comprehensive memory management for Phase 1.4.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
from torch import Tensor

from .fragment_aware_pool import FragmentAwareMemoryPool
from .bucketed_memory_pool import BucketedMemoryPool
from .numa_aware_pool import NUMAAwareMemoryPool
from .memory_profiler import get_memory_profiler

logger = logging.getLogger("dilated_attention_pytorch.enhanced_memory_pool")


class EnhancedMemoryPool:
    """
    Enhanced memory pool combining fragment-aware and bucketed allocation.

    Features:
    - Automatic strategy selection based on allocation patterns
    - Fragment-aware allocation for reducing memory fragmentation
    - Size bucketing for common allocation sizes
    - Adaptive bucket creation
    - Comprehensive monitoring and statistics
    """

    def __init__(
        self,
        enable_fragment_aware: bool = True,
        enable_bucketed: bool = True,
        enable_numa: bool = True,
        enable_profiling: bool = False,
        fragmentation_threshold: float = 0.3,
        bucket_sizes: Optional[list] = None,
        adaptive_buckets: bool = True,
    ):
        """
        Initialize the enhanced memory pool.

        Args:
            enable_fragment_aware: Enable fragment-aware allocation
            enable_bucketed: Enable bucketed allocation
            enable_numa: Enable NUMA-aware allocation
            enable_profiling: Enable memory profiling and monitoring
            fragmentation_threshold: Threshold for defragmentation
            bucket_sizes: Custom bucket sizes
            adaptive_buckets: Enable adaptive bucket creation
        """
        self.enable_fragment_aware = enable_fragment_aware
        self.enable_bucketed = enable_bucketed
        self.enable_numa = enable_numa
        self.enable_profiling = enable_profiling

        # Initialize pools
        self.fragment_pool = None
        self.bucketed_pool = None
        self.numa_pool = None

        # Initialize profiler
        self.profiler = None
        if enable_profiling:
            self.profiler = get_memory_profiler()
            if not self.profiler._profiling_active:
                self.profiler.start_profiling()

        if enable_fragment_aware:
            self.fragment_pool = FragmentAwareMemoryPool(
                fragmentation_threshold=fragmentation_threshold,
                compaction_strategy="best_fit",
            )

        if enable_bucketed:
            self.bucketed_pool = BucketedMemoryPool(
                bucket_sizes=bucket_sizes,
                adaptive_buckets=adaptive_buckets,
            )

        if enable_numa:
            self.numa_pool = NUMAAwareMemoryPool(
                enable_numa=True,
                prefer_local_allocation=True,
            )

        # Statistics
        self.stats = {
            "total_allocations": 0,
            "fragment_aware_allocations": 0,
            "bucketed_allocations": 0,
            "numa_allocations": 0,
            "fallback_allocations": 0,
        }

        # Lock for thread safety
        self._lock = threading.Lock()

        logger.info(
            f"Enhanced memory pool initialized: "
            f"fragment_aware={enable_fragment_aware}, "
            f"bucketed={enable_bucketed}, "
            f"numa_aware={enable_numa}, "
            f"profiling={enable_profiling}"
        )

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        strategy: str = "auto",
    ) -> Tensor:
        """
        Allocate memory using the best available strategy.

        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device to allocate on
            strategy: Allocation strategy ("auto", "bucketed", "fragment_aware", "fallback")

        Returns:
            Allocated tensor
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Record profiling context if enabled
        operation_name = "enhanced_pool_allocate"
        profiler_context = None
        if self.profiler:
            profiler_context = self.profiler.profile_operation(operation_name)
            profiler_context.__enter__()

        try:
            with self._lock:
                self.stats["total_allocations"] += 1

                # Calculate size in bytes
                element_size = torch.finfo(dtype).bits // 8
                size_bytes = int(torch.prod(torch.tensor(shape)).item() * element_size)

                # Auto strategy selection
                if strategy == "auto":
                    strategy = self._select_strategy(size_bytes, device)

                tensor = None
                pool_type = "unknown"
                numa_node = None

                # Try NUMA-aware allocation for large tensors or when explicitly requested
                if strategy == "numa_aware" and self.numa_pool is not None:
                    try:
                        tensor = self.numa_pool.allocate(shape, dtype, device)
                        self.stats["numa_allocations"] += 1
                        pool_type = "numa_aware"
                        # Try to get NUMA node info
                        if hasattr(self.numa_pool, "_select_numa_node"):
                            numa_node = self.numa_pool._select_numa_node(device, None)
                    except Exception as e:
                        logger.debug(f"NUMA-aware allocation failed: {e}")

                # Try bucketed allocation first for suitable sizes
                if (
                    tensor is None
                    and strategy == "bucketed"
                    and self.bucketed_pool is not None
                ):
                    try:
                        tensor = self.bucketed_pool.allocate(
                            size_bytes, dtype, device, shape
                        )
                        self.stats["bucketed_allocations"] += 1
                        pool_type = "bucketed"
                    except Exception as e:
                        logger.debug(f"Bucketed allocation failed: {e}")

                # Try fragment-aware allocation
                if (
                    tensor is None
                    and strategy == "fragment_aware"
                    and self.fragment_pool is not None
                ):
                    try:
                        tensor = self.fragment_pool.allocate(
                            size_bytes, dtype, device, shape
                        )
                        self.stats["fragment_aware_allocations"] += 1
                        pool_type = "fragment_aware"
                    except Exception as e:
                        logger.debug(f"Fragment-aware allocation failed: {e}")

                # Fallback to direct PyTorch allocation
                if tensor is None:
                    tensor = self._fallback_allocate(shape, dtype, device)
                    pool_type = "fallback"

                # Record allocation in profiler
                if self.profiler and tensor is not None:
                    self.profiler.record_allocation(
                        tensor, pool_type=pool_type, numa_node=numa_node
                    )

                return tensor

        finally:
            if profiler_context:
                profiler_context.__exit__(None, None, None)

    def deallocate(self, tensor: Tensor) -> None:
        """
        Return memory to the appropriate pool.

        Args:
            tensor: Tensor to deallocate
        """
        # Record deallocation in profiler
        if self.profiler:
            self.profiler.record_deallocation(tensor)

        with self._lock:
            # Try bucketed pool first
            if self.bucketed_pool is not None:
                try:
                    self.bucketed_pool.deallocate(tensor)
                    return
                except Exception:
                    pass

            # Try fragment-aware pool
            if self.fragment_pool is not None:
                try:
                    self.fragment_pool.deallocate(tensor)
                    return
                except Exception:
                    pass

            # If neither pool handles it, it was a fallback allocation
            # Nothing to do for fallback allocations

    def _select_strategy(self, size_bytes: int, device: torch.device) -> str:
        """
        Select the best allocation strategy based on size and device state.

        Args:
            size_bytes: Allocation size in bytes
            device: Target device

        Returns:
            Selected strategy name
        """
        # For very large allocations (>16MB), prefer NUMA-aware allocation
        if size_bytes > 16 * 1024 * 1024 and self.numa_pool is not None:
            return "numa_aware"

        # For small allocations, prefer bucketed
        if size_bytes <= 65536 and self.bucketed_pool is not None:  # <= 64KB
            return "bucketed"

        # For medium-large allocations on multi-socket systems, consider NUMA
        if size_bytes > 1024 * 1024 and self.numa_pool is not None:  # > 1MB
            # Check if NUMA is available and beneficial
            numa_stats = (
                self.numa_pool.get_stats()
                if hasattr(self.numa_pool, "get_stats")
                else {}
            )
            topology = numa_stats.get("topology", {})
            if topology.get("numa_available") and topology.get("num_nodes", 0) > 1:
                return "numa_aware"

        # For large allocations, check fragmentation
        if self.fragment_pool is not None:
            fragment_stats = self.fragment_pool.get_stats(device)
            if isinstance(fragment_stats, dict):
                frag_score = fragment_stats.get("fragmentation_score", 0.0)
                if frag_score < 0.3:  # Low fragmentation
                    return "bucketed" if self.bucketed_pool else "fragment_aware"
                else:  # High fragmentation
                    return "fragment_aware"

        # Default to bucketed if available
        return "bucketed" if self.bucketed_pool else "fallback"

    def _fallback_allocate(
        self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """Fallback allocation using PyTorch directly."""
        self.stats["fallback_allocations"] += 1
        return torch.empty(shape, dtype=dtype, device=device)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory pool statistics."""
        with self._lock:
            combined_stats = {
                "enhanced_pool": self.stats.copy(),
                "fragment_aware": None,
                "bucketed": None,
                "numa_aware": None,
            }

            if self.fragment_pool is not None:
                combined_stats["fragment_aware"] = self.fragment_pool.get_stats()

            if self.bucketed_pool is not None:
                combined_stats["bucketed"] = self.bucketed_pool.get_stats()

            if self.numa_pool is not None:
                combined_stats["numa_aware"] = self.numa_pool.get_stats()

            return combined_stats

    def get_efficiency_report(self) -> str:
        """Generate comprehensive efficiency report."""
        lines = ["Enhanced Memory Pool Report", "=" * 30, ""]

        stats = self.get_stats()
        enhanced = stats["enhanced_pool"]

        # Overall statistics
        total = enhanced["total_allocations"]
        if total > 0:
            bucketed_pct = enhanced["bucketed_allocations"] / total * 100
            fragment_pct = enhanced["fragment_aware_allocations"] / total * 100
            numa_pct = enhanced["numa_allocations"] / total * 100
            fallback_pct = enhanced["fallback_allocations"] / total * 100

            lines.extend(
                [
                    f"Total Allocations: {total:,}",
                    f"Bucketed: {enhanced['bucketed_allocations']:,} ({bucketed_pct:.1f}%)",
                    f"Fragment-Aware: {enhanced['fragment_aware_allocations']:,} ({fragment_pct:.1f}%)",
                    f"NUMA-Aware: {enhanced['numa_allocations']:,} ({numa_pct:.1f}%)",
                    f"Fallback: {enhanced['fallback_allocations']:,} ({fallback_pct:.1f}%)",
                    "",
                ]
            )

        # Fragment-aware pool stats
        if stats["fragment_aware"] is not None:
            lines.append("Fragment-Aware Pool:")
            lines.append("-" * 20)

            frag_report = self.fragment_pool.get_fragmentation_report()
            lines.extend(frag_report.split("\n")[3:])  # Skip header
            lines.append("")

        # Bucketed pool stats
        if stats["bucketed"] is not None:
            lines.append("Bucketed Pool:")
            lines.append("-" * 15)

            bucket_report = self.bucketed_pool.get_efficiency_report()
            lines.extend(bucket_report.split("\n")[7:])  # Skip header
            lines.append("")

        # NUMA-aware pool stats
        if stats["numa_aware"] is not None:
            lines.append("NUMA-Aware Pool:")
            lines.append("-" * 17)

            try:
                numa_report = self.numa_pool.get_numa_report()
                lines.extend(numa_report.split("\n")[3:])  # Skip header
            except Exception as e:
                lines.append(f"NUMA report generation failed: {e}")

        return "\n".join(lines)

    def get_profiling_report(self) -> str:
        """Get memory profiling report if profiling is enabled."""
        if not self.profiler:
            return "Memory profiling is not enabled. Enable with enable_profiling=True"

        return self.profiler.generate_report()

    def export_profiling_data(self, filepath: Path) -> None:
        """Export profiling data to file."""
        if not self.profiler:
            logger.warning("Profiling not enabled - no data to export")
            return

        self.profiler.export_data(filepath)

    def create_memory_dashboard(self, save_path: Optional[Path] = None) -> str:
        """Create interactive memory profiling dashboard."""
        if not self.profiler:
            return "<html><body><h1>Memory profiling not enabled</h1></body></html>"

        try:
            from .memory_visualizer import create_memory_dashboard

            return create_memory_dashboard(self.profiler, save_path)
        except ImportError:
            return "<html><body><h1>Visualization libraries not available</h1><p>Install with: pip install matplotlib plotly</p></body></html>"

    def defragment_all(self) -> Dict[str, bool]:
        """Trigger defragmentation on all pools that support it."""
        results = {}

        if self.fragment_pool is not None:
            try:
                frag_results = self.fragment_pool.defragment_all_devices()
                results.update(frag_results)
            except Exception as e:
                logger.warning(f"Fragment pool defragmentation failed: {e}")

        return results

    def clear(self) -> None:
        """Clear all memory pools."""
        with self._lock:
            if self.fragment_pool is not None:
                # Fragment pool doesn't have clear method, simulate it
                pass

            if self.bucketed_pool is not None:
                self.bucketed_pool.clear()

            if self.numa_pool is not None:
                # NUMA pool doesn't have clear method, reset global instance
                try:
                    from .numa_aware_pool import reset_numa_pool

                    reset_numa_pool()
                except Exception as e:
                    logger.debug(f"Failed to reset NUMA pool: {e}")

            # Reset stats
            self.stats = {
                "total_allocations": 0,
                "fragment_aware_allocations": 0,
                "bucketed_allocations": 0,
                "numa_allocations": 0,
                "fallback_allocations": 0,
            }


# Global enhanced memory pool instance
_ENHANCED_POOL: Optional[EnhancedMemoryPool] = None
_POOL_LOCK = threading.Lock()


def get_enhanced_memory_pool(**kwargs) -> EnhancedMemoryPool:
    """
    Get the global enhanced memory pool instance.

    Args:
        **kwargs: Configuration for the pool (only used on first call)

    Returns:
        Global enhanced memory pool instance
    """
    global _ENHANCED_POOL

    if _ENHANCED_POOL is None:
        with _POOL_LOCK:
            if _ENHANCED_POOL is None:
                _ENHANCED_POOL = EnhancedMemoryPool(**kwargs)

    return _ENHANCED_POOL


def reset_enhanced_memory_pool() -> None:
    """Reset the global enhanced memory pool."""
    global _ENHANCED_POOL

    with _POOL_LOCK:
        if _ENHANCED_POOL is not None:
            _ENHANCED_POOL.clear()
            _ENHANCED_POOL = None
