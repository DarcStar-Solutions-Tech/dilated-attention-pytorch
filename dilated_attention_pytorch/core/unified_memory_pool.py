"""
Unified memory pool implementation consolidating all memory pool features.

This module consolidates the functionality from:
- memory_pool.py (UnifiedMemoryPool)
- enhanced_memory_pool.py (EnhancedMemoryPool)
- bucketed_memory_pool.py (BucketedMemoryPool)
- fragment_aware_pool.py (FragmentAwareMemoryPool)
- numa_aware_pool.py (NUMAAwareMemoryPool)
"""

import gc
import logging
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class MemoryPoolConfig:
    """Configuration for unified memory pool."""

    # Core settings
    enable_bucketing: bool = True
    enable_numa_awareness: bool = False  # Disabled by default (rarely needed)
    enable_fragmentation_tracking: bool = False  # Disabled by default (overhead)
    enable_profiling: bool = False

    # Memory management
    max_cached_tensors: int = 100
    cleanup_threshold_mb: float = 100.0
    fragmentation_threshold: float = 0.3

    # Bucketing settings
    bucket_sizes_mb: List[float] = None  # Auto-determined if None
    adaptive_buckets: bool = True

    # Device settings
    device: Optional[torch.device] = None

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.bucket_sizes_mb is None:
            # Common sizes for transformer workloads (in MB)
            self.bucket_sizes_mb = [0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplifiedMemoryPool:
    """
    Simplified unified memory pool with essential features.

    This implementation focuses on the most useful features while
    eliminating complexity and redundancy from the 5 separate pools.
    """

    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        """Initialize the unified memory pool."""
        self.config = config or MemoryPoolConfig()
        self._lock = threading.RLock()

        # Core storage
        self._free_tensors: Dict[Tuple[torch.Size, torch.dtype], List[Tensor]] = (
            defaultdict(list)
        )
        self._allocated_tensors: weakref.WeakSet = weakref.WeakSet()

        # Bucketing (if enabled)
        if self.config.enable_bucketing:
            self._buckets = self._init_buckets()
            self._bucket_stats = defaultdict(lambda: {"hits": 0, "misses": 0})

        # Statistics
        self._stats = {
            "allocations": 0,
            "deallocations": 0,
            "reuses": 0,
            "cleanups": 0,
            "current_allocated_mb": 0.0,
            "peak_allocated_mb": 0.0,
        }

        # Register cleanup with garbage collector
        gc_callback = weakref.finalize(self, self._cleanup_all)
        gc_callback.atexit = True

    def _init_buckets(self) -> Dict[float, List[Tensor]]:
        """Initialize memory buckets for common sizes."""
        buckets = {}
        for size_mb in self.config.bucket_sizes_mb:
            buckets[size_mb] = []
        return buckets

    def allocate(
        self,
        size: Union[torch.Size, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Allocate a tensor from the pool.

        Args:
            size: Tensor size
            dtype: Data type
            device: Device (uses config default if None)

        Returns:
            Allocated tensor
        """
        with self._lock:
            device = device or self.config.device
            size = torch.Size(size)
            key = (size, dtype)

            # Try to reuse from free pool
            if key in self._free_tensors and self._free_tensors[key]:
                tensor = self._free_tensors[key].pop()
                self._stats["reuses"] += 1
                logger.debug(f"Reused tensor {size} {dtype}")
            else:
                # Check buckets if enabled
                if self.config.enable_bucketing:
                    tensor = self._allocate_from_bucket(size, dtype, device)
                    if tensor is not None:
                        self._stats["reuses"] += 1
                        return tensor

                # Allocate new tensor
                try:
                    tensor = torch.empty(size, dtype=dtype, device=device)
                    self._stats["allocations"] += 1
                    logger.debug(f"Allocated new tensor {size} {dtype}")
                except torch.cuda.OutOfMemoryError:
                    # Emergency cleanup and retry
                    self._emergency_cleanup()
                    tensor = torch.empty(size, dtype=dtype, device=device)
                    self._stats["allocations"] += 1

            # Track allocation
            self._allocated_tensors.add(tensor)
            self._update_memory_stats()

            return tensor

    def deallocate(self, tensor: Tensor) -> None:
        """
        Return a tensor to the pool.

        Args:
            tensor: Tensor to deallocate
        """
        if tensor is None:
            return

        with self._lock:
            size = tensor.size()
            dtype = tensor.dtype
            key = (size, dtype)

            # Add to free pool
            self._free_tensors[key].append(tensor)
            self._stats["deallocations"] += 1

            # Check if cleanup needed
            if self._should_cleanup():
                self._cleanup()

    def _allocate_from_bucket(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Allocate from size buckets if possible."""
        if not self.config.enable_bucketing:
            return None

        # Calculate size in MB
        size_mb = (torch.tensor(size).prod().item() * torch.finfo(dtype).bits) / (
            8 * 1024 * 1024
        )

        # Find appropriate bucket
        for bucket_size in sorted(self._buckets.keys()):
            if bucket_size >= size_mb and self._buckets[bucket_size]:
                # Found a suitable bucket with available tensors
                candidates = self._buckets[bucket_size]
                for i, tensor in enumerate(candidates):
                    if tensor.dtype == dtype and tensor.device == device:
                        # Reshape if needed
                        if tensor.numel() >= torch.tensor(size).prod().item():
                            candidates.pop(i)
                            self._bucket_stats[bucket_size]["hits"] += 1
                            return tensor[: torch.tensor(size).prod().item()].reshape(
                                size
                            )

        self._bucket_stats[bucket_size]["misses"] += 1
        return None

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        # Simple heuristic: cleanup if we have too many free tensors
        total_free = sum(len(tensors) for tensors in self._free_tensors.values())
        return total_free > self.config.max_cached_tensors

    def _cleanup(self) -> None:
        """Clean up excess cached tensors."""
        with self._lock:
            total_freed = 0

            # Keep only the most recent tensors for each key
            for key, tensors in list(self._free_tensors.items()):
                if len(tensors) > 5:  # Keep at most 5 per size/dtype
                    freed = tensors[:-5]
                    self._free_tensors[key] = tensors[-5:]
                    total_freed += len(freed)

            # Clear buckets if needed
            if self.config.enable_bucketing:
                for bucket_size, tensors in self._buckets.items():
                    if len(tensors) > 2:
                        self._buckets[bucket_size] = tensors[-2:]
                        total_freed += len(tensors) - 2

            self._stats["cleanups"] += 1
            logger.debug(f"Cleaned up {total_freed} tensors")

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup on OOM."""
        logger.warning("Emergency cleanup triggered due to OOM")
        with self._lock:
            # Clear all free tensors
            self._free_tensors.clear()

            # Clear all buckets
            if self.config.enable_bucketing:
                for bucket in self._buckets.values():
                    bucket.clear()

            # Force garbage collection
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _cleanup_all(self) -> None:
        """Clean up all resources (called on deletion)."""
        with self._lock:
            self._free_tensors.clear()
            if hasattr(self, "_buckets"):
                self._buckets.clear()

    def _update_memory_stats(self) -> None:
        """Update memory statistics."""
        if not self.config.enable_profiling:
            return

        current_mb = 0.0

        # Calculate from free tensors
        for tensors in self._free_tensors.values():
            for tensor in tensors:
                current_mb += tensor.element_size() * tensor.numel() / (1024 * 1024)

        self._stats["current_allocated_mb"] = current_mb
        self._stats["peak_allocated_mb"] = max(
            self._stats["peak_allocated_mb"], current_mb
        )

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get memory pool statistics."""
        with self._lock:
            stats = self._stats.copy()

            # Add cache info
            total_cached = sum(len(tensors) for tensors in self._free_tensors.values())
            stats["cached_tensors"] = total_cached

            if self.config.enable_bucketing:
                stats["bucket_stats"] = dict(self._bucket_stats)

            return stats

    def reset(self) -> None:
        """Reset the memory pool, clearing all cached tensors."""
        with self._lock:
            self._free_tensors.clear()
            if self.config.enable_bucketing:
                for bucket in self._buckets.values():
                    bucket.clear()

            # Reset stats
            self._stats = {
                "allocations": 0,
                "deallocations": 0,
                "reuses": 0,
                "cleanups": 0,
                "current_allocated_mb": 0.0,
                "peak_allocated_mb": 0.0,
            }

            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global instance
_global_pool: Optional[SimplifiedMemoryPool] = None
_pool_lock = threading.Lock()


def get_global_memory_pool(
    config: Optional[MemoryPoolConfig] = None,
) -> SimplifiedMemoryPool:
    """Get or create the global memory pool."""
    global _global_pool

    with _pool_lock:
        if _global_pool is None:
            _global_pool = SimplifiedMemoryPool(config)
        elif config is not None:
            logger.warning("Global memory pool already exists, ignoring new config")

        return _global_pool


def reset_global_memory_pool() -> None:
    """Reset the global memory pool."""
    global _global_pool

    with _pool_lock:
        if _global_pool is not None:
            _global_pool.reset()
            _global_pool = None


# Compatibility aliases for existing code
UnifiedMemoryPool = SimplifiedMemoryPool
MemoryPool = SimplifiedMemoryPool


__all__ = [
    "SimplifiedMemoryPool",
    "MemoryPoolConfig",
    "get_global_memory_pool",
    "reset_global_memory_pool",
    "UnifiedMemoryPool",  # Compatibility
    "MemoryPool",  # Compatibility
]
