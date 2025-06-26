"""
Unified memory pool implementation for efficient buffer management.

This module provides a consolidated memory pool that can be used across all
dilated attention implementations, replacing multiple separate pools.
"""

import gc
import logging
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import MemoryPoolConfig

logger = logging.getLogger("dilated_attention_pytorch.memory_pool")


@dataclass
class BufferStats:
    """Statistics for a buffer in the pool."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    access_count: int = 0
    last_access_time: float = 0.0
    pinned: bool = False
    size_bytes: int = 0

    def __post_init__(self):
        """Calculate size in bytes."""
        self.size_bytes = (
            torch.prod(torch.tensor(self.shape)).item() * torch.finfo(self.dtype).bits // 8
        )


class UnifiedMemoryPool:
    """
    Unified memory pool for efficient buffer management across all attention types.

    Features:
    - Adaptive cleanup based on memory pressure
    - Hot buffer cache for frequently accessed patterns
    - Support for pinned memory
    - Thread-safe operations
    - Automatic garbage collection integration
    - Buffer statistics tracking
    - Multiple pool strategies (default, ring, sparse)

    Args:
        config: Memory pool configuration
    """

    def __init__(self, config: MemoryPoolConfig | None = None):
        """Initialize the unified memory pool."""
        self.config = config or MemoryPoolConfig()

        # Separate pools for different strategies
        self._pools: dict[str, OrderedDict[tuple, Tensor]] = {
            "default": OrderedDict(),
            "ring": OrderedDict(),
            "sparse": OrderedDict(),
            "distributed": OrderedDict(),
        }

        # Buffer statistics
        self._stats: dict[tuple, BufferStats] = {}

        # Hot cache for frequently accessed buffers
        self._hot_cache: OrderedDict[tuple, Tensor] = OrderedDict()

        # Lock for thread safety
        self._lock = threading.RLock()

        # Cleanup thresholds
        self._aggressive_threshold = 0.1  # 10% free memory
        self._conservative_threshold = 0.5  # 50% free memory

        # Track total allocated memory
        self._total_allocated_bytes = 0

        # Weak references to track buffer usage
        # Use WeakSet instead of WeakValueDictionary to avoid circular references
        self._active_buffers = weakref.WeakSet()

        # Register with garbage collector
        self._register_gc_callback()

        logger.debug(f"Initialized UnifiedMemoryPool with config: {config}")

    def get_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        pinned: bool = False,
        pool_type: str = "default",
    ) -> Tensor:
        """
        Get a buffer from the pool or allocate a new one.

        Args:
            shape: Shape of the buffer
            dtype: Data type
            device: Device to allocate on
            pinned: Whether to use pinned memory
            pool_type: Type of pool to use

        Returns:
            Buffer tensor
        """
        with self._lock:
            # Validate pool type
            if pool_type not in self._pools:
                pool_type = "default"

            # Create key for this buffer specification
            key = (shape, dtype, device, pinned, pool_type)

            # Check hot cache first
            if key in self._hot_cache:
                buffer = self._hot_cache[key]
                self._update_stats(key, buffer)
                return buffer

            # Check main pool
            pool = self._pools[pool_type]
            if key in pool:
                buffer = pool[key]
                # Move to end for LRU
                pool.move_to_end(key)
                # Update stats first so promotion logic sees the correct count
                self._update_stats(key, buffer)
                # Promote to hot cache if frequently accessed
                self._maybe_promote_to_hot_cache(key, buffer)
                return buffer

            # Try to find a compatible buffer
            buffer = self._find_compatible_buffer(shape, dtype, device, pool)
            if buffer is not None:
                self._update_stats(key, buffer)
                return buffer

            # Allocate new buffer
            buffer = self._allocate_new_buffer(shape, dtype, device, pinned)

            # Add to pool
            pool[key] = buffer
            self._track_buffer(key, buffer)
            self._update_stats(key, buffer)

            # Check memory pressure and clean if needed
            self._maybe_cleanup()

            return buffer

    def _find_compatible_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | None,
        pool: OrderedDict,
    ) -> Tensor | None:
        """
        Find a compatible buffer that can be reshaped or sliced.

        Args:
            shape: Desired shape
            dtype: Desired dtype
            device: Desired device
            pool: Pool to search in

        Returns:
            Compatible buffer or None
        """
        target_numel = torch.prod(torch.tensor(shape)).item()

        for (buf_shape, buf_dtype, buf_device, _, _), buffer in pool.items():
            # Check dtype and device match
            if buf_dtype != dtype or buf_device != device:
                continue

            buf_numel = buffer.numel()

            # Exact match in number of elements - can reshape
            if buf_numel == target_numel:
                try:
                    reshaped = buffer.view(shape)
                    return reshaped
                except RuntimeError:
                    continue

            # Buffer is larger - can slice
            if buf_numel > target_numel and self.config.allow_buffer_slicing:
                flat_buffer = buffer.flatten()
                sliced = flat_buffer[:target_numel].view(shape)
                return sliced

        return None

    def _allocate_new_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | None,
        pinned: bool,
    ) -> Tensor:
        """Allocate a new buffer with the specified properties."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if pinned and device.type == "cuda":
                # Allocate pinned memory on CPU then move to GPU
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
                buffer = buffer.to(device, non_blocking=True)
            else:
                buffer = torch.empty(shape, dtype=dtype, device=device)

            # Track allocation
            size_bytes = buffer.numel() * buffer.element_size()
            self._total_allocated_bytes += size_bytes

            return buffer

        except torch.cuda.OutOfMemoryError:
            # Try cleanup and retry
            logger.warning("CUDA OOM during allocation, attempting cleanup")
            self._aggressive_cleanup()

            # Retry allocation
            if pinned and device.type == "cuda":
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
                buffer = buffer.to(device, non_blocking=True)
            else:
                buffer = torch.empty(shape, dtype=dtype, device=device)

            return buffer

    def _update_stats(self, key: tuple, buffer: Tensor) -> None:
        """Update statistics for buffer access."""
        import time

        if key not in self._stats:
            self._stats[key] = BufferStats(
                shape=key[0],
                dtype=key[1],
                device=key[2] or buffer.device,
                pinned=key[3],
            )

        stats = self._stats[key]
        stats.access_count += 1
        stats.last_access_time = time.time()

    def _maybe_promote_to_hot_cache(self, key: tuple, buffer: Tensor) -> None:
        """Promote frequently accessed buffers to hot cache."""
        if key not in self._stats:
            return

        stats = self._stats[key]

        # Promote if accessed frequently
        if stats.access_count >= self.config.hot_cache_threshold:
            self._hot_cache[key] = buffer

            # Limit hot cache size
            while len(self._hot_cache) > self.config.hot_cache_size:
                self._hot_cache.popitem(last=False)

    def _track_buffer(self, key: tuple, buffer: Tensor) -> None:
        """Track buffer with weak reference."""
        self._active_buffers.add(buffer)

        # Initialize stats
        if key not in self._stats:
            self._stats[key] = BufferStats(
                shape=key[0],
                dtype=key[1],
                device=key[2] or buffer.device,
                pinned=key[3],
            )

    def _maybe_cleanup(self) -> None:
        """Check memory pressure and cleanup if needed."""
        if not torch.cuda.is_available():
            return

        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory

            free_ratio = 1.0 - (allocated / total)

            # Determine cleanup strategy
            if free_ratio < self._aggressive_threshold:
                logger.info(f"Low memory ({free_ratio:.1%} free), aggressive cleanup")
                self._aggressive_cleanup()
            elif free_ratio < self._conservative_threshold:
                logger.debug(
                    f"Moderate memory pressure ({free_ratio:.1%} free), conservative cleanup"
                )
                self._conservative_cleanup()

        except Exception as e:
            logger.warning(f"Error checking memory pressure: {e}")

    def _aggressive_cleanup(self) -> None:
        """Aggressive cleanup - remove most unused buffers."""
        with self._lock:
            # Clear hot cache
            self._hot_cache.clear()

            # Clear all pools, keeping only active buffers
            for pool_name, pool in self._pools.items():
                # Get list of keys to remove
                keys_to_remove = []
                for key, buffer in pool.items():
                    if buffer not in self._active_buffers:
                        keys_to_remove.append(key)

                # Remove inactive buffers
                for key in keys_to_remove:
                    buffer = pool.pop(key)
                    size_bytes = buffer.numel() * buffer.element_size()
                    self._total_allocated_bytes -= size_bytes
                    del buffer

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            logger.info("Aggressive cleanup complete, freed memory")

    def _conservative_cleanup(self) -> None:
        """Conservative cleanup - remove old unused buffers."""
        import time

        current_time = time.time()

        with self._lock:
            for pool_name, pool in self._pools.items():
                # Remove buffers not accessed recently
                keys_to_remove = []
                for key in pool:
                    if key in self._stats:
                        stats = self._stats[key]
                        time_since_access = current_time - stats.last_access_time

                        # Remove if not accessed in last minute and not in hot cache
                        if (
                            time_since_access > 60
                            and key not in self._hot_cache
                            and pool[key] not in self._active_buffers
                        ):
                            keys_to_remove.append(key)

                # Remove old buffers
                for key in keys_to_remove:
                    buffer = pool.pop(key)
                    size_bytes = buffer.numel() * buffer.element_size()
                    self._total_allocated_bytes -= size_bytes
                    del buffer

    def clear_pool(self, pool_type: str | None = None) -> None:
        """
        Clear buffers from specified pool or all pools.

        Args:
            pool_type: Specific pool to clear, or None for all
        """
        with self._lock:
            if pool_type is None:
                # Clear all pools
                for pool in self._pools.values():
                    pool.clear()
                self._hot_cache.clear()
                self._stats.clear()
                self._total_allocated_bytes = 0
            elif pool_type in self._pools:
                # Clear specific pool
                self._pools[pool_type].clear()

            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_stats(self) -> dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_buffers = sum(len(pool) for pool in self._pools.values())
            hot_cache_size = len(self._hot_cache)

            pool_sizes = {name: len(pool) for name, pool in self._pools.items()}

            # Get memory usage by pool
            memory_by_pool = {}
            for pool_name, pool in self._pools.items():
                pool_memory = 0
                for buffer in pool.values():
                    pool_memory += buffer.numel() * buffer.element_size()
                memory_by_pool[pool_name] = pool_memory

            return {
                "total_buffers": total_buffers,
                "hot_cache_size": hot_cache_size,
                "pool_sizes": pool_sizes,
                "total_allocated_bytes": self._total_allocated_bytes,
                "memory_by_pool": memory_by_pool,
                "active_buffers": len(self._active_buffers),
            }

    def _register_gc_callback(self) -> None:
        """Register callback with garbage collector."""

        def gc_callback(phase, info):
            if phase == "stop" and info["collected"] > 0:
                # Some objects were collected, check our buffers
                self._maybe_cleanup()

        # Only available in Python 3.8+
        if hasattr(gc, "callbacks"):
            gc.callbacks.append(gc_callback)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.clear_pool()
        except Exception:
            pass

    def __getstate__(self):
        """Support for pickling - exclude unpickleable objects."""
        state = self.__dict__.copy()
        # Remove the unpickleable locks
        state["_lock"] = None
        # Clear the pools to avoid pickling large tensors
        state["_pools"] = {}
        state["_hot_cache"] = {}
        state["_active_buffers"] = {}
        return state

    def __setstate__(self, state):
        """Support for unpickling - recreate locks."""
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = threading.RLock()


# Global memory pool instance (created lazily)
_GLOBAL_POOL: UnifiedMemoryPool | None = None
_POOL_LOCK = threading.Lock()


def get_global_memory_pool(config: MemoryPoolConfig | None = None) -> UnifiedMemoryPool:
    """
    Get the global memory pool instance.

    Args:
        config: Configuration for the pool (only used on first call)

    Returns:
        Global memory pool instance
    """
    global _GLOBAL_POOL

    if _GLOBAL_POOL is None:
        with _POOL_LOCK:
            if _GLOBAL_POOL is None:
                _GLOBAL_POOL = UnifiedMemoryPool(config)

    return _GLOBAL_POOL


def reset_global_memory_pool() -> None:
    """Reset the global memory pool."""
    global _GLOBAL_POOL

    with _POOL_LOCK:
        if _GLOBAL_POOL is not None:
            _GLOBAL_POOL.clear_pool()
            _GLOBAL_POOL = None
