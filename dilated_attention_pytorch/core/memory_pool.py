"""
Unified memory pool implementation for efficient buffer management.

This module provides a consolidated memory pool that can be used across all
dilated attention implementations, replacing multiple separate pools.

DEPRECATED: This implementation is scheduled for removal in v0.4.0.
Please use unified_memory_pool.py instead.
"""
# ruff: noqa: PLR0912

import gc
import logging
import math
import threading
import warnings
import weakref
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import MemoryPoolConfig

# Issue deprecation warning
warnings.warn(
    "memory_pool.py is deprecated and will be removed in v0.4.0. "
    "Please use unified_memory_pool.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
    fragmentation_score: float = 0.0
    size_bucket: int = 0

    def __post_init__(self):
        """Calculate size in bytes and fragmentation score."""
        self.size_bytes = (
            torch.prod(torch.tensor(self.shape)).item()
            * torch.finfo(self.dtype).bits
            // 8
        )
        self.size_bucket = self._calculate_size_bucket()
        self.fragmentation_score = self._calculate_fragmentation_score()

    def _calculate_size_bucket(self) -> int:
        """Calculate size bucket for efficient allocation."""
        # Use power-of-2 buckets: 0=1KB, 1=2KB, 2=4KB, etc.
        if self.size_bytes <= 0:
            return 0
        return max(0, int(math.log2(self.size_bytes / 1024)) + 10)

    def _calculate_fragmentation_score(self) -> float:
        """Calculate fragmentation score based on shape irregularity."""
        # More irregular shapes have higher fragmentation scores
        if not self.shape:
            return 0.0

        # Calculate shape variance as fragmentation indicator
        shape_array = torch.tensor(self.shape, dtype=torch.float32)
        if len(shape_array) <= 1:
            return 0.0

        mean_dim = torch.mean(shape_array)
        variance = torch.var(shape_array)

        # Normalize by mean to get relative fragmentation
        if mean_dim > 0:
            return (variance / (mean_dim**2)).item()
        return 0.0


@dataclass
class MemoryFragment:
    """Represents a memory fragment for tracking."""

    size_bytes: int
    device: torch.device
    dtype: torch.dtype
    is_free: bool = True
    allocation_time: float = 0.0
    last_access_time: float = 0.0


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
    - Fragment-aware allocation and defragmentation
    - Size bucketing for efficient allocation

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

        # Size-bucketed pools for efficient allocation
        self._size_buckets: dict[int, OrderedDict[tuple, Tensor]] = defaultdict(
            OrderedDict
        )

        # Fragment tracking for defragmentation
        self._fragments: dict[torch.device, list[MemoryFragment]] = defaultdict(list)
        self._fragmentation_threshold = 0.3  # 30% fragmentation triggers defrag

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

        # Fragmentation tracking
        self._fragmentation_scores: dict[torch.device, float] = defaultdict(float)

        # NUMA awareness (if available)
        self._numa_nodes: dict[int, list[torch.device]] = {}
        self._detect_numa_topology()

        # Weak references to track buffer usage
        # Use WeakSet instead of WeakValueDictionary to avoid circular references
        self._active_buffers = weakref.WeakSet()

        # Register with garbage collector
        self._register_gc_callback()

        logger.debug(f"Initialized UnifiedMemoryPool with config: {config}")

    def _detect_numa_topology(self) -> None:
        """Detect NUMA topology for multi-socket systems."""
        try:
            # Try to detect NUMA nodes (requires psutil)
            import psutil  # noqa: PLC0415

            if hasattr(psutil, "cpu_count"):
                # Basic NUMA detection - assume one device per NUMA node
                numa_nodes = 1
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    numa_nodes = min(
                        device_count, 2
                    )  # Most systems have 1-2 NUMA nodes

                for node in range(numa_nodes):
                    devices = []
                    if torch.cuda.is_available():
                        # Distribute CUDA devices across NUMA nodes
                        for device_id in range(torch.cuda.device_count()):
                            if device_id % numa_nodes == node:
                                devices.append(torch.device(f"cuda:{device_id}"))

                    if not devices and node == 0:  # At least one node with CPU
                        devices.append(torch.device("cpu"))

                    if devices:
                        self._numa_nodes[node] = devices

        except ImportError:
            # Fallback: single NUMA node
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.extend(
                    torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
                )
            self._numa_nodes[0] = devices

        logger.debug(f"Detected NUMA topology: {self._numa_nodes}")

    def _get_optimal_device(
        self, preferred_device: torch.device | None
    ) -> torch.device:
        """Get optimal device considering NUMA topology."""
        if preferred_device is not None:
            return preferred_device

        # Find least loaded NUMA node
        if self._numa_nodes:
            min_load = float("inf")
            best_device = None

            for node, devices in self._numa_nodes.items():
                for device in devices:
                    # Simple load metric: number of allocated buffers
                    load = sum(
                        1 for stats in self._stats.values() if stats.device == device
                    )
                    if load < min_load:
                        min_load = load
                        best_device = device

            if best_device:
                return best_device

        # Fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        pinned: bool = False,
        pool_type: str = "default",
    ) -> Tensor:
        """
        Get a buffer from the pool or allocate a new one with fragment-aware allocation.

        Args:
            shape: Shape of the buffer
            dtype: Data type
            device: Device to allocate on (NUMA-aware if None)
            pinned: Whether to use pinned memory
            pool_type: Type of pool to use

        Returns:
            Buffer tensor
        """
        with self._lock:
            # Optimize device selection using NUMA awareness
            device = self._get_optimal_device(device)

            # Validate pool type
            if pool_type not in self._pools:
                pool_type = "default"

            # Create key for this buffer specification
            key = (shape, dtype, device, pinned, pool_type)

            # Calculate size bucket for efficient lookup
            size_bytes = (
                torch.prod(torch.tensor(shape)).item() * torch.finfo(dtype).bits // 8
            )
            size_bucket = (
                max(0, int(math.log2(size_bytes / 1024)) + 10) if size_bytes > 0 else 0
            )

            # Check hot cache first
            if key in self._hot_cache:
                buffer = self._hot_cache[key]
                self._update_stats(key, buffer)
                return buffer

            # Check size bucket first for exact matches
            bucket_pool = self._size_buckets[size_bucket]
            if key in bucket_pool:
                buffer = bucket_pool[key]
                bucket_pool.move_to_end(key)
                self._update_stats(key, buffer)
                self._maybe_promote_to_hot_cache(key, buffer)
                return buffer

            # Check main pool
            pool = self._pools[pool_type]
            if key in pool:
                buffer = pool[key]
                # Move to end for LRU
                pool.move_to_end(key)
                # Also add to size bucket for future lookups
                bucket_pool[key] = buffer
                # Update stats first so promotion logic sees the correct count
                self._update_stats(key, buffer)
                # Promote to hot cache if frequently accessed
                self._maybe_promote_to_hot_cache(key, buffer)
                return buffer

            # Try to find a compatible buffer using fragment-aware search
            buffer = self._find_compatible_buffer_with_fragmentation(
                shape, dtype, device, pool, size_bucket
            )
            if buffer is not None:
                self._update_stats(key, buffer)
                return buffer

            # Check if defragmentation might help
            if self._should_defragment(device):
                self._defragment_device(device)
                # Retry after defragmentation
                buffer = self._find_compatible_buffer_with_fragmentation(
                    shape, dtype, device, pool, size_bucket
                )
                if buffer is not None:
                    self._update_stats(key, buffer)
                    return buffer

            # Allocate new buffer
            buffer = self._allocate_new_buffer(shape, dtype, device, pinned)

            # Add to both pools
            pool[key] = buffer
            bucket_pool[key] = buffer
            self._track_buffer(key, buffer)
            self._update_stats(key, buffer)

            # Update fragmentation tracking
            self._track_allocation(device, size_bytes)

            # Check memory pressure and clean if needed
            self._maybe_cleanup()

            return buffer

    def _find_compatible_buffer_with_fragmentation(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | None,
        pool: OrderedDict,
        size_bucket: int,
    ) -> Tensor | None:
        """
        Find a compatible buffer considering fragmentation scores.

        Args:
            shape: Desired shape
            dtype: Desired dtype
            device: Desired device
            pool: Pool to search in
            size_bucket: Size bucket for the request

        Returns:
            Compatible buffer or None
        """
        target_numel = torch.prod(torch.tensor(shape)).item()
        best_buffer = None
        best_score = float("inf")

        # Search in nearby size buckets first
        for bucket_offset in range(-2, 3):  # Check buckets ±2 from target
            search_bucket = max(0, size_bucket + bucket_offset)
            bucket_pool = self._size_buckets[search_bucket]

            for (buf_shape, buf_dtype, buf_device, _, _), buffer in bucket_pool.items():
                # Check dtype and device match
                if buf_dtype != dtype or buf_device != device:
                    continue

                buf_numel = buffer.numel()

                # Calculate fragmentation score for this buffer
                key = (buf_shape, buf_dtype, buf_device, False, "default")
                frag_score = 0.0
                if key in self._stats:
                    frag_score = self._stats[key].fragmentation_score

                # Exact match in number of elements - can reshape
                if buf_numel == target_numel:
                    try:
                        reshaped = buffer.view(shape)
                        # Prefer buffers with lower fragmentation
                        if frag_score < best_score:
                            best_buffer = reshaped
                            best_score = frag_score
                    except RuntimeError:
                        continue

                # Buffer is larger - can slice
                elif buf_numel > target_numel and self.config.allow_buffer_slicing:
                    # Add penalty for slicing to encourage exact matches
                    slice_penalty = 0.1
                    total_score = frag_score + slice_penalty

                    if total_score < best_score:
                        try:
                            flat_buffer = buffer.flatten()
                            sliced = flat_buffer[:target_numel].view(shape)
                            best_buffer = sliced
                            best_score = total_score
                        except RuntimeError:
                            continue

        # Fall back to original method if no fragmentation-aware match found
        if best_buffer is None:
            return self._find_compatible_buffer(shape, dtype, device, pool)

        return best_buffer

    def _should_defragment(self, device: torch.device) -> bool:
        """Check if device memory should be defragmented."""
        fragmentation_score = self._fragmentation_scores.get(device, 0.0)
        return fragmentation_score > self._fragmentation_threshold

    def _defragment_device(self, device: torch.device) -> None:
        """Perform memory defragmentation for a specific device."""
        logger.info(f"Defragmenting memory for device {device}")

        # Strategy: Remove all fragments and compact active buffers
        with self._lock:
            # Clear fragments for this device
            if device in self._fragments:
                self._fragments[device].clear()

            # Force CUDA cache cleanup to defragment GPU memory
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Also synchronize to ensure all operations complete
                torch.cuda.synchronize(device)

            # Reset fragmentation score
            self._fragmentation_scores[device] = 0.0

            # Trigger garbage collection
            gc.collect()

    def _track_allocation(self, device: torch.device, size_bytes: int) -> None:
        """Track memory allocation for fragmentation analysis."""
        import time

        # Add fragment record
        fragment = MemoryFragment(
            size_bytes=size_bytes,
            device=device,
            dtype=torch.float32,  # Default for tracking
            is_free=False,
            allocation_time=time.time(),
            last_access_time=time.time(),
        )

        self._fragments[device].append(fragment)

        # Update fragmentation score
        self._update_fragmentation_score(device)

    def _update_fragmentation_score(self, device: torch.device) -> None:
        """Update fragmentation score for a device."""
        fragments = self._fragments.get(device, [])
        if not fragments:
            self._fragmentation_scores[device] = 0.0
            return

        # Calculate fragmentation as ratio of free/allocated fragments
        free_fragments = sum(1 for f in fragments if f.is_free)
        total_fragments = len(fragments)

        if total_fragments == 0:
            score = 0.0
        else:
            # Higher score = more fragmented
            score = free_fragments / total_fragments

            # Add penalty for having many small fragments
            avg_size = sum(f.size_bytes for f in fragments) / total_fragments
            if avg_size < 1024 * 1024:  # Less than 1MB average
                score += 0.1

        self._fragmentation_scores[device] = score

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
        """Aggressive cleanup - remove most unused buffers and defragment."""
        with self._lock:
            # Clear hot cache
            self._hot_cache.clear()

            # Clear all pools, keeping only active buffers
            keys_to_remove_from_buckets = []

            for pool_name, pool in self._pools.items():
                # Get list of keys to remove
                keys_to_remove = []
                for key, buffer in pool.items():
                    # Check if buffer is in active buffers (safely handle WeakSet)
                    is_active = False
                    try:
                        is_active = buffer in self._active_buffers
                    except RuntimeError:
                        # Handle tensor comparison issues in WeakSet
                        # Assume inactive if we can't check safely
                        is_active = False

                    if not is_active:
                        keys_to_remove.append(key)

                # Remove inactive buffers
                for key in keys_to_remove:
                    buffer = pool.pop(key)
                    size_bytes = buffer.numel() * buffer.element_size()
                    self._total_allocated_bytes -= size_bytes

                    # Mark corresponding fragments as free
                    device = key[2] if key[2] else buffer.device
                    for fragment in self._fragments.get(device, []):
                        if fragment.size_bytes == size_bytes and not fragment.is_free:
                            fragment.is_free = True
                            break

                    # Also remove from size buckets
                    keys_to_remove_from_buckets.append(key)
                    del buffer

            # Clean up size buckets
            for key in keys_to_remove_from_buckets:
                for bucket_pool in self._size_buckets.values():
                    bucket_pool.pop(key, None)

            # Defragment all devices with high fragmentation
            devices_defragmented = self.defragment_all_devices()

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"Aggressive cleanup complete, freed memory and defragmented {sum(devices_defragmented.values())} devices"
            )

    def _conservative_cleanup(self) -> None:
        """Conservative cleanup - remove old unused buffers and update fragmentation."""
        import time

        current_time = time.time()

        with self._lock:
            keys_to_remove_from_buckets = []

            for pool_name, pool in self._pools.items():
                # Remove buffers not accessed recently
                keys_to_remove = []
                for key in pool:
                    if key in self._stats:
                        stats = self._stats[key]
                        time_since_access = current_time - stats.last_access_time

                        # Check if buffer is active (safely handle WeakSet)
                        is_active = False
                        try:
                            is_active = pool[key] in self._active_buffers
                        except RuntimeError:
                            # Handle tensor comparison issues in WeakSet
                            is_active = False

                        # Remove if not accessed in last minute and not in hot cache
                        if (
                            time_since_access > 60
                            and key not in self._hot_cache
                            and not is_active
                        ):
                            keys_to_remove.append(key)

                # Remove old buffers
                for key in keys_to_remove:
                    buffer = pool.pop(key)
                    size_bytes = buffer.numel() * buffer.element_size()
                    self._total_allocated_bytes -= size_bytes

                    # Mark corresponding fragments as free
                    device = key[2] if key[2] else buffer.device
                    for fragment in self._fragments.get(device, []):
                        if fragment.size_bytes == size_bytes and not fragment.is_free:
                            fragment.is_free = True
                            break

                    # Update fragmentation score for this device
                    self._update_fragmentation_score(device)

                    keys_to_remove_from_buckets.append(key)
                    del buffer

            # Clean up size buckets
            for key in keys_to_remove_from_buckets:
                for bucket_pool in self._size_buckets.values():
                    bucket_pool.pop(key, None)

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
                self._size_buckets.clear()
                self._hot_cache.clear()
                self._stats.clear()
                self._fragments.clear()
                self._fragmentation_scores.clear()
                self._total_allocated_bytes = 0
            elif pool_type in self._pools:
                # Clear specific pool
                self._pools[pool_type].clear()
                # Also clear corresponding entries from buckets
                keys_to_remove = []
                for bucket, bucket_pool in self._size_buckets.items():
                    for key in list(bucket_pool.keys()):
                        if key[4] == pool_type:  # pool_type is 5th element in key
                            keys_to_remove.append((bucket, key))

                for bucket, key in keys_to_remove:
                    self._size_buckets[bucket].pop(key, None)

            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive memory pool statistics including fragmentation info."""
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

            # Size bucket statistics
            bucket_stats = {}
            for bucket, bucket_pool in self._size_buckets.items():
                bucket_stats[f"bucket_{bucket}"] = len(bucket_pool)

            # Fragmentation statistics
            fragmentation_stats = {}
            for device, score in self._fragmentation_scores.items():
                fragmentation_stats[str(device)] = {
                    "fragmentation_score": score,
                    "fragments_count": len(self._fragments.get(device, [])),
                    "needs_defrag": score > self._fragmentation_threshold,
                }

            # NUMA statistics
            numa_stats = {}
            for node, devices in self._numa_nodes.items():
                node_buffers = 0
                for device in devices:
                    node_buffers += sum(
                        1 for stats in self._stats.values() if stats.device == device
                    )
                numa_stats[f"numa_node_{node}"] = {
                    "devices": [str(d) for d in devices],
                    "buffers": node_buffers,
                }

            return {
                "total_buffers": total_buffers,
                "hot_cache_size": hot_cache_size,
                "pool_sizes": pool_sizes,
                "bucket_stats": bucket_stats,
                "total_allocated_bytes": self._total_allocated_bytes,
                "memory_by_pool": memory_by_pool,
                "active_buffers": len(self._active_buffers),
                "fragmentation_stats": fragmentation_stats,
                "numa_stats": numa_stats,
                "config": {
                    "fragmentation_threshold": self._fragmentation_threshold,
                    "aggressive_threshold": self._aggressive_threshold,
                    "conservative_threshold": self._conservative_threshold,
                },
            }

    def defragment_all_devices(self) -> dict[str, bool]:
        """
        Perform defragmentation on all devices that need it.

        Returns:
            Dictionary mapping device names to whether defragmentation was performed
        """
        results = {}

        with self._lock:
            for device, score in self._fragmentation_scores.items():
                device_str = str(device)
                if score > self._fragmentation_threshold:
                    self._defragment_device(device)
                    results[device_str] = True
                    logger.info(f"Defragmented {device_str} (score: {score:.3f})")
                else:
                    results[device_str] = False

        return results

    def get_fragmentation_report(self) -> str:
        """Generate a human-readable fragmentation report."""
        lines = ["Memory Pool Fragmentation Report", "=" * 35, ""]

        with self._lock:
            for device, score in self._fragmentation_scores.items():
                fragments = self._fragments.get(device, [])
                free_fragments = sum(1 for f in fragments if f.is_free)
                total_fragments = len(fragments)

                status = (
                    "🔴 NEEDS DEFRAG"
                    if score > self._fragmentation_threshold
                    else "🟢 OK"
                )

                lines.extend(
                    [
                        f"Device: {device} {status}",
                        f"  Fragmentation Score: {score:.3f} (threshold: {self._fragmentation_threshold})",
                        f"  Total Fragments: {total_fragments}",
                        f"  Free Fragments: {free_fragments}",
                        "",
                    ]
                )

        return "\n".join(lines)

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
