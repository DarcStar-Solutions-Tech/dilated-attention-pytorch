"""
Size-bucketed memory pool for efficient allocation.

This module implements a bucketed memory pool that reduces allocation overhead
by maintaining pre-allocated buffers in size buckets optimized for transformer
workloads.
"""

import logging
import math
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor

logger = logging.getLogger("dilated_attention_pytorch.bucketed_memory_pool")


@dataclass
class BucketConfig:
    """Configuration for a memory bucket."""

    size: int  # Bucket size in bytes
    initial_count: int = 4  # Initial number of buffers
    max_count: int = 32  # Maximum buffers to keep
    growth_factor: float = 2.0  # Growth factor when expanding

    @property
    def name(self) -> str:
        """Human-readable bucket name."""
        if self.size < 1024:
            return f"{self.size}B"
        elif self.size < 1024 * 1024:
            return f"{self.size // 1024}KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size // (1024 * 1024)}MB"
        else:
            return f"{self.size // (1024 * 1024 * 1024)}GB"


class MemoryBucket:
    """
    A single memory bucket managing buffers of a specific size.

    Features:
    - Pre-allocation of buffers
    - Dynamic growth and shrinking
    - LRU eviction when at capacity
    - Usage statistics tracking
    """

    def __init__(self, config: BucketConfig):
        """Initialize the memory bucket."""
        self.config = config
        self.size = config.size

        # Available buffers (using OrderedDict for LRU)
        self._available: OrderedDict[int, Tensor] = OrderedDict()

        # In-use buffers (tracked by tensor id)
        self._in_use: Dict[int, Tensor] = {}

        # Statistics
        self.stats = {
            "allocations": 0,
            "deallocations": 0,
            "hits": 0,
            "misses": 0,
            "current_count": 0,
            "peak_count": 0,
            "total_allocated": 0,
            "last_access_time": 0.0,
        }

        # Lock for thread safety
        self._lock = threading.Lock()

        # Pre-allocate initial buffers
        self._preallocate()

        logger.debug(
            f"Initialized bucket {config.name} with {config.initial_count} buffers"
        )

    def _preallocate(self) -> None:
        """Pre-allocate initial buffers."""
        for _ in range(self.config.initial_count):
            # Don't actually allocate tensor memory yet - lazy allocation
            # Just track that we have capacity
            self.stats["current_count"] += 1
            self.stats["peak_count"] = max(
                self.stats["peak_count"], self.stats["current_count"]
            )

    def allocate(
        self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> Optional[Tensor]:
        """
        Allocate a buffer from this bucket.

        Args:
            shape: Desired tensor shape
            dtype: Data type
            device: Device to allocate on

        Returns:
            Allocated tensor or None if size doesn't match bucket
        """
        # Check if requested size matches bucket size
        element_size = torch.finfo(dtype).bits // 8
        requested_size = math.prod(shape) * element_size

        if requested_size > self.size:
            return None

        with self._lock:
            self.stats["allocations"] += 1
            self.stats["last_access_time"] = time.time()

            # Try to reuse available buffer
            for buffer_id, buffer in list(self._available.items()):
                if buffer.device == device and buffer.dtype == dtype:
                    # Check if we can reshape
                    if buffer.numel() >= math.prod(shape):
                        # Remove from available
                        del self._available[buffer_id]

                        # Reshape to requested shape
                        try:
                            if buffer.numel() == math.prod(shape):
                                result = buffer.view(shape)
                            else:
                                # Use a slice if buffer is larger
                                flat = buffer.flatten()
                                result = flat[: math.prod(shape)].view(shape)

                            # Track as in-use
                            self._in_use[id(result)] = buffer

                            self.stats["hits"] += 1
                            return result

                        except RuntimeError:
                            # Reshape failed, put back in available
                            self._available[buffer_id] = buffer

            # No suitable buffer found - allocate new
            self.stats["misses"] += 1

            # Check if we're at capacity
            total_buffers = len(self._available) + len(self._in_use)
            if total_buffers >= self.config.max_count:
                # Evict least recently used available buffer
                if self._available:
                    _, evicted = self._available.popitem(last=False)
                    del evicted  # Release memory
                    self.stats["current_count"] -= 1

            # Allocate new buffer
            try:
                buffer = torch.empty(
                    self.size // element_size, dtype=dtype, device=device
                )

                # Reshape to requested shape
                if buffer.numel() == math.prod(shape):
                    result = buffer.view(shape)
                else:
                    result = buffer.flatten()[: math.prod(shape)].view(shape)

                # Track as in-use
                self._in_use[id(result)] = buffer

                self.stats["current_count"] += 1
                self.stats["peak_count"] = max(
                    self.stats["peak_count"], self.stats["current_count"]
                )
                self.stats["total_allocated"] += self.size

                return result

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM allocating {self.config.name} buffer on {device}")
                return None

    def deallocate(self, tensor: Tensor) -> bool:
        """
        Return a buffer to this bucket.

        Args:
            tensor: Tensor to deallocate

        Returns:
            True if tensor belonged to this bucket
        """
        tensor_id = id(tensor)

        with self._lock:
            if tensor_id not in self._in_use:
                return False

            # Get original buffer
            buffer = self._in_use.pop(tensor_id)

            self.stats["deallocations"] += 1

            # Add to available pool
            self._available[id(buffer)] = buffer

            # Limit available pool size
            while len(self._available) > self.config.max_count // 2:
                _, evicted = self._available.popitem(last=False)
                del evicted
                self.stats["current_count"] -= 1

            return True

    def clear(self) -> None:
        """Clear all buffers from this bucket."""
        with self._lock:
            self._available.clear()
            self._in_use.clear()
            self.stats["current_count"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get bucket statistics."""
        with self._lock:
            return {
                "name": self.config.name,
                "size": self.size,
                "allocations": self.stats["allocations"],
                "deallocations": self.stats["deallocations"],
                "hit_rate": (
                    self.stats["hits"] / max(1, self.stats["allocations"])
                    if self.stats["allocations"] > 0
                    else 0.0
                ),
                "current_buffers": self.stats["current_count"],
                "peak_buffers": self.stats["peak_count"],
                "available": len(self._available),
                "in_use": len(self._in_use),
                "memory_allocated": self.stats["total_allocated"],
                "efficiency": (
                    len(self._in_use) / max(1, self.stats["current_count"])
                    if self.stats["current_count"] > 0
                    else 0.0
                ),
            }


class BucketedMemoryPool:
    """
    Memory pool with size bucketing for efficient allocation.

    This pool maintains separate buckets for different allocation sizes,
    optimized for transformer model memory patterns.
    """

    # Default bucket sizes optimized for transformers
    DEFAULT_BUCKET_SIZES = [
        64,  # Very small tensors
        256,  # Small bias terms
        1024,  # 1KB - small activations
        4096,  # 4KB - medium activations
        16384,  # 16KB - large activations
        65536,  # 64KB - attention weights
        262144,  # 256KB - attention matrices
        1048576,  # 1MB - large buffers
        4194304,  # 4MB - very large buffers
        16777216,  # 16MB - extreme cases
        67108864,  # 64MB - huge attention
        268435456,  # 256MB - massive buffers
    ]

    def __init__(
        self,
        bucket_sizes: Optional[List[int]] = None,
        enable_statistics: bool = True,
        adaptive_buckets: bool = True,
    ):
        """
        Initialize the bucketed memory pool.

        Args:
            bucket_sizes: Custom bucket sizes (uses defaults if None)
            enable_statistics: Whether to track detailed statistics
            adaptive_buckets: Whether to adaptively create new buckets
        """
        self.bucket_sizes = bucket_sizes or self.DEFAULT_BUCKET_SIZES
        self.enable_statistics = enable_statistics
        self.adaptive_buckets = adaptive_buckets

        # Create buckets
        self.buckets: Dict[int, MemoryBucket] = {}
        for size in self.bucket_sizes:
            config = BucketConfig(
                size=size,
                initial_count=self._get_initial_count(size),
                max_count=self._get_max_count(size),
            )
            self.buckets[size] = MemoryBucket(config)

        # Large allocation pool for sizes beyond buckets
        self.large_allocations: Dict[int, Tensor] = {}

        # Adaptive bucket tracking
        self.allocation_sizes: defaultdict = defaultdict(int)
        self.adaptation_threshold = 100  # Min allocations before creating bucket

        # Statistics
        self.stats = {
            "total_allocations": 0,
            "bucketed_allocations": 0,
            "large_allocations": 0,
            "adaptive_buckets_created": 0,
        }

        # Lock for thread safety
        self._lock = threading.Lock()

        logger.info(f"Initialized BucketedMemoryPool with {len(self.buckets)} buckets")

    def _get_initial_count(self, size: int) -> int:
        """Get initial buffer count for a bucket size."""
        # Smaller buckets get more initial buffers
        if size <= 1024:
            return 16
        elif size <= 65536:
            return 8
        elif size <= 1048576:
            return 4
        else:
            return 2

    def _get_max_count(self, size: int) -> int:
        """Get maximum buffer count for a bucket size."""
        # Smaller buckets can have more buffers
        if size <= 1024:
            return 128
        elif size <= 65536:
            return 64
        elif size <= 1048576:
            return 32
        else:
            return 16

    def allocate(
        self,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Tensor:
        """
        Allocate memory using appropriate bucket.

        Args:
            size: Size in bytes
            dtype: Data type
            device: Device to allocate on
            shape: Optional shape for the tensor

        Returns:
            Allocated tensor
        """
        with self._lock:
            self.stats["total_allocations"] += 1

            # Track allocation size for adaptive bucketing
            if self.adaptive_buckets:
                self.allocation_sizes[size] += 1

            # Find appropriate bucket
            bucket_size = self._find_bucket_size(size)

            if bucket_size is not None:
                bucket = self.buckets[bucket_size]

                # Calculate shape if not provided
                if shape is None:
                    element_size = torch.finfo(dtype).bits // 8
                    num_elements = size // element_size
                    shape = (num_elements,)

                # Try to allocate from bucket
                tensor = bucket.allocate(shape, dtype, device)

                if tensor is not None:
                    self.stats["bucketed_allocations"] += 1
                    return tensor

            # Check if we should create a new adaptive bucket
            if self.adaptive_buckets and self._should_create_bucket(size):
                self._create_adaptive_bucket(size)
                # Retry with new bucket
                return self.allocate(size, dtype, device, shape)

            # Fall back to large allocation
            return self._allocate_large(size, dtype, device, shape)

    def deallocate(self, tensor: Tensor) -> None:
        """
        Return memory to the pool.

        Args:
            tensor: Tensor to deallocate
        """
        tensor_id = id(tensor)

        with self._lock:
            # Try each bucket
            for bucket in self.buckets.values():
                if bucket.deallocate(tensor):
                    return

            # Check large allocations
            if tensor_id in self.large_allocations:
                del self.large_allocations[tensor_id]
                return

            # Not tracked - log warning
            logger.warning(f"Deallocating untracked tensor {tensor_id}")

    def _find_bucket_size(self, size: int) -> Optional[int]:
        """Find the best bucket size for an allocation."""
        # Find smallest bucket that fits
        for bucket_size in sorted(self.buckets.keys()):
            if bucket_size >= size:
                # Check if size is reasonably close (within 2x)
                if size >= bucket_size // 2:
                    return bucket_size

        return None

    def _should_create_bucket(self, size: int) -> bool:
        """Check if we should create a new adaptive bucket."""
        # Don't create if size already has a good bucket
        if self._find_bucket_size(size) is not None:
            return False

        # Check if this size is requested frequently
        count = self.allocation_sizes[size]
        return count >= self.adaptation_threshold

    def _create_adaptive_bucket(self, size: int) -> None:
        """Create a new adaptive bucket for frequently requested size."""
        # Round up to nice size
        if size < 1024:
            bucket_size = ((size + 63) // 64) * 64  # Round to 64 bytes
        elif size < 1024 * 1024:
            bucket_size = ((size + 1023) // 1024) * 1024  # Round to KB
        else:
            bucket_size = ((size + 1048575) // 1048576) * 1048576  # Round to MB

        # Don't create if already exists
        if bucket_size in self.buckets:
            return

        # Create new bucket
        config = BucketConfig(
            size=bucket_size,
            initial_count=4,
            max_count=16,
        )

        self.buckets[bucket_size] = MemoryBucket(config)
        self.stats["adaptive_buckets_created"] += 1

        logger.info(f"Created adaptive bucket: {config.name}")

    def _allocate_large(
        self,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Tensor:
        """Allocate memory for large requests."""
        if shape is None:
            element_size = torch.finfo(dtype).bits // 8
            num_elements = size // element_size
            shape = (num_elements,)

        try:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self.large_allocations[id(tensor)] = tensor
            self.stats["large_allocations"] += 1
            return tensor

        except torch.cuda.OutOfMemoryError:
            # Try to free some memory
            self._emergency_cleanup()

            # Retry
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self.large_allocations[id(tensor)] = tensor
            self.stats["large_allocations"] += 1
            return tensor

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when OOM."""
        logger.warning("Emergency cleanup triggered")

        # Clear half of each bucket's available buffers
        for bucket in self.buckets.values():
            with bucket._lock:
                count = len(bucket._available) // 2
                for _ in range(count):
                    if bucket._available:
                        _, buffer = bucket._available.popitem(last=False)
                        del buffer
                        bucket.stats["current_count"] -= 1

        # Force CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self) -> None:
        """Clear all memory pools."""
        with self._lock:
            # Clear all buckets
            for bucket in self.buckets.values():
                bucket.clear()

            # Clear large allocations
            self.large_allocations.clear()

            # Reset adaptive tracking
            self.allocation_sizes.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._lock:
            bucket_stats = {}
            total_memory = 0
            total_buffers = 0

            for size, bucket in sorted(self.buckets.items()):
                stats = bucket.get_stats()
                bucket_stats[stats["name"]] = stats
                total_memory += stats["memory_allocated"]
                total_buffers += stats["current_buffers"]

            return {
                "total_allocations": self.stats["total_allocations"],
                "bucketed_allocations": self.stats["bucketed_allocations"],
                "large_allocations": self.stats["large_allocations"],
                "bucketed_ratio": (
                    self.stats["bucketed_allocations"]
                    / max(1, self.stats["total_allocations"])
                ),
                "adaptive_buckets_created": self.stats["adaptive_buckets_created"],
                "total_buckets": len(self.buckets),
                "total_memory_allocated": total_memory,
                "total_buffers": total_buffers,
                "bucket_stats": bucket_stats,
            }

    def get_efficiency_report(self) -> str:
        """Generate efficiency report."""
        lines = ["Bucketed Memory Pool Efficiency Report", "=" * 40, ""]

        stats = self.get_stats()

        lines.extend(
            [
                f"Total Allocations: {stats['total_allocations']:,}",
                f"Bucketed Allocations: {stats['bucketed_allocations']:,} ({stats['bucketed_ratio']:.1%})",
                f"Large Allocations: {stats['large_allocations']:,}",
                f"Adaptive Buckets Created: {stats['adaptive_buckets_created']}",
                f"Total Memory: {stats['total_memory_allocated'] / (1024**3):.2f} GB",
                "",
                "Bucket Efficiency:",
                "-" * 40,
            ]
        )

        # Sort buckets by efficiency
        bucket_list = []
        for name, bucket_stats in stats["bucket_stats"].items():
            bucket_list.append(
                (
                    name,
                    bucket_stats["allocations"],
                    bucket_stats["hit_rate"],
                    bucket_stats["efficiency"],
                )
            )

        bucket_list.sort(key=lambda x: x[1], reverse=True)  # Sort by allocations

        for name, allocs, hit_rate, efficiency in bucket_list:
            if allocs > 0:
                lines.append(
                    f"{name:>10}: {allocs:>8} allocs, "
                    f"{hit_rate:>5.1%} hits, {efficiency:>5.1%} usage"
                )

        return "\n".join(lines)
