"""
Optimized pattern cache with GPU-resident patterns and async transfers.

This module provides an enhanced pattern cache that reduces CPU→GPU transfer overhead
by keeping frequently accessed patterns on GPU and using async transfers.
"""

import logging
import torch
from collections import OrderedDict
from typing import Dict, Any, Optional
from threading import RLock
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class OptimizedPatternCache:
    """
    Optimized pattern cache with multi-tier storage and async transfers.

    Features:
    - GPU-resident cache for hot patterns
    - Async CPU→GPU transfers with prefetching
    - Batch pattern transfers
    - Adaptive tier management
    """

    def __init__(
        self,
        max_gpu_patterns: int = 50,
        max_cpu_patterns: int = 500,
        gpu_memory_limit_mb: float = 100.0,
        enable_async: bool = True,
        enable_prefetch: bool = True,
    ):
        """
        Initialize optimized pattern cache.

        Args:
            max_gpu_patterns: Maximum patterns to keep on GPU
            max_cpu_patterns: Maximum patterns to keep on CPU
            gpu_memory_limit_mb: GPU memory limit for pattern storage
            enable_async: Enable async transfers
            enable_prefetch: Enable pattern prefetching
        """
        self.max_gpu_patterns = max_gpu_patterns
        self.max_cpu_patterns = max_cpu_patterns
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.enable_async = enable_async
        self.enable_prefetch = enable_prefetch

        # Two-tier cache: GPU (hot) and CPU (cold)
        self._gpu_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cpu_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Access statistics for adaptive management
        self._access_counts: Dict[str, int] = {}
        self._gpu_memory_used = 0.0

        # Thread safety
        self._lock = RLock()

        # Async transfer management
        self._executor = ThreadPoolExecutor(max_workers=2) if enable_async else None
        self._pending_transfers: Dict[str, asyncio.Future] = {}

        # Statistics
        self._gpu_hits = 0
        self._cpu_hits = 0
        self._misses = 0
        self._async_transfers = 0
        self._prefetch_hits = 0

    def get(
        self,
        key: str,
        target_device: Optional[torch.device] = None,
        prefetch_next: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get pattern from cache with optimized transfer.

        Args:
            key: Pattern key
            target_device: Target device for the pattern
            prefetch_next: Optional key to prefetch for next access

        Returns:
            Pattern tensor on target device or None if not found
        """
        with self._lock:
            # Update access count
            self._access_counts[key] = self._access_counts.get(key, 0) + 1

            # Check GPU cache first (fastest)
            if key in self._gpu_cache:
                self._gpu_hits += 1
                pattern = self._gpu_cache[key]
                # Move to end (LRU)
                self._gpu_cache.move_to_end(key)

                # Prefetch next pattern if specified
                if prefetch_next and self.enable_prefetch:
                    self._prefetch_pattern(prefetch_next, target_device)

                return pattern if target_device is None else pattern.to(target_device)

            # Check CPU cache
            if key in self._cpu_cache:
                self._cpu_hits += 1
                pattern = self._cpu_cache[key]
                # Move to end (LRU)
                self._cpu_cache.move_to_end(key)

                # Promote to GPU if frequently accessed
                if self._should_promote_to_gpu(key):
                    self._promote_to_gpu(key, pattern)

                # Transfer to target device
                if target_device and target_device.type == "cuda":
                    if self.enable_async:
                        return self._async_transfer(key, pattern, target_device)
                    else:
                        return pattern.to(target_device)

                return pattern

            # Pattern not found
            self._misses += 1
            return None

    def put(
        self,
        key: str,
        pattern: torch.Tensor,
        store_on_gpu: bool = False,
    ) -> None:
        """
        Store pattern in cache.

        Args:
            key: Pattern key
            pattern: Pattern tensor
            store_on_gpu: Whether to store directly on GPU
        """
        with self._lock:
            # Determine storage tier
            if store_on_gpu and pattern.device.type == "cuda":
                self._add_to_gpu_cache(key, pattern)
            else:
                # Always store CPU copy
                cpu_pattern = (
                    pattern.cpu() if pattern.device.type == "cuda" else pattern
                )
                self._add_to_cpu_cache(key, cpu_pattern)

                # Also store GPU copy if pattern is already on GPU and frequently accessed
                if pattern.device.type == "cuda" and self._should_promote_to_gpu(key):
                    self._add_to_gpu_cache(key, pattern)

    def get_batch(
        self,
        keys: list[str],
        target_device: Optional[torch.device] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get multiple patterns in a single batch for efficiency.

        Args:
            keys: List of pattern keys
            target_device: Target device for all patterns

        Returns:
            Dictionary mapping keys to patterns
        """
        results = {}
        gpu_transfers = []

        with self._lock:
            for key in keys:
                # Check GPU cache
                if key in self._gpu_cache:
                    self._gpu_hits += 1
                    pattern = self._gpu_cache[key]
                    self._gpu_cache.move_to_end(key)
                    results[key] = pattern
                # Check CPU cache
                elif key in self._cpu_cache:
                    self._cpu_hits += 1
                    pattern = self._cpu_cache[key]
                    self._cpu_cache.move_to_end(key)

                    if target_device and target_device.type == "cuda":
                        gpu_transfers.append((key, pattern))
                    else:
                        results[key] = pattern
                else:
                    self._misses += 1
                    results[key] = None

        # Batch GPU transfers for efficiency
        if gpu_transfers and target_device:
            # Stack patterns for single transfer
            keys_to_transfer = [k for k, _ in gpu_transfers]
            patterns_to_transfer = [p for _, p in gpu_transfers]

            # Batch transfer
            transferred = self._batch_transfer(patterns_to_transfer, target_device)

            # Update results
            for key, pattern in zip(keys_to_transfer, transferred):
                results[key] = pattern

        return results

    def pin_pattern(self, key: str, device: torch.device) -> bool:
        """
        Pin a pattern to stay on a specific device.

        Args:
            key: Pattern key
            device: Device to pin the pattern to

        Returns:
            True if pattern was successfully pinned
        """
        with self._lock:
            # Find pattern
            pattern = None
            if key in self._gpu_cache:
                pattern = self._gpu_cache[key]
            elif key in self._cpu_cache:
                pattern = self._cpu_cache[key]
            else:
                return False

            # Pin to specified device
            if device.type == "cuda":
                if key not in self._gpu_cache:
                    self._add_to_gpu_cache(key, pattern.to(device))
                # Mark as pinned (high access count)
                self._access_counts[key] = max(self._access_counts.get(key, 0), 1000)

            return True

    def _should_promote_to_gpu(self, key: str) -> bool:
        """Check if pattern should be promoted to GPU cache."""
        access_count = self._access_counts.get(key, 0)
        # Promote if accessed more than 3 times
        return access_count > 3 and len(self._gpu_cache) < self.max_gpu_patterns

    def _promote_to_gpu(self, key: str, pattern: torch.Tensor) -> None:
        """Promote pattern from CPU to GPU cache."""
        if pattern.device.type == "cpu":
            # Check if we have CUDA available
            if torch.cuda.is_available():
                gpu_pattern = pattern.cuda()
                self._add_to_gpu_cache(key, gpu_pattern)

    def _add_to_gpu_cache(self, key: str, pattern: torch.Tensor) -> None:
        """Add pattern to GPU cache with memory management."""
        pattern_size_mb = pattern.numel() * pattern.element_size() / (1024 * 1024)

        # Check memory limit
        if self._gpu_memory_used + pattern_size_mb > self.gpu_memory_limit_mb:
            # Evict least recently used patterns
            while (
                self._gpu_memory_used + pattern_size_mb > self.gpu_memory_limit_mb
                and len(self._gpu_cache) > 0
            ):
                evict_key = next(iter(self._gpu_cache))
                evicted = self._gpu_cache.pop(evict_key)
                evicted_size = evicted.numel() * evicted.element_size() / (1024 * 1024)
                self._gpu_memory_used -= evicted_size

        # Check count limit
        if len(self._gpu_cache) >= self.max_gpu_patterns:
            evict_key = next(iter(self._gpu_cache))
            evicted = self._gpu_cache.pop(evict_key)
            evicted_size = evicted.numel() * evicted.element_size() / (1024 * 1024)
            self._gpu_memory_used -= evicted_size

        # Add to cache
        self._gpu_cache[key] = pattern
        self._gpu_memory_used += pattern_size_mb

    def _add_to_cpu_cache(self, key: str, pattern: torch.Tensor) -> None:
        """Add pattern to CPU cache."""
        # Evict if full
        if len(self._cpu_cache) >= self.max_cpu_patterns:
            evict_key = next(iter(self._cpu_cache))
            del self._cpu_cache[evict_key]

        self._cpu_cache[key] = pattern

    def _async_transfer(
        self,
        key: str,
        pattern: torch.Tensor,
        target_device: torch.device,
    ) -> torch.Tensor:
        """Perform async CPU→GPU transfer."""
        # For now, do sync transfer (async requires more complex stream management)
        self._async_transfers += 1
        return pattern.to(target_device, non_blocking=True)

    def _batch_transfer(
        self,
        patterns: list[torch.Tensor],
        target_device: torch.device,
    ) -> list[torch.Tensor]:
        """Batch transfer multiple patterns to GPU."""
        # Transfer all patterns in one operation for efficiency
        with torch.cuda.stream(torch.cuda.Stream()):
            transferred = [p.to(target_device, non_blocking=True) for p in patterns]
        return transferred

    def _prefetch_pattern(
        self, key: str, target_device: Optional[torch.device]
    ) -> None:
        """Prefetch a pattern that might be needed next."""
        if key in self._cpu_cache and key not in self._gpu_cache:
            if target_device and target_device.type == "cuda":
                # Prefetch to GPU in background
                pattern = self._cpu_cache[key]
                if self._should_promote_to_gpu(key):
                    self._promote_to_gpu(key, pattern)
                    self._prefetch_hits += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = self._gpu_hits + self._cpu_hits
            total_accesses = total_hits + self._misses
            hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

            return {
                "gpu_cache_size": len(self._gpu_cache),
                "cpu_cache_size": len(self._cpu_cache),
                "gpu_memory_used_mb": self._gpu_memory_used,
                "gpu_hits": self._gpu_hits,
                "cpu_hits": self._cpu_hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "async_transfers": self._async_transfers,
                "prefetch_hits": self._prefetch_hits,
                "total_accesses": total_accesses,
            }

    def clear(self) -> None:
        """Clear all cached patterns."""
        with self._lock:
            self._gpu_cache.clear()
            self._cpu_cache.clear()
            self._access_counts.clear()
            self._gpu_memory_used = 0.0

            # Reset stats
            self._gpu_hits = 0
            self._cpu_hits = 0
            self._misses = 0
            self._async_transfers = 0
            self._prefetch_hits = 0


# Global optimized cache instance
_optimized_pattern_cache: Optional[OptimizedPatternCache] = None


def get_optimized_pattern_cache(**kwargs) -> OptimizedPatternCache:
    """Get or create the global optimized pattern cache."""
    global _optimized_pattern_cache
    if _optimized_pattern_cache is None:
        _optimized_pattern_cache = OptimizedPatternCache(**kwargs)
    return _optimized_pattern_cache


def clear_optimized_cache() -> None:
    """Clear the global optimized pattern cache."""
    global _optimized_pattern_cache
    if _optimized_pattern_cache is not None:
        _optimized_pattern_cache.clear()
