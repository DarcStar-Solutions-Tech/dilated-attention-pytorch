"""
Cache management utilities for kernel implementations.

This module provides a bounded LRU cache for Hilbert mappings and other
cached tensors to prevent unbounded memory growth.
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import torch


class BoundedCache:
    """
    A bounded LRU cache for tensors with size and memory limits.

    Features:
    - LRU eviction when size limit is reached
    - Optional memory limit tracking
    - Thread-safe operations (via GIL)
    - Device-aware storage
    """

    def __init__(
        self,
        max_size: int = 32,
        max_memory_mb: Optional[float] = None,
        name: str = "cache",
    ):
        """
        Initialize bounded cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB (optional)
            name: Cache name for logging
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.name = name

        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[Union[int, Tuple], torch.Tensor] = OrderedDict()
        self._memory_usage_mb = 0.0

    def get(self, key: Union[int, Tuple], default=None) -> Optional[torch.Tensor]:
        """Get item from cache, updating LRU order."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return default

    def put(self, key: Union[int, Tuple], value: torch.Tensor) -> None:
        """Put item in cache with LRU eviction if needed."""
        # Calculate memory usage
        tensor_memory_mb = value.element_size() * value.nelement() / (1024 * 1024)

        # Check if we need to evict based on memory limit
        if self.max_memory_mb is not None:
            while (
                self._memory_usage_mb + tensor_memory_mb > self.max_memory_mb
                and len(self._cache) > 0
            ):
                self._evict_lru()

        # Check if we need to evict based on size limit
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        # Add new entry
        self._cache[key] = value
        self._memory_usage_mb += tensor_memory_mb

        # Move to end (most recently used)
        self._cache.move_to_end(key)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if len(self._cache) == 0:
            return

        # Pop first item (least recently used)
        key, value = self._cache.popitem(last=False)

        # Update memory usage
        tensor_memory_mb = value.element_size() * value.nelement() / (1024 * 1024)
        self._memory_usage_mb -= tensor_memory_mb

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._memory_usage_mb = 0.0

    def __contains__(self, key: Union[int, Tuple]) -> bool:
        """Check if key is in cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    @property
    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        return self._memory_usage_mb

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._memory_usage_mb,
            "max_memory_mb": self.max_memory_mb or float("inf"),
        }


class CachedHilbertMixin:
    """
    Mixin class that provides cached Hilbert mapping functionality.

    This should be mixed into kernel implementations that need
    Hilbert mapping caching with proper memory management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize cache with reasonable defaults
        cache_size = kwargs.get("cache_size", 32)
        cache_memory_mb = kwargs.get("cache_memory_mb", 100.0)

        self._hilbert_cache = BoundedCache(
            max_size=cache_size,
            max_memory_mb=cache_memory_mb,
            name=f"{self.__class__.__name__}_hilbert",
        )

    def _get_cached_hilbert_mapping(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Get Hilbert mapping from cache or create new one.

        Args:
            seq_len: Sequence length
            device: Target device

        Returns:
            Hilbert mapping tensor
        """
        # Check cache first
        mapping = self._hilbert_cache.get(seq_len)
        if mapping is not None and mapping.device == device:
            return mapping

        # Create new mapping
        from .hilbert_attention_core import create_hilbert_mapping

        mapping = create_hilbert_mapping(seq_len).to(device)

        # Cache it
        self._hilbert_cache.put(seq_len, mapping)

        return mapping

    def clear_cache(self) -> None:
        """Clear the Hilbert mapping cache."""
        self._hilbert_cache.clear()

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return self._hilbert_cache.get_stats()


class MultiLevelCache:
    """
    Multi-level cache for different types of cached data.

    Useful for kernels that cache multiple types of data
    (e.g., Hilbert mappings, sparse patterns, etc.)
    """

    def __init__(self, cache_configs: Dict[str, Dict[str, Union[int, float]]]):
        """
        Initialize multi-level cache.

        Args:
            cache_configs: Dictionary mapping cache names to their configs
                          e.g., {"hilbert": {"max_size": 32, "max_memory_mb": 50}}
        """
        self.caches = {}

        for name, config in cache_configs.items():
            self.caches[name] = BoundedCache(
                max_size=config.get("max_size", 32),
                max_memory_mb=config.get("max_memory_mb"),
                name=name,
            )

    def get(self, cache_name: str, key: Union[int, Tuple], default=None):
        """Get from specific cache."""
        if cache_name in self.caches:
            return self.caches[cache_name].get(key, default)
        return default

    def put(self, cache_name: str, key: Union[int, Tuple], value: torch.Tensor):
        """Put into specific cache."""
        if cache_name in self.caches:
            self.caches[cache_name].put(key, value)

    def clear(self, cache_name: Optional[str] = None):
        """Clear specific cache or all caches."""
        if cache_name is None:
            for cache in self.caches.values():
                cache.clear()
        elif cache_name in self.caches:
            self.caches[cache_name].clear()

    def get_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics for all caches."""
        return {name: cache.get_stats() for name, cache in self.caches.items()}

    @property
    def total_memory_usage_mb(self) -> float:
        """Get total memory usage across all caches."""
        return sum(cache.memory_usage_mb for cache in self.caches.values())
