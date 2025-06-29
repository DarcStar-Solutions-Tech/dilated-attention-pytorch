"""
Pattern caching utilities for dilated attention patterns.

This module provides a unified pattern caching system that stores commonly used
attention patterns (dilated indices, sparse masks, etc.) on CPU to avoid
redundant computation and reduce GPU memory usage.
"""

import threading
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any, Union
import torch
import logging

logger = logging.getLogger(__name__)


class PatternCache:
    """
    Thread-safe pattern cache for dilated attention indices and masks.

    Patterns are stored on CPU to save GPU memory and transferred to GPU
    on demand. Uses LRU eviction when cache size exceeds the limit.

    Args:
        max_size: Maximum number of patterns to cache (default: 100)
        device: Default device for cached patterns (default: 'cpu')
        enable_stats: Whether to track cache hit/miss statistics
    """

    def __init__(
        self,
        max_size: int = 100,
        device: Union[str, torch.device] = "cpu",
        enable_stats: bool = True,
    ):
        self.max_size = max_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.enable_stats = enable_stats

        # Main cache storage
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(
        self, key: str, target_device: Optional[torch.device] = None
    ) -> Optional[Any]:
        """
        Retrieve a pattern from cache, optionally moving it to target device.

        Args:
            key: Cache key for the pattern
            target_device: Device to move the pattern to (if different from storage)

        Returns:
            Cached pattern or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)

                if self.enable_stats:
                    self._hits += 1

                pattern = self._cache[key]

                # Move to target device if requested
                if target_device is not None and target_device != self.device:
                    if isinstance(pattern, torch.Tensor):
                        pattern = pattern.to(target_device)
                    elif isinstance(pattern, tuple):
                        # Handle tuples of tensors (e.g., row_idx, col_idx)
                        pattern = tuple(
                            t.to(target_device) if isinstance(t, torch.Tensor) else t
                            for t in pattern
                        )

                return pattern
            else:
                if self.enable_stats:
                    self._misses += 1
                return None

    def put(self, key: str, pattern: Any, move_to_cpu: bool = True) -> None:
        """
        Store a pattern in the cache.

        Args:
            key: Cache key for the pattern
            pattern: Pattern to cache (tensor or tuple of tensors)
            move_to_cpu: Whether to move the pattern to CPU before caching
        """
        with self._lock:
            # Move pattern to CPU if requested
            if move_to_cpu and self.device.type == "cpu":
                if isinstance(pattern, torch.Tensor):
                    pattern = pattern.cpu()
                elif isinstance(pattern, tuple):
                    pattern = tuple(
                        t.cpu() if isinstance(t, torch.Tensor) else t for t in pattern
                    )

            # Remove oldest item if cache is full
            if len(self._cache) >= self.max_size:
                removed_key = next(iter(self._cache))
                del self._cache[removed_key]
                if self.enable_stats:
                    self._evictions += 1
                logger.debug(f"Evicted pattern with key: {removed_key}")

            self._cache[key] = pattern
            self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cached patterns."""
        with self._lock:
            self._cache.clear()
            logger.info("Pattern cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "total_accesses": total_accesses,
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0


class DilatedPatternCache(PatternCache):
    """
    Specialized pattern cache for dilated attention indices.

    Provides convenience methods for caching and retrieving dilated
    attention patterns with automatic key generation.
    """

    def get_dilated_indices(
        self,
        seq_len: int,
        segment_lengths: Tuple[int, ...],
        dilation_rates: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get cached dilated indices for given parameters.

        Args:
            seq_len: Sequence length
            segment_lengths: Tuple of segment lengths
            dilation_rates: Tuple of dilation rates
            device: Target device for indices

        Returns:
            Cached indices tensor or None if not found
        """
        key = self._make_dilated_key(seq_len, segment_lengths, dilation_rates)
        return self.get(key, target_device=device)

    def put_dilated_indices(
        self,
        indices: torch.Tensor,
        seq_len: int,
        segment_lengths: Tuple[int, ...],
        dilation_rates: Tuple[int, ...],
    ) -> None:
        """
        Cache dilated indices.

        Args:
            indices: Indices tensor to cache
            seq_len: Sequence length
            segment_lengths: Tuple of segment lengths
            dilation_rates: Tuple of dilation rates
        """
        key = self._make_dilated_key(seq_len, segment_lengths, dilation_rates)
        self.put(key, indices, move_to_cpu=True)

    def get_sparse_pattern(
        self,
        seq_len: int,
        pattern_type: str,
        sparsity_ratio: float,
        block_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached sparse pattern (row and column indices).

        Args:
            seq_len: Sequence length
            pattern_type: Type of sparse pattern
            sparsity_ratio: Sparsity ratio
            block_size: Block size for block-sparse patterns
            device: Target device for pattern

        Returns:
            Tuple of (row_indices, col_indices) or None if not found
        """
        key = self._make_sparse_key(seq_len, pattern_type, sparsity_ratio, block_size)
        return self.get(key, target_device=device)

    def put_sparse_pattern(
        self,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        seq_len: int,
        pattern_type: str,
        sparsity_ratio: float,
        block_size: Optional[int] = None,
    ) -> None:
        """
        Cache sparse pattern indices.

        Args:
            row_indices: Row indices tensor
            col_indices: Column indices tensor
            seq_len: Sequence length
            pattern_type: Type of sparse pattern
            sparsity_ratio: Sparsity ratio
            block_size: Block size for block-sparse patterns
        """
        key = self._make_sparse_key(seq_len, pattern_type, sparsity_ratio, block_size)
        self.put(key, (row_indices, col_indices), move_to_cpu=True)

    @staticmethod
    def _make_dilated_key(
        seq_len: int, segment_lengths: Tuple[int, ...], dilation_rates: Tuple[int, ...]
    ) -> str:
        """Generate cache key for dilated pattern."""
        return f"dilated_s{seq_len}_seg{segment_lengths}_dil{dilation_rates}"

    @staticmethod
    def _make_sparse_key(
        seq_len: int,
        pattern_type: str,
        sparsity_ratio: float,
        block_size: Optional[int] = None,
    ) -> str:
        """Generate cache key for sparse pattern."""
        key = f"sparse_s{seq_len}_t{pattern_type}_r{sparsity_ratio:.3f}"
        if block_size is not None:
            key += f"_b{block_size}"
        return key


# Global cache instance for shared use across modules
_global_pattern_cache: Optional[DilatedPatternCache] = None
_cache_lock = threading.Lock()


def get_global_pattern_cache(
    max_size: int = 100, enable_stats: bool = True
) -> DilatedPatternCache:
    """
    Get or create the global pattern cache instance.

    Args:
        max_size: Maximum cache size
        enable_stats: Whether to enable statistics tracking

    Returns:
        Global DilatedPatternCache instance
    """
    global _global_pattern_cache

    if _global_pattern_cache is None:
        with _cache_lock:
            if _global_pattern_cache is None:
                _global_pattern_cache = DilatedPatternCache(
                    max_size=max_size, enable_stats=enable_stats
                )
                logger.info(f"Created global pattern cache with max_size={max_size}")

    return _global_pattern_cache


def clear_global_cache():
    """Clear the global pattern cache."""
    global _global_pattern_cache
    if _global_pattern_cache is not None:
        _global_pattern_cache.clear()
        logger.info("Global pattern cache cleared")
