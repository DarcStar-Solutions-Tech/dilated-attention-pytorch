"""
Pattern cache implementation for dilated attention.

This module provides caching for dilated attention patterns to avoid recomputation.
"""

import threading
from typing import Any, Dict, Optional

import torch

# Global pattern cache instance
_global_pattern_cache: Optional[Dict[str, Any]] = None
_cache_lock = threading.Lock()


class PatternCache:
    """Simple pattern cache for dilated attention patterns."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a pattern from the cache."""
        return self.cache.get(key, default)

    def put(self, key: str, value: Any) -> None:
        """Put a pattern into the cache."""
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class DilatedPatternCache(PatternCache):
    """Extended pattern cache with device management."""

    def get(
        self, key: str, target_device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        """Get a pattern and optionally move to target device."""
        value = self.cache.get(key)
        if (
            value is not None
            and target_device is not None
            and isinstance(value, torch.Tensor)
        ):
            value = value.to(target_device)
        return value

    def put(self, key: str, value: torch.Tensor, move_to_cpu: bool = True) -> None:
        """Put a pattern into the cache, optionally moving to CPU."""
        if (
            move_to_cpu
            and isinstance(value, torch.Tensor)
            and value.device.type != "cpu"
        ):
            value = value.cpu()
        self.cache[key] = value


def get_global_pattern_cache() -> Dict[str, Any]:
    """Get the global pattern cache dictionary."""
    global _global_pattern_cache
    with _cache_lock:
        if _global_pattern_cache is None:
            _global_pattern_cache = {}
        return _global_pattern_cache


def clear_global_cache() -> None:
    """Clear the global pattern cache."""
    global _global_pattern_cache
    with _cache_lock:
        if _global_pattern_cache is not None:
            _global_pattern_cache.clear()


# For backward compatibility
class OptimizedPatternCache(DilatedPatternCache):
    """Optimized pattern cache (same as DilatedPatternCache for now)."""

    pass


def get_optimized_pattern_cache() -> DilatedPatternCache:
    """Get an optimized pattern cache instance."""
    return DilatedPatternCache()


def clear_optimized_cache() -> None:
    """Clear optimized cache (no-op for instance-based cache)."""
    pass
