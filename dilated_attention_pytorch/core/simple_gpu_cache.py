"""
Simple GPU-resident pattern cache with minimal overhead.

This implementation focuses on performance by pre-generating all patterns
and keeping them on GPU without complex tier management.
"""

import torch
from typing import Dict, Optional, Set
from threading import RLock


class SimpleGPUPatternCache:
    """
    Simple pattern cache that keeps all patterns on GPU.

    This cache pre-generates common patterns and stores them directly on GPU,
    eliminating transfer overhead without complex management logic.
    """

    def __init__(
        self,
        device: torch.device = None,
        pregenerate: bool = True,
        max_patterns: int = 100,
    ):
        """
        Initialize simple GPU pattern cache.

        Args:
            device: Target device (defaults to cuda if available)
            pregenerate: Whether to pre-generate common patterns
            max_patterns: Maximum number of patterns to cache
        """
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_patterns = max_patterns

        # Simple dict for O(1) lookup
        self._cache: Dict[str, torch.Tensor] = {}
        self._lock = RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

        if pregenerate and self.device.type == "cuda":
            self._pregenerate_common_patterns()

    def _pregenerate_common_patterns(self):
        """Pre-generate commonly used dilation patterns."""
        common_configs = [
            # (segment_length, dilation_rate, offset)
            (64, 1, 0),
            (128, 1, 0),
            (128, 2, 0),
            (128, 2, 1),
            (256, 1, 0),
            (256, 2, 0),
            (256, 2, 1),
            (512, 1, 0),
            (512, 2, 0),
            (512, 2, 1),
            (1024, 1, 0),
            (1024, 2, 0),
            (1024, 2, 1),
            (2048, 1, 0),
            (2048, 2, 0),
            (2048, 2, 1),
            (4096, 1, 0),
            (4096, 2, 0),
            (4096, 2, 1),
            (8192, 1, 0),
            (8192, 2, 0),
            (8192, 2, 1),
        ]

        for seg_len, dil_rate, offset in common_configs:
            if len(self._cache) >= self.max_patterns:
                break

            key = f"ring_dilated_s{seg_len}_r{dil_rate}_off{offset}"
            pattern = self._generate_pattern(seg_len, dil_rate, offset)
            self._cache[key] = pattern

    def _generate_pattern(
        self,
        segment_len: int,
        dilation_rate: int,
        offset: int,
    ) -> torch.Tensor:
        """Generate a dilation pattern."""
        if dilation_rate > 1:
            # Generate dilated indices
            indices = torch.arange(
                offset, segment_len, dilation_rate, device=self.device
            )
            # Pad if necessary
            if len(indices) < segment_len:
                repeats = (segment_len + len(indices) - 1) // len(indices)
                indices = indices.repeat(repeats)[:segment_len]
            return indices % segment_len
        else:
            return torch.arange(0, segment_len, device=self.device)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get pattern from cache.

        Args:
            key: Pattern key

        Returns:
            Pattern tensor on GPU or None if not found
        """
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None

    def put(self, key: str, pattern: torch.Tensor) -> None:
        """
        Store pattern in cache.

        Args:
            key: Pattern key
            pattern: Pattern tensor (will be moved to cache device)
        """
        with self._lock:
            if len(self._cache) >= self.max_patterns:
                # Simple FIFO eviction
                first_key = next(iter(self._cache))
                del self._cache[first_key]

            # Store on target device
            if pattern.device != self.device:
                pattern = pattern.to(self.device)

            self._cache[key] = pattern

    def get_or_generate(
        self,
        key: str,
        segment_len: int,
        dilation_rate: int,
        offset: int,
    ) -> torch.Tensor:
        """
        Get pattern from cache or generate if not found.

        Args:
            key: Pattern key
            segment_len: Segment length for generation
            dilation_rate: Dilation rate for generation
            offset: Offset for generation

        Returns:
            Pattern tensor on GPU
        """
        pattern = self.get(key)
        if pattern is None:
            # Generate and cache
            pattern = self._generate_pattern(segment_len, dilation_rate, offset)
            self.put(key, pattern)
        return pattern

    def warmup(self, patterns: Set[tuple[int, int, int]]) -> None:
        """
        Pre-generate specific patterns.

        Args:
            patterns: Set of (segment_len, dilation_rate, offset) tuples
        """
        for seg_len, dil_rate, offset in patterns:
            key = f"ring_dilated_s{seg_len}_r{dil_rate}_off{offset}"
            if key not in self._cache:
                pattern = self._generate_pattern(seg_len, dil_rate, offset)
                self.put(key, pattern)

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            # Calculate memory usage
            memory_bytes = 0
            for pattern in self._cache.values():
                memory_bytes += pattern.numel() * pattern.element_size()

            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "memory_mb": memory_bytes / (1024 * 1024),
                "device": str(self.device),
            }

    def clear(self) -> None:
        """Clear cache and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Global instance
_simple_gpu_cache: Optional[SimpleGPUPatternCache] = None


def get_simple_gpu_cache(**kwargs) -> SimpleGPUPatternCache:
    """Get or create global simple GPU cache."""
    global _simple_gpu_cache
    if _simple_gpu_cache is None:
        _simple_gpu_cache = SimpleGPUPatternCache(**kwargs)
    return _simple_gpu_cache


def clear_simple_gpu_cache() -> None:
    """Clear the global simple GPU cache."""
    global _simple_gpu_cache
    if _simple_gpu_cache is not None:
        _simple_gpu_cache.clear()
