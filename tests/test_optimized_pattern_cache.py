"""
Test optimized pattern cache functionality.
"""

import torch
import pytest

from dilated_attention_pytorch.core.optimized_pattern_cache import (
    OptimizedPatternCache,
    get_optimized_pattern_cache,
    clear_optimized_cache,
)


class TestOptimizedPatternCache:
    """Test suite for optimized pattern cache."""

    def test_gpu_cache_promotion(self):
        """Test that frequently accessed patterns get promoted to GPU."""
        cache = OptimizedPatternCache(
            max_gpu_patterns=5,
            max_cpu_patterns=10,
            gpu_memory_limit_mb=10.0,
        )

        # Create a pattern
        pattern = torch.arange(100)
        key = "test_pattern"

        # Store on CPU initially
        cache.put(key, pattern, store_on_gpu=False)

        # Access it multiple times (should promote to GPU after 3 accesses)
        for i in range(5):
            retrieved = cache.get(
                key,
                target_device=torch.device("cuda")
                if torch.cuda.is_available()
                else None,
            )
            assert retrieved is not None

        stats = cache.get_stats()
        if torch.cuda.is_available():
            # Should have GPU cache entries after promotion
            assert stats["gpu_cache_size"] > 0
            assert stats["gpu_hits"] > 0

    def test_batch_transfer(self):
        """Test batch pattern transfer efficiency."""
        cache = OptimizedPatternCache()

        # Store multiple patterns
        patterns = {}
        for i in range(5):
            key = f"pattern_{i}"
            pattern = torch.arange(i * 10, (i + 1) * 10)
            cache.put(key, pattern)
            patterns[key] = pattern

        # Batch retrieve
        keys = list(patterns.keys())
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        retrieved = cache.get_batch(keys, target_device=device)

        # Verify all patterns retrieved correctly
        for key, original in patterns.items():
            assert key in retrieved
            assert retrieved[key] is not None
            torch.testing.assert_close(retrieved[key].cpu(), original)

    def test_memory_limit(self):
        """Test that cache respects GPU memory limit."""
        cache = OptimizedPatternCache(
            max_gpu_patterns=10,
            gpu_memory_limit_mb=1.0,  # Very small limit
        )

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Try to store patterns that exceed memory limit
        for i in range(5):
            key = f"large_pattern_{i}"
            # Each pattern is ~4MB (1M elements * 4 bytes)
            pattern = torch.randn(1_000_000, device="cuda")
            cache.put(key, pattern, store_on_gpu=True)

        stats = cache.get_stats()
        # Should not exceed memory limit
        assert stats["gpu_memory_used_mb"] <= 1.5  # Allow small overhead

    def test_pattern_pinning(self):
        """Test pinning patterns to GPU."""
        cache = OptimizedPatternCache()

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create and store pattern
        pattern = torch.arange(100)
        key = "pinned_pattern"
        cache.put(key, pattern)

        # Pin to GPU
        success = cache.pin_pattern(key, torch.device("cuda"))
        assert success

        # Clear GPU cache to test pinning persistence
        cache._gpu_cache.clear()

        # Re-pin and verify it stays
        success = cache.pin_pattern(key, torch.device("cuda"))
        assert success

        # Access count should be high (pinned)
        assert cache._access_counts[key] >= 1000

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = OptimizedPatternCache()

        # Initial stats
        stats = cache.get_stats()
        assert stats["gpu_hits"] == 0
        assert stats["cpu_hits"] == 0
        assert stats["misses"] == 0

        # Add pattern and access
        cache.put("test", torch.tensor([1, 2, 3]))

        # First access - CPU hit
        _ = cache.get("test")
        stats = cache.get_stats()
        assert stats["cpu_hits"] == 1

        # Access non-existent - miss
        _ = cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_clear_cache(self):
        """Test cache clearing."""
        cache = OptimizedPatternCache()

        # Add patterns
        for i in range(5):
            cache.put(f"pattern_{i}", torch.arange(i * 10))

        stats = cache.get_stats()
        assert stats["cpu_cache_size"] == 5

        # Clear cache
        cache.clear()

        stats = cache.get_stats()
        assert stats["cpu_cache_size"] == 0
        assert stats["gpu_cache_size"] == 0
        assert stats["gpu_memory_used_mb"] == 0.0

    def test_global_cache(self):
        """Test global cache singleton."""
        clear_optimized_cache()

        cache1 = get_optimized_pattern_cache()
        cache2 = get_optimized_pattern_cache()

        # Should be same instance
        assert cache1 is cache2

        # Test pattern sharing
        cache1.put("shared", torch.tensor([1, 2, 3]))
        retrieved = cache2.get("shared")
        assert retrieved is not None
        torch.testing.assert_close(retrieved, torch.tensor([1, 2, 3]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
