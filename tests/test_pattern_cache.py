"""
Tests for pattern caching functionality.

This module tests the PatternCache and DilatedPatternCache classes to ensure
they correctly cache and retrieve patterns with proper memory management.
"""

import threading
import time

import pytest
import torch

from dilated_attention_pytorch.core import (
    DilatedPatternCache,
    PatternCache,
    clear_global_cache,
    get_global_pattern_cache,
)


class TestPatternCache:
    """Test suite for basic PatternCache functionality."""

    def test_basic_cache_operations(self):
        """Test basic get/put operations."""
        cache = PatternCache(max_size=10)

        # Test putting and getting a tensor
        key = "test_pattern"
        pattern = torch.randn(100, 100)

        cache.put(key, pattern)
        retrieved = cache.get(key)

        assert retrieved is not None
        assert torch.equal(pattern, retrieved)

    def test_cpu_storage(self):
        """Test that patterns are stored on CPU."""
        cache = PatternCache()

        # Create a CUDA tensor if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pattern = torch.randn(50, 50, device=device)

        cache.put("gpu_pattern", pattern, move_to_cpu=True)

        # Check that cached pattern is on CPU
        cached_pattern = cache.get("gpu_pattern")
        assert cached_pattern.device.type == "cpu"

    def test_device_transfer(self):
        """Test transferring patterns to target device."""
        cache = PatternCache()

        # Store pattern on CPU
        pattern = torch.randn(30, 30, device="cpu")
        cache.put("test", pattern)

        # Retrieve to different device if available
        if torch.cuda.is_available():
            retrieved = cache.get("test", target_device=torch.device("cuda"))
            assert retrieved.device.type == "cuda"
            assert torch.equal(pattern, retrieved.cpu())

    def test_tuple_patterns(self):
        """Test caching tuple patterns (e.g., row and col indices)."""
        cache = PatternCache()

        row_idx = torch.tensor([0, 1, 2, 3])
        col_idx = torch.tensor([1, 2, 3, 4])
        pattern = (row_idx, col_idx)

        cache.put("sparse_pattern", pattern)
        retrieved = cache.get("sparse_pattern")

        assert retrieved is not None
        assert len(retrieved) == 2
        assert torch.equal(row_idx, retrieved[0])
        assert torch.equal(col_idx, retrieved[1])

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = PatternCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.put(f"pattern_{i}", torch.tensor([i]))

        # Access pattern_0 to make it most recently used
        cache.get("pattern_0")

        # Add new pattern, should evict pattern_1
        cache.put("pattern_3", torch.tensor([3]))

        # Check eviction
        assert cache.get("pattern_1") is None  # Evicted
        assert cache.get("pattern_0") is not None  # Still present
        assert cache.get("pattern_2") is not None  # Still present
        assert cache.get("pattern_3") is not None  # Newly added

    def test_statistics(self):
        """Test cache statistics tracking."""
        cache = PatternCache(enable_stats=True)

        # Generate some hits and misses
        cache.put("pattern", torch.tensor([1, 2, 3]))

        cache.get("pattern")  # Hit
        cache.get("pattern")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["size"] == 1

    def test_thread_safety(self):
        """Test thread-safe operations."""
        cache = PatternCache(max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    key = f"thread_{thread_id}_pattern_{i}"
                    pattern = torch.tensor([thread_id, i])
                    cache.put(key, pattern)

                    retrieved = cache.get(key)
                    if not torch.equal(pattern, retrieved):
                        errors.append(f"Mismatch in thread {thread_id}")
            except Exception as e:
                errors.append(f"Error in thread {thread_id}: {e}")

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestDilatedPatternCache:
    """Test suite for DilatedPatternCache functionality."""

    def test_dilated_indices_caching(self):
        """Test caching of dilated indices."""
        cache = DilatedPatternCache()

        # Create dilated indices
        seq_len = 1024
        segment_lengths = (256, 512)
        dilation_rates = (1, 2)

        indices = torch.arange(0, 256, 2)  # Dilated indices

        # Store indices
        cache.put_dilated_indices(indices, seq_len, segment_lengths, dilation_rates)

        # Retrieve indices
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        retrieved = cache.get_dilated_indices(
            seq_len, segment_lengths, dilation_rates, device=device
        )

        assert retrieved is not None
        assert torch.equal(indices, retrieved.cpu())
        assert retrieved.device.type == device.type

    def test_sparse_pattern_caching(self):
        """Test caching of sparse patterns."""
        cache = DilatedPatternCache()

        seq_len = 512
        pattern_type = "local_window"
        sparsity_ratio = 0.9
        block_size = 32

        # Create sparse pattern
        row_idx = torch.tensor([0, 1, 2, 10, 11, 12])
        col_idx = torch.tensor([0, 1, 2, 10, 11, 12])

        # Store pattern
        cache.put_sparse_pattern(
            row_idx, col_idx, seq_len, pattern_type, sparsity_ratio, block_size
        )

        # Retrieve pattern
        retrieved = cache.get_sparse_pattern(
            seq_len,
            pattern_type,
            sparsity_ratio,
            block_size,
            device=torch.device("cpu"),
        )

        assert retrieved is not None
        assert len(retrieved) == 2
        assert torch.equal(row_idx, retrieved[0])
        assert torch.equal(col_idx, retrieved[1])

    def test_key_generation(self):
        """Test cache key generation for different patterns."""
        # Test dilated key
        key1 = DilatedPatternCache._make_dilated_key(1024, (256, 512), (1, 2))
        key2 = DilatedPatternCache._make_dilated_key(1024, (256, 512), (1, 2))
        key3 = DilatedPatternCache._make_dilated_key(2048, (256, 512), (1, 2))

        assert key1 == key2  # Same parameters
        assert key1 != key3  # Different seq_len

        # Test sparse key
        skey1 = DilatedPatternCache._make_sparse_key(512, "local_window", 0.9, 32)
        skey2 = DilatedPatternCache._make_sparse_key(512, "local_window", 0.9, 32)
        skey3 = DilatedPatternCache._make_sparse_key(512, "local_window", 0.9, None)

        assert skey1 == skey2  # Same parameters
        assert skey1 != skey3  # Different block_size


class TestGlobalPatternCache:
    """Test suite for global pattern cache functionality."""

    def test_global_cache_singleton(self):
        """Test that global cache is a singleton."""
        cache1 = get_global_pattern_cache()
        cache2 = get_global_pattern_cache()

        assert cache1 is cache2

    def test_global_cache_clear(self):
        """Test clearing global cache."""
        cache = get_global_pattern_cache()

        # Add some patterns
        cache.put("test1", torch.tensor([1, 2, 3]))
        cache.put("test2", torch.tensor([4, 5, 6]))

        assert cache.get("test1") is not None
        assert cache.get("test2") is not None

        # Clear cache
        clear_global_cache()

        assert cache.get("test1") is None
        assert cache.get("test2") is None

    def test_global_cache_with_attention_modules(self):
        """Test global cache usage with attention modules."""
        from dilated_attention_pytorch import DilatedAttention

        # Create attention module
        attention = DilatedAttention(segment_lengths=[128, 256], dilation_rates=[1, 2])

        # Create input
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # First forward pass - should cache patterns
        cache = get_global_pattern_cache()
        initial_stats = cache.get_stats()

        _ = attention(q, k, v)

        # Check that patterns were cached
        stats_after_first = cache.get_stats()
        assert stats_after_first["size"] > initial_stats["size"]

        # Second forward pass - should use cached patterns
        _ = attention(q, k, v)

        stats_after_second = cache.get_stats()
        assert stats_after_second["hits"] > stats_after_first["hits"]


class TestPatternCacheIntegration:
    """Integration tests for pattern caching with attention modules."""

    @pytest.mark.parametrize(
        "attention_class,kwargs",
        [
            (
                "DilatedAttention",
                {"segment_lengths": [64, 128], "dilation_rates": [1, 2]},
            ),
            (
                "ImprovedDilatedAttention",
                {"segment_lengths": [64, 128], "dilation_rates": [1, 2]},
            ),
        ],
    )
    def test_pattern_cache_reduces_allocations(self, attention_class, kwargs):
        """Test that pattern caching reduces tensor allocations."""
        from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention

        # Map class names to actual classes
        class_map = {
            "DilatedAttention": DilatedAttention,
            "ImprovedDilatedAttention": ImprovedDilatedAttention,
        }

        AttentionClass = class_map[attention_class]

        # Clear global cache
        clear_global_cache()

        # Create attention module
        attention = AttentionClass(**kwargs)

        # Create input
        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Count torch.arange calls by monitoring the pattern cache
        cache = get_global_pattern_cache()

        # First forward pass
        initial_size = cache.get_stats()["size"]
        _ = attention(q, k, v)
        first_pass_size = cache.get_stats()["size"]
        first_pass_patterns_cached = first_pass_size - initial_size

        # Second forward pass
        cache_stats_before = cache.get_stats()
        _ = attention(q, k, v)
        cache_stats_after = cache.get_stats()

        # Check that cache was hit on second pass
        cache_hits_increased = cache_stats_after["hits"] > cache_stats_before["hits"]

        # Verify caching behavior
        assert first_pass_patterns_cached > 0  # Patterns were cached on first pass
        assert cache_hits_increased  # Cache was hit on second pass

    def test_cache_performance_improvement(self):
        """Test that caching improves performance."""
        from dilated_attention_pytorch import DilatedAttention

        # Clear cache
        clear_global_cache()

        # Create attention with larger segments
        attention = DilatedAttention(
            segment_lengths=[512, 1024, 2048], dilation_rates=[1, 2, 4]
        )

        # Create larger input
        batch_size = 4
        seq_len = 2048
        num_heads = 16
        head_dim = 64

        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Warmup
        _ = attention(q, k, v)

        # Time first forward pass (cold cache)
        clear_global_cache()
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(10):
            _ = attention(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        cold_cache_time = time.perf_counter() - start_time

        # Time subsequent passes (warm cache)
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(10):
            _ = attention(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        warm_cache_time = time.perf_counter() - start_time

        # Warm cache should be faster (allow small margin for timing variance)
        # Only check if running on GPU where the difference is more noticeable
        if device == "cuda":
            assert warm_cache_time <= cold_cache_time * 1.1  # Allow 10% margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
