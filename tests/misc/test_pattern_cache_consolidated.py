"""Consolidated pattern cache tests.

This file consolidates multiple pattern cache test files:
- test_pattern_cache.py
- test_pattern_cache_integration.py
- test_pattern_cache_memory.py
- test_ring_pattern_cache.py
- test_optimized_pattern_cache.py
"""

import pytest
import torch
import time


class TestPatternCacheCore:
    """Core pattern cache functionality tests."""

    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reset cache before each test
        from dilated_attention_pytorch.core import reset_global_pattern_cache

        reset_global_pattern_cache()

    def test_pattern_cache_creation(self):
        """Test pattern cache creation and basic operations."""
        from dilated_attention_pytorch.core import get_global_pattern_cache

        cache = get_global_pattern_cache()

        # Test cache is empty initially
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_pattern_cache_hit_miss(self):
        """Test cache hits and misses."""
        from dilated_attention_pytorch.core import PatternCache

        cache = PatternCache(max_size=10)

        # Create a key
        key = (1024, 8, 1, "segment")

        # First access - miss
        pattern = cache.get(key)
        assert pattern is None

        # Store pattern
        test_pattern = torch.ones(10, 10)
        cache.put(key, test_pattern)

        # Second access - hit
        pattern = cache.get(key)
        assert pattern is not None
        assert torch.equal(pattern, test_pattern)

        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_pattern_cache_with_ring_attention(self):
        """Test pattern cache integration with ring attention."""
        from dilated_attention_pytorch import (
            RingDilatedAttentionHilbertGPUOptimized as RingDilatedAttentionProduction,
        )

        # Create model with pattern cache
        model = RingDilatedAttentionProduction(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            use_pattern_cache=True,
        ).to(self.device)

        # Test data
        batch_size, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)

        # First forward pass - cache miss
        output1 = model(q, k, v)

        # Second forward pass - cache hit (same sequence length)
        output2 = model(q, k, v)

        # Outputs should be identical when using same inputs
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_pattern_cache_memory_efficiency(self):
        """Test memory efficiency of pattern cache."""
        from dilated_attention_pytorch.core import PatternCache

        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory test")

        cache = PatternCache(max_size=100, device="cpu")  # Store on CPU

        # Generate large patterns
        patterns = []
        for i in range(10):
            size = 1000 * (i + 1)
            pattern = torch.ones(size, size, device="cpu")
            key = (size, 8, 1, f"pattern_{i}")
            cache.put(key, pattern)
            patterns.append((key, pattern))

        # Verify all patterns are cached
        for key, expected in patterns:
            cached = cache.get(key)
            assert cached is not None
            assert torch.equal(cached, expected)

    def test_pattern_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        from dilated_attention_pytorch.core import PatternCache

        cache = PatternCache(max_size=3)

        # Fill cache
        for i in range(4):
            key = (i, i, i, f"pattern_{i}")
            pattern = torch.ones(10, 10) * i
            cache.put(key, pattern)

        # First pattern should be evicted
        assert cache.get((0, 0, 0, "pattern_0")) is None

        # Others should still be there
        for i in range(1, 4):
            key = (i, i, i, f"pattern_{i}")
            assert cache.get(key) is not None


class TestPatternCacheIntegration:
    """Integration tests for pattern cache across different modules."""

    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from dilated_attention_pytorch.core import reset_global_pattern_cache

        reset_global_pattern_cache()

    def test_global_cache_sharing(self):
        """Test that global cache is shared across modules."""
        from dilated_attention_pytorch import (
            RingDilatedAttentionHilbertGPUOptimized as RingDilatedAttentionProduction,
        )
        from dilated_attention_pytorch.core import get_global_pattern_cache

        # Create two models with same config
        model1 = RingDilatedAttentionProduction(
            segment_lengths=[512],
            dilation_rates=[1],
            use_pattern_cache=True,
        ).to(self.device)

        model2 = RingDilatedAttentionProduction(
            segment_lengths=[512],
            dilation_rates=[1],
            use_pattern_cache=True,
        ).to(self.device)

        # Get cache stats before
        cache = get_global_pattern_cache()
        stats_before = cache.get_stats()

        # Run model1
        batch_size, seq_len, num_heads, head_dim = 1, 512, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)

        _ = model1(q, k, v)

        # Check cache was populated
        stats_mid = cache.get_stats()
        assert stats_mid["misses"] > stats_before["misses"]

        # Run model2 - should hit cache
        _ = model2(q, k, v)

        stats_after = cache.get_stats()
        assert stats_after["hits"] > stats_mid["hits"]

    def test_cache_performance_benefit(self):
        """Test performance improvement from pattern cache."""
        from dilated_attention_pytorch import (
            RingDilatedAttentionHilbertGPUOptimized as RingDilatedAttentionProduction,
        )

        # Model with cache
        model_cached = RingDilatedAttentionProduction(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            use_pattern_cache=True,
        ).to(self.device)

        # Test data
        batch_size, seq_len, num_heads, head_dim = 2, 2048, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)

        # Warmup
        for _ in range(3):
            _ = model_cached(q, k, v)

        # Time cached runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            _ = model_cached(q, k, v)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        cached_time = time.perf_counter() - start

        # Create uncached model
        model_uncached = RingDilatedAttentionProduction(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            use_pattern_cache=False,
        ).to(self.device)

        # Time uncached runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            _ = model_uncached(q, k, v)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        uncached_time = time.perf_counter() - start

        print(f"Cached time: {cached_time:.4f}s")
        print(f"Uncached time: {uncached_time:.4f}s")
        print(f"Speedup: {uncached_time / cached_time:.2f}x")

        # Cached should be faster (allowing some variance)
        # Note: Speedup might be small for small sequences
        assert cached_time <= uncached_time * 1.1  # Allow 10% variance


class TestPatternCacheOptimizations:
    """Test pattern cache optimizations."""

    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_gpu_pattern_cache(self):
        """Test GPU-resident pattern cache."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from dilated_attention_pytorch.core import PatternCache

        # Create GPU cache
        cache = PatternCache(max_size=10, device="cuda")

        # Store patterns on GPU
        key = (1024, 8, 1, "gpu_pattern")
        pattern = torch.ones(1024, 1024, device="cuda")
        cache.put(key, pattern)

        # Retrieve and verify it's on GPU
        cached = cache.get(key)
        assert cached is not None
        assert cached.is_cuda
        assert torch.equal(cached, pattern)

    def test_pattern_cache_different_dtypes(self):
        """Test pattern cache with different data types."""
        from dilated_attention_pytorch.core import PatternCache

        cache = PatternCache(max_size=10)

        dtypes = [torch.float32, torch.float16, torch.int32, torch.bool]

        for i, dtype in enumerate(dtypes):
            key = (100, 8, 1, f"dtype_{dtype}")

            if dtype == torch.bool:
                pattern = torch.ones(100, 100, dtype=dtype)
            else:
                pattern = torch.ones(100, 100, dtype=dtype) * (i + 1)

            cache.put(key, pattern)

            # Retrieve and verify
            cached = cache.get(key)
            assert cached is not None
            assert cached.dtype == dtype
            assert torch.equal(cached, pattern)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
