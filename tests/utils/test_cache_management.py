#!/usr/bin/env python3
"""
Test cache management in kernel implementations.
"""

import torch
import pytest
import gc

from dilated_attention_pytorch.kernels.cache_manager import (
    BoundedCache,
    CachedHilbertMixin,
)
from dilated_attention_pytorch.kernels import UnifiedHilbertAttention
from dilated_attention_pytorch.kernels.hilbert_attention_sparse_simple import (
    HilbertAttentionSparseSimple,
)


class TestBoundedCache:
    """Test the BoundedCache implementation."""

    def test_basic_functionality(self):
        """Test basic cache operations."""
        cache = BoundedCache(max_size=3)

        # Test put and get
        tensor1 = torch.randn(10, 10)
        cache.put("key1", tensor1)
        assert torch.equal(cache.get("key1"), tensor1)

        # Test missing key
        assert cache.get("missing") is None

        # Test contains
        assert "key1" in cache
        assert "missing" not in cache

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = BoundedCache(max_size=3)

        # Fill cache
        tensors = [torch.randn(10, 10) for _ in range(4)]
        for i, tensor in enumerate(tensors[:3]):
            cache.put(f"key{i}", tensor)

        assert len(cache) == 3

        # Add one more - should evict key0 (least recently used)
        cache.put("key3", tensors[3])
        assert len(cache) == 3
        assert "key0" not in cache
        assert "key1" in cache
        assert "key2" in cache
        assert "key3" in cache

        # Access key1 to make it more recent
        _ = cache.get("key1")

        # Add another - should evict key2 now
        cache.put("key4", torch.randn(10, 10))
        assert "key1" in cache  # Recently accessed
        assert "key2" not in cache  # Evicted
        assert "key3" in cache
        assert "key4" in cache

    def test_memory_limit(self):
        """Test memory-based eviction."""
        # Set very low memory limit (1MB)
        cache = BoundedCache(max_size=100, max_memory_mb=1.0)

        # Create tensors that are ~0.4MB each (1000x1000 float32)
        large_tensor_size = 1000
        tensor_size_mb = (large_tensor_size * large_tensor_size * 4) / (1024 * 1024)

        tensors = []
        for i in range(5):
            tensor = torch.randn(large_tensor_size, large_tensor_size)
            cache.put(f"key{i}", tensor)
            tensors.append(tensor)

        # Should have evicted some entries to stay under memory limit
        assert cache.memory_usage_mb <= 1.0 + tensor_size_mb  # Allow one tensor over
        assert len(cache) < 5

    def test_device_handling(self):
        """Test handling of tensors on different devices."""
        cache = BoundedCache(max_size=3)

        # CPU tensor
        cpu_tensor = torch.randn(10, 10)
        cache.put("cpu", cpu_tensor)

        if torch.cuda.is_available():
            # CUDA tensor
            cuda_tensor = torch.randn(10, 10, device="cuda")
            cache.put("cuda", cuda_tensor)

            # Retrieved tensors should maintain their device
            assert cache.get("cpu").device.type == "cpu"
            assert cache.get("cuda").device.type == "cuda"

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = BoundedCache(max_size=5, max_memory_mb=10.0, name="test_cache")

        # Add some tensors
        for i in range(3):
            cache.put(f"key{i}", torch.randn(100, 100))

        stats = cache.get_stats()
        assert stats["name"] == "test_cache"
        assert stats["size"] == 3
        assert stats["max_size"] == 5
        assert stats["memory_usage_mb"] > 0
        assert stats["max_memory_mb"] == 10.0


class TestKernelCacheManagement:
    """Test cache management in actual kernel implementations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hilbert_core_cache_management(self):
        """Test cache management in UnifiedHilbertAttention."""
        device = torch.device("cuda")
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=2,
        ).to(device)

        # Generate different sequence lengths
        seq_lengths = [128, 256, 512, 1024, 2048]

        # Create inputs and process them
        for seq_len in seq_lengths * 3:  # Repeat to test cache hits
            x = torch.randn(2, seq_len, 768, device=device)
            _ = module(x)

        # Check cache stats
        stats = module.get_cache_stats()
        assert stats["size"] <= 32  # Should respect max size
        assert stats["memory_usage_mb"] <= 100.0  # Should respect memory limit

        # Clear cache
        module.clear_cache()
        stats_after = module.get_cache_stats()
        assert stats_after["size"] == 0
        assert stats_after["memory_usage_mb"] == 0.0

    def test_sparse_cache_management(self):
        """Test cache management in sparse implementation."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = HilbertAttentionSparseSimple(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=4,
        ).to(device)

        # Process multiple sequences to populate cache
        for i in range(50):  # More than cache size
            seq_len = 128 * (i % 8 + 1)  # Various sequence lengths
            x = torch.randn(2, seq_len, 768, device=device)
            _ = module(x)

        # Check cache is bounded
        stats = module.get_cache_stats()
        assert stats["size"] <= 32
        assert stats["memory_usage_mb"] <= 50.0

        # Verify cache works correctly after many operations
        x_test = torch.randn(2, 512, 768, device=device)
        out1 = module(x_test, use_hilbert=True)
        out2 = module(x_test, use_hilbert=True)  # Should use cache
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_memory_pressure_scenario(self):
        """Test behavior under memory pressure."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create multiple modules to simulate memory pressure
        modules = []
        for i in range(10):
            module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=1,
            ).to(device)
            modules.append(module)

        # Process many sequences
        for module in modules:
            for seq_len in [512, 1024, 2048, 4096]:
                x = torch.randn(1, seq_len, 768, device=device)
                _ = module(x)

        # Check total memory usage is reasonable
        total_cache_memory = sum(
            m.get_cache_stats()["memory_usage_mb"] for m in modules
        )
        # Each module has 100MB limit, but with 10 modules we should see eviction
        assert (
            total_cache_memory <= 1000.0
        )  # Should not exceed individual limits summed

        # Clear all caches
        for module in modules:
            module.clear_cache()

        # Force garbage collection
        del modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestCachedHilbertMixin:
    """Test the CachedHilbertMixin functionality."""

    class DummyModule(CachedHilbertMixin, torch.nn.Module):
        """Dummy module for testing mixin."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def test_mixin_initialization(self):
        """Test mixin initializes cache correctly."""
        module = self.DummyModule()

        # Should have cache
        assert hasattr(module, "_hilbert_cache")
        assert isinstance(module._hilbert_cache, BoundedCache)

        # Check configuration (defaults)
        stats = module.get_cache_stats()
        assert stats["max_size"] == 32  # Default value
        assert stats["max_memory_mb"] == 100.0  # Default value

    def test_mixin_cache_operations(self):
        """Test mixin cache operations."""
        module = self.DummyModule()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get mapping - should create and cache
        mapping1 = module._get_cached_hilbert_mapping(512, device)
        assert mapping1.shape == (512,)

        # Get again - should use cache
        mapping2 = module._get_cached_hilbert_mapping(512, device)
        assert torch.equal(mapping1, mapping2)

        # Clear cache
        module.clear_cache()
        stats = module.get_cache_stats()
        assert stats["size"] == 0


if __name__ == "__main__":
    # Run tests
    test_cache = TestBoundedCache()
    test_cache.test_basic_functionality()
    test_cache.test_lru_eviction()
    test_cache.test_memory_limit()
    test_cache.test_device_handling()
    test_cache.test_cache_stats()
    print("✓ All BoundedCache tests passed")

    test_kernel = TestKernelCacheManagement()
    if torch.cuda.is_available():
        test_kernel.test_hilbert_core_cache_management()
    test_kernel.test_sparse_cache_management()
    test_kernel.test_memory_pressure_scenario()
    print("✓ All kernel cache management tests passed")

    test_mixin = TestCachedHilbertMixin()
    test_mixin.test_mixin_initialization()
    test_mixin.test_mixin_cache_operations()
    print("✓ All mixin tests passed")

    print("\nAll cache management tests completed successfully!")
