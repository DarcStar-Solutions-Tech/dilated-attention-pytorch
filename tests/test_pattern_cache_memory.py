"""
Test memory efficiency of pattern caching.

This test verifies that pattern caching doesn't introduce memory leaks
and that cached patterns are properly managed.
"""

import gc
import torch

from dilated_attention_pytorch import DilatedAttention
from dilated_attention_pytorch.core import clear_global_cache, get_global_pattern_cache


def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0


def test_pattern_cache_memory_lifecycle():
    """Test that pattern cache properly manages memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear cache and force garbage collection
    clear_global_cache()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB")

    # Create attention module
    attention = DilatedAttention(
        segment_lengths=[512, 1024, 2048],
        dilation_rates=[1, 2, 4],
    ).to(device)

    # Create large input
    batch_size = 4
    seq_len = 2048
    num_heads = 16
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    after_allocation = get_memory_usage()
    print(f"After allocation: {after_allocation:.2f} MB")

    # Run forward passes to populate cache
    for i in range(10):
        _ = attention(q, k, v)
        if i == 0:
            first_pass_memory = get_memory_usage()
            print(f"After first pass: {first_pass_memory:.2f} MB")

    after_passes = get_memory_usage()
    print(f"After 10 passes: {after_passes:.2f} MB")

    # Check cache statistics
    cache = get_global_pattern_cache()
    stats = cache.get_stats()
    print(f"Cache size: {stats['size']} patterns")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")

    # Clear cache and check memory
    clear_global_cache()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    after_clear = get_memory_usage()
    print(f"After cache clear: {after_clear:.2f} MB")

    # Delete tensors and check final memory
    del q, k, v, attention
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    final_memory = get_memory_usage()
    print(f"Final memory: {final_memory:.2f} MB")

    # Memory should return close to initial state
    memory_leak = final_memory - initial_memory
    print(f"Memory difference: {memory_leak:.2f} MB")

    # Assert no significant memory leak (allow 10MB tolerance)
    assert memory_leak < 10, f"Potential memory leak detected: {memory_leak:.2f} MB"


def test_cache_size_limits():
    """Test that cache respects size limits."""
    # Create cache with small size limit
    cache = get_global_pattern_cache()
    cache.max_size = 5  # Set small limit

    # Add many patterns
    for i in range(20):
        pattern = torch.randn(100, 100)
        cache.put(f"pattern_{i}", pattern)

    # Check cache size doesn't exceed limit
    stats = cache.get_stats()
    assert stats["size"] <= 5, f"Cache size {stats['size']} exceeds limit of 5"
    assert stats["evictions"] > 0, "Expected evictions when exceeding cache size"

    # Verify LRU behavior
    # Recently added patterns should be in cache
    assert cache.get("pattern_19") is not None
    assert cache.get("pattern_18") is not None

    # Early patterns should be evicted
    assert cache.get("pattern_0") is None
    assert cache.get("pattern_1") is None

    # Reset cache size
    cache.max_size = 100


def test_pattern_memory_efficiency():
    """Test memory efficiency of cached patterns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear cache
    clear_global_cache()
    cache = get_global_pattern_cache()

    # Create patterns of different sizes
    small_pattern = torch.arange(10, device=device)
    medium_pattern = torch.arange(1000, device=device)
    large_pattern = torch.arange(100000, device=device)

    # Cache patterns (should move to CPU)
    cache.put("small", small_pattern, move_to_cpu=True)
    cache.put("medium", medium_pattern, move_to_cpu=True)
    cache.put("large", large_pattern, move_to_cpu=True)

    # Verify patterns are on CPU
    cached_small = cache.get("small")
    cached_medium = cache.get("medium")
    cached_large = cache.get("large")

    assert cached_small.device.type == "cpu"
    assert cached_medium.device.type == "cpu"
    assert cached_large.device.type == "cpu"

    # Test device transfer efficiency
    if device == "cuda":
        # Get patterns on GPU
        gpu_small = cache.get("small", target_device=torch.device("cuda"))
        gpu_medium = cache.get("medium", target_device=torch.device("cuda"))
        gpu_large = cache.get("large", target_device=torch.device("cuda"))

        assert gpu_small.device.type == "cuda"
        assert gpu_medium.device.type == "cuda"
        assert gpu_large.device.type == "cuda"

        # Verify values are correct
        assert torch.equal(gpu_small.cpu(), cached_small)
        assert torch.equal(gpu_medium.cpu(), cached_medium)
        assert torch.equal(gpu_large.cpu(), cached_large)


if __name__ == "__main__":
    print("Testing pattern cache memory lifecycle...")
    test_pattern_cache_memory_lifecycle()

    print("\nTesting cache size limits...")
    test_cache_size_limits()

    print("\nTesting pattern memory efficiency...")
    test_pattern_memory_efficiency()

    print("\nAll memory tests passed!")
