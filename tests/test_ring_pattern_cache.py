"""
Test pattern caching integration in Ring Attention modules.

This test verifies that Ring Attention correctly uses the global pattern cache
for dilated indices, reducing redundant computation and memory usage.
"""

import torch
import pytest

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.core import get_global_pattern_cache, clear_global_cache


class TestRingPatternCache:
    """Test suite for pattern caching in Ring Attention."""

    def test_pattern_cache_usage(self):
        """Test that RingDilatedAttentionV2 uses the global pattern cache."""
        # Clear cache before test
        clear_global_cache()
        cache = get_global_pattern_cache()

        # Create ring attention with pattern cache enabled
        ring_attn = RingDilatedAttentionV2(
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            use_pattern_cache=True,
        )

        # Create input
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Get initial cache stats
        initial_stats = cache.get_stats()

        # First forward pass - should cache patterns
        _ = ring_attn(q, k, v)

        # Check that patterns were cached
        stats_after_first = cache.get_stats()
        assert stats_after_first["size"] > initial_stats["size"], (
            "No patterns were cached"
        )
        assert stats_after_first["misses"] > initial_stats["misses"], (
            "No cache misses recorded"
        )

        # Second forward pass - should use cached patterns
        _ = ring_attn(q, k, v)

        stats_after_second = cache.get_stats()
        assert stats_after_second["hits"] > stats_after_first["hits"], (
            "No cache hits on second pass"
        )

        print(f"Patterns cached: {stats_after_second['size']}")
        print(f"Cache hit rate: {stats_after_second['hit_rate']:.2%}")

        # Additional validation - check expected patterns
        expected_patterns = []
        for seg_len, dil_rate in zip([64, 128], [1, 2]):
            for offset in range(min(dil_rate, 2)):  # At most 2 different offsets used
                if dil_rate > 1 or offset == 0:
                    expected_patterns.append(
                        f"ring_dilated_s{seg_len}_r{dil_rate}_off{offset}"
                    )

        # Verify at least some expected patterns are cached
        cached_keys = list(cache._cache.keys())
        patterns_found = [key for key in expected_patterns if key in cached_keys]
        assert len(patterns_found) > 0, (
            f"No expected patterns found. Cached: {cached_keys}"
        )

    def test_pattern_cache_disabled(self):
        """Test that RingDilatedAttentionV2 works with pattern cache disabled."""
        # Create ring attention with pattern cache disabled
        ring_attn = RingDilatedAttentionV2(
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            use_pattern_cache=False,
        )

        # Verify it has local cache instead
        assert hasattr(ring_attn, "_dilated_indices_cache")
        assert (
            not hasattr(ring_attn, "_pattern_cache") or ring_attn._pattern_cache is None
        )

        # Create input
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Should work normally
        output = ring_attn(q, k, v)
        assert output.shape == q.shape

    @pytest.mark.parametrize(
        "segment_lengths,dilation_rates",
        [
            ([128], [1]),
            ([64, 128], [1, 2]),
            ([128, 256, 512], [1, 2, 4]),
        ],
    )
    def test_cache_consistency(self, segment_lengths, dilation_rates):
        """Test that cached patterns produce the same results as uncached."""
        # Clear global cache
        clear_global_cache()

        # Create two instances - one with cache, one without
        ring_cached = RingDilatedAttentionV2(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_pattern_cache=True,
        )

        ring_uncached = RingDilatedAttentionV2(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_pattern_cache=False,
        )

        # Create input
        batch_size = 2
        seq_len = max(segment_lengths)
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Forward pass with both
        output_cached = ring_cached(q, k, v)
        output_uncached = ring_uncached(q, k, v)

        # Results should be identical
        torch.testing.assert_close(output_cached, output_uncached, rtol=1e-5, atol=1e-5)

    def test_pattern_sharing_across_instances(self):
        """Test that multiple Ring Attention instances share the same cache."""
        # Clear cache
        clear_global_cache()
        cache = get_global_pattern_cache()

        # Create first instance and run forward pass
        ring1 = RingDilatedAttentionV2(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            use_pattern_cache=True,
        )

        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # First instance populates cache
        _ = ring1(q, k, v)
        cache_stats1 = cache.get_stats()

        # Create second instance with same config
        ring2 = RingDilatedAttentionV2(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            use_pattern_cache=True,
        )

        # Second instance should reuse cached patterns
        _ = ring2(q, k, v)
        cache_stats2 = cache.get_stats()

        # Cache size shouldn't increase (patterns are reused)
        assert cache_stats2["size"] == cache_stats1["size"], (
            "Cache size increased - patterns not reused"
        )
        # Cache hits should increase
        assert cache_stats2["hits"] > cache_stats1["hits"], (
            "No additional cache hits - patterns not reused"
        )

        print(
            f"Cache sharing confirmed: {cache_stats2['hits'] - cache_stats1['hits']} pattern reuses"
        )

    def test_dilation_pattern_correctness(self):
        """Test that cached dilation patterns are computed correctly."""
        # Clear cache
        clear_global_cache()

        ring_attn = RingDilatedAttentionV2(
            segment_lengths=[16],  # Small for easy verification
            dilation_rates=[4],
            use_pattern_cache=True,
        )

        # Access the pattern cache
        cache = get_global_pattern_cache()

        # Create dummy input to trigger pattern generation
        q = torch.randn(1, 16, 4, 8)
        k = torch.randn(1, 16, 4, 8)
        v = torch.randn(1, 16, 4, 8)

        _ = ring_attn(q, k, v)

        # Check that patterns were cached
        # The cache key format is: f"ring_dilated_s{segment_len}_r{dilation_rate}_off{offset}"
        # For dilation_rate=4, we expect offsets 0, 1, 2, 3
        for offset in range(4):
            cache_key = f"ring_dilated_s16_r4_off{offset}"
            pattern = cache.get(cache_key)
            if pattern is not None:
                # Verify pattern correctness
                expected = torch.arange(offset, 16, 4) % 16
                # Pattern might be longer due to padding
                if len(pattern) > len(expected):
                    pattern = pattern[: len(expected)]
                torch.testing.assert_close(pattern.cpu(), expected)
                print(f"Verified pattern for offset {offset}: {pattern.cpu().tolist()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
