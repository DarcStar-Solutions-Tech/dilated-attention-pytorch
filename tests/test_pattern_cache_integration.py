"""
Test pattern cache integration across all attention variants.
"""

import torch
import pytest

from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
    ImprovedMultiheadDilatedAttention,
)
from dilated_attention_pytorch.core import clear_global_cache, get_global_pattern_cache


@pytest.mark.parametrize(
    "attention_class,use_multihead",
    [
        (DilatedAttention, False),
        (ImprovedDilatedAttention, False),
        (MultiheadDilatedAttention, True),
        (ImprovedMultiheadDilatedAttention, True),
    ],
)
def test_pattern_cache_integration(attention_class, use_multihead):
    """Test that all attention variants properly use pattern caching."""
    # Clear cache
    clear_global_cache()
    cache = get_global_pattern_cache()
    cache.reset_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create attention module
    if use_multihead:
        attention = attention_class(
            embed_dim=256,
            num_heads=8,
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
        ).to(device)
    else:
        attention = attention_class(
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
        ).to(device)

    # Create input
    batch_size = 2
    seq_len = 128

    if use_multihead:
        # Multihead expects (batch, seq, embed_dim)
        q = torch.randn(batch_size, seq_len, 256, device=device)
        k = torch.randn(batch_size, seq_len, 256, device=device)
        v = torch.randn(batch_size, seq_len, 256, device=device)
    else:
        # Raw attention expects (batch, seq, num_heads, head_dim)
        q = torch.randn(batch_size, seq_len, 8, 32, device=device)
        k = torch.randn(batch_size, seq_len, 8, 32, device=device)
        v = torch.randn(batch_size, seq_len, 8, 32, device=device)

    # First forward pass - should cache patterns
    initial_stats = cache.get_stats()
    _ = attention(q, k, v)
    after_first = cache.get_stats()

    # Verify patterns were cached
    patterns_cached = after_first["size"] - initial_stats["size"]
    assert patterns_cached > 0, f"{attention_class.__name__} did not cache any patterns"

    # Second forward pass - should use cached patterns
    _ = attention(q, k, v)
    after_second = cache.get_stats()

    # Verify cache hits increased
    hits_increased = after_second["hits"] > after_first["hits"]
    assert hits_increased, f"{attention_class.__name__} did not use cached patterns"

    print(f"{attention_class.__name__}:")
    print(f"  Patterns cached: {patterns_cached}")
    print(f"  Cache hits: {after_second['hits']}")
    print(f"  Hit rate: {after_second['hit_rate']:.2%}")


def test_cache_sharing_across_modules():
    """Test that multiple modules share the same cache."""
    # Clear cache
    clear_global_cache()
    cache = get_global_pattern_cache()

    # Create two modules with same configuration
    attention1 = DilatedAttention(
        segment_lengths=[128, 256],
        dilation_rates=[1, 2],
    )

    attention2 = ImprovedDilatedAttention(
        segment_lengths=[128, 256],
        dilation_rates=[1, 2],
    )

    # Create input
    batch_size = 2
    seq_len = 256
    num_heads = 8
    head_dim = 32

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # First module creates patterns
    cache.reset_stats()
    _ = attention1(q, k, v)
    stats1 = cache.get_stats()

    # Second module should reuse patterns
    _ = attention2(q, k, v)
    stats2 = cache.get_stats()

    # Verify cache was shared
    assert stats2["hits"] > stats1["hits"], "Modules did not share cache"
    print(f"Cache sharing verified: {stats2['hits']} hits after second module")


def test_different_configurations_cached_separately():
    """Test that different configurations get separate cache entries."""
    # Clear cache
    clear_global_cache()
    cache = get_global_pattern_cache()

    # Create modules with different configurations
    configs = [
        ([64, 128], [1, 2]),
        ([128, 256], [1, 2]),
        ([64, 128], [1, 4]),  # Different dilation rates
    ]

    total_patterns = 0

    for segment_lengths, dilation_rates in configs:
        attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        )

        # Create appropriate input size
        seq_len = max(segment_lengths)
        q = torch.randn(1, seq_len, 4, 32)
        k = torch.randn(1, seq_len, 4, 32)
        v = torch.randn(1, seq_len, 4, 32)

        # Run forward pass
        initial_size = cache.get_stats()["size"]
        _ = attention(q, k, v)
        new_size = cache.get_stats()["size"]

        patterns_added = new_size - initial_size
        total_patterns += patterns_added

        print(
            f"Config {segment_lengths}, {dilation_rates}: {patterns_added} patterns cached"
        )

    # Verify all configurations cached separately
    assert total_patterns >= len(configs), "Not all configurations cached separately"
    print(f"Total unique patterns cached: {total_patterns}")


if __name__ == "__main__":
    print("Testing pattern cache integration across attention variants...")
    test_pattern_cache_integration(DilatedAttention, False)
    test_pattern_cache_integration(ImprovedDilatedAttention, False)
    test_pattern_cache_integration(MultiheadDilatedAttention, True)
    test_pattern_cache_integration(ImprovedMultiheadDilatedAttention, True)

    print("\nTesting cache sharing...")
    test_cache_sharing_across_modules()

    print("\nTesting different configurations...")
    test_different_configurations_cached_separately()

    print("\nAll integration tests passed!")
