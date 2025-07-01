"""
Debug Ring Attention pattern cache usage.
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective as RingDilatedAttentionV2,
)
from dilated_attention_pytorch.core import get_global_pattern_cache, clear_global_cache


def debug_pattern_generation():
    """Debug pattern generation and caching."""
    clear_global_cache()
    cache = get_global_pattern_cache()

    print("Creating Ring Attention with pattern cache...")
    ring_attn = RingDilatedAttentionV2(
        segment_lengths=[64, 128],
        dilation_rates=[1, 2],
        use_pattern_cache=True,
    )

    # Small input
    q = torch.randn(1, 128, 4, 16)
    k = torch.randn(1, 128, 4, 16)
    v = torch.randn(1, 128, 4, 16)

    print("\nBefore forward pass:")
    stats = cache.get_stats()
    print(f"  Cache size: {stats['size']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")

    # First forward pass
    print("\nRunning first forward pass...")
    _ = ring_attn(q, k, v)

    stats = cache.get_stats()
    print("\nAfter first forward pass:")
    print(f"  Cache size: {stats['size']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")

    # List cached keys
    print("\nCached keys:")
    for key in cache._cache.keys():
        print(f"  - {key}")

    # Second forward pass
    print("\nRunning second forward pass...")
    _ = ring_attn(q, k, v)

    stats = cache.get_stats()
    print("\nAfter second forward pass:")
    print(f"  Cache size: {stats['size']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")

    # Test specific pattern access
    print("\nTesting specific pattern access...")

    # Try to access a pattern that should be cached
    test_key = "ring_dilated_s64_r1_off0"
    pattern = cache.get(test_key)
    print(f"  Pattern for {test_key}: {pattern}")

    test_key = "ring_dilated_s128_r2_off0"
    pattern = cache.get(test_key)
    print(
        f"  Pattern for {test_key}: {pattern[:10] if pattern is not None else None}..."
    )


if __name__ == "__main__":
    debug_pattern_generation()
