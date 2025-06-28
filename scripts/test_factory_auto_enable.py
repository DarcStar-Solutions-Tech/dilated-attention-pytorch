#!/usr/bin/env python3
"""
Quick test script to verify factory auto-enable functionality.
"""

import torch
from dilated_attention_pytorch.core import (
    create_dilated_attention,
    create_multihead_dilated_attention,
)


def test_auto_enable():
    """Test auto-enable functionality."""
    print("Testing Factory Auto-Enable Functionality\n")

    # Test 1: Long sequences should auto-enable memory pool
    print("1. Testing long sequences (8192 tokens):")
    attention1 = create_dilated_attention(
        "improved", segment_lengths=[2048, 4096, 8192], dilation_rates=[1, 2, 4]
    )
    print(f"   enable_memory_pool: {getattr(attention1, 'enable_memory_pool', 'N/A')}")
    print(f"   lightweight_pool: {getattr(attention1, 'lightweight_pool', 'N/A')}")
    print(
        f"   Has _memory_pool: {hasattr(attention1, '_memory_pool') and attention1._memory_pool is not None}"
    )

    # Test 2: Short sequences should not enable memory pool
    print("\n2. Testing short sequences (1024 tokens):")
    attention2 = create_dilated_attention(
        "improved", segment_lengths=[256, 512, 1024], dilation_rates=[1, 2, 4]
    )
    print(f"   enable_memory_pool: {getattr(attention2, 'enable_memory_pool', 'N/A')}")
    print(
        f"   Has _memory_pool: {hasattr(attention2, '_memory_pool') and attention2._memory_pool is not None}"
    )

    # Test 3: Medium sequences should use lightweight pool
    print("\n3. Testing medium sequences (4096 tokens):")
    attention3 = create_dilated_attention(
        "improved", segment_lengths=[1024, 2048, 4096], dilation_rates=[1, 2, 4]
    )
    print(f"   enable_memory_pool: {getattr(attention3, 'enable_memory_pool', 'N/A')}")
    print(f"   lightweight_pool: {getattr(attention3, 'lightweight_pool', 'N/A')}")

    # Test 4: User override
    print("\n4. Testing user override (explicitly disable):")
    attention4 = create_dilated_attention(
        "improved",
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        enable_memory_pool=False,
    )
    print(f"   enable_memory_pool: {getattr(attention4, 'enable_memory_pool', 'N/A')}")
    print(
        f"   Has _memory_pool: {hasattr(attention4, '_memory_pool') and attention4._memory_pool is not None}"
    )

    # Test 5: Multihead attention
    print("\n5. Testing multihead attention:")
    multihead = create_multihead_dilated_attention(
        "improved",
        embed_dim=768,
        num_heads=12,
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
    )
    print(f"   Successfully created: {multihead is not None}")
    print(f"   Type: {type(multihead).__name__}")

    # Test 6: Special implementations
    if torch.cuda.is_available():
        print("\n6. Testing special implementations:")

        # Ring attention
        try:
            ring = create_dilated_attention(
                "ring", segment_lengths=[512, 1024], dilation_rates=[1, 2]
            )
            print(f"   Ring attention created: {ring is not None}")
        except Exception as e:
            print(f"   Ring attention error: {e}")

        # Block sparse
        try:
            sparse = create_dilated_attention(
                "block_sparse_ring", segment_lengths=[512, 1024], dilation_rates=[1, 2]
            )
            print(f"   Block sparse created: {sparse is not None}")
        except Exception as e:
            print(f"   Block sparse error: {e}")

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_auto_enable()
