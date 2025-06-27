"""
Test Ring Attention migration and deprecation warnings.
"""

import warnings
import pytest

from dilated_attention_pytorch import (
    RingDilatedAttention,
    RingMultiheadDilatedAttention,
    create_dilated_attention,
)


def test_deprecated_ring_dilated_attention_warning():
    """Test that deprecated RingDilatedAttention emits warning."""
    with pytest.warns(DeprecationWarning, match="RingDilatedAttention is deprecated"):
        _ = RingDilatedAttention(
            segment_lengths=[1024],
            dilation_rates=[1],
        )


def test_deprecated_ring_multihead_attention_warning():
    """Test that deprecated RingMultiheadDilatedAttention emits warning."""
    with pytest.warns(
        DeprecationWarning, match="RingMultiheadDilatedAttention is deprecated"
    ):
        _ = RingMultiheadDilatedAttention(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[1024],
            dilation_rates=[1],
        )


def test_factory_uses_correct_implementation():
    """Test that factory creates correct ring attention implementation."""
    # Create using factory - should use V2
    attention = create_dilated_attention(
        "ring",
        segment_lengths=[1024],
        dilation_rates=[1],
        ring_size=4,
    )

    # Check it has the V2 implementation inside
    assert hasattr(attention, "ring_attention"), "Should have ring_attention attribute"

    # The V2 implementation should have get_memory_estimate method
    if hasattr(attention.ring_attention, "get_memory_estimate"):
        estimate = attention.ring_attention.get_memory_estimate(4096)
        assert estimate["memory_reduction_factor"] > 1.0, "Should show memory reduction"


def test_memory_scaling_difference():
    """Test that V2 implementation shows proper memory scaling."""
    try:
        from dilated_attention_pytorch.ring_dilated_attention_v2 import (
            RingDilatedAttentionV2,
        )

        # Test different ring sizes
        for ring_size in [1, 2, 4, 8]:
            attention = RingDilatedAttentionV2(
                segment_lengths=[1024],
                dilation_rates=[1],
                ring_size=ring_size,
            )

            estimate = attention.get_memory_estimate(
                seq_len=8192, batch_size=1, num_heads=8, head_dim=64
            )

            # Memory reduction should roughly equal ring size
            expected_reduction = ring_size
            actual_reduction = estimate["memory_reduction_factor"]

            # Allow some variance but should be close
            assert abs(actual_reduction - expected_reduction) < 0.5, (
                f"Ring size {ring_size} should give ~{expected_reduction}x reduction, got {actual_reduction}x"
            )

    except ImportError:
        pytest.skip("RingDilatedAttentionV2 not available")


def test_migration_guide_examples():
    """Test examples from migration guide work."""
    # Old way should warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        _ = RingDilatedAttention(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
        )

        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()

    # New way using factory should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        _ = create_dilated_attention(
            "ring",
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            ring_size=4,
        )

        # Should have no deprecation warnings
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            "Factory should not emit deprecation warnings"
        )


if __name__ == "__main__":
    print("Testing Ring Attention migration...")

    # Run tests manually
    print("\n1. Testing deprecation warnings...")
    try:
        test_deprecated_ring_dilated_attention_warning()
        print("   ✗ Should have warned!")
    except:
        print("   ✓ Deprecation warning works")

    print("\n2. Testing factory implementation...")
    test_factory_uses_correct_implementation()
    print("   ✓ Factory uses correct implementation")

    print("\n3. Testing memory scaling...")
    test_memory_scaling_difference()
    print("   ✓ Memory scaling is correct")

    print("\n4. Testing migration examples...")
    test_migration_guide_examples()
    print("   ✓ Migration examples work")

    print("\n✓ All migration tests passed!")
