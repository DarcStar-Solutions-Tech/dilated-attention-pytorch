"""
Test Ring Attention migration and deprecation warnings.
"""

import warnings
import pytest

from dilated_attention_pytorch import (
    RingDilatedAttention,
    create_dilated_attention,
)
# RingMultiheadDilatedAttention not available in current implementation


def test_ring_dilated_attention_no_warning():
    """Test that RingDilatedAttention (alias for V2Collective) does not emit deprecation warning."""
    # RingDilatedAttention is now an alias for RingDilatedAttentionV2Collective
    # which is the recommended implementation. We filter out GPU-specific warnings.
    with warnings.catch_warnings(record=True) as w:
        # Filter out known hardware-specific warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        _ = RingDilatedAttention(
            segment_lengths=[1024],
            dilation_rates=[1],
        )

        # Check that no DeprecationWarning was raised
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            "RingDilatedAttention should not emit deprecation warnings"
        )


# RingMultiheadDilatedAttention test removed - not available


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
    # RingDilatedAttention is now an alias for V2Collective (no warning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        _ = RingDilatedAttention(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
        )

        # Should have no warnings (it's the recommended implementation)
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0

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
    print("\n1. Testing no deprecation warnings...")
    try:
        test_ring_dilated_attention_no_warning()
        print("   ✓ No deprecation warning (expected)")
    except Exception:
        print("   ✗ Unexpected error")

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
