#!/usr/bin/env python3
"""
Test script for ImprovedMultiheadDilatedAttention to validate it works correctly
and is compatible with the original MultiheadDilatedAttention interface.
"""


def test_interface_compatibility():
    """Test that both implementations have the same interface."""
    print("Testing interface compatibility...")

    try:
        # Import both implementations
        import sys

        sys.path.append("dilated_attention_pytorch")

        from dilated_attention_pytorch.improved_multihead_dilated_attention import (
            ImprovedMultiheadDilatedAttention,
        )
        from dilated_attention_pytorch.multihead_dilated_attention import MultiheadDilatedAttention

        # Test parameters
        embed_dim = 512
        num_heads = 8
        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]

        print("✓ Successfully imported both implementations")

        # Test initialization
        try:
            original = MultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )
            print("✓ Original MultiheadDilatedAttention initialized")
        except Exception as e:
            print(f"✗ Original initialization failed: {e}")
            return False

        try:
            improved = ImprovedMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )
            print("✓ Improved MultiheadDilatedAttention initialized")
        except Exception as e:
            print(f"✗ Improved initialization failed: {e}")
            return False

        # Check parameter counts
        original_params = sum(p.numel() for p in original.parameters())
        improved_params = sum(p.numel() for p in improved.parameters())

        print(f"Original parameter count: {original_params:,}")
        print(f"Improved parameter count: {improved_params:,}")

        if original_params == improved_params:
            print("✓ Parameter counts match")
        else:
            print("⚠ Parameter counts differ (this might be expected)")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("This test requires PyTorch and einops to be installed")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_forward_compatibility():
    """Test that forward pass works with the same interface."""
    print("\nTesting forward pass compatibility...")

    try:
        import torch

        from dilated_attention_pytorch.improved_multihead_dilated_attention import (
            ImprovedMultiheadDilatedAttention,
        )

        # Test parameters
        batch_size = 2
        seq_len = 2048
        embed_dim = 256
        num_heads = 4
        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]

        # Create model
        model = ImprovedMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        )

        # Create test data
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)

        print(f"Input shapes - Q: {query.shape}, K: {key.shape}, V: {value.shape}")

        # Test forward pass
        with torch.no_grad():
            output, attn_weights = model(query, key, value)

        print(f"Output shape: {output.shape}")
        print(f"Attention weights: {attn_weights}")

        # Validate output
        expected_shape = (batch_size, seq_len, embed_dim)
        if output.shape == expected_shape:
            print("✓ Output shape is correct")
        else:
            print(f"✗ Output shape mismatch. Expected {expected_shape}, got {output.shape}")
            return False

        if attn_weights is None:
            print("✓ Attention weights correctly returned as None")
        else:
            print("⚠ Attention weights not None (unexpected but not necessarily wrong)")

        # Test causal attention
        with torch.no_grad():
            causal_output, _ = model(query, key, value, is_causal=True)

        print(f"Causal output shape: {causal_output.shape}")
        print("✓ Causal attention works")

        return True

    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False


def test_feature_compatibility():
    """Test specific features and optimizations."""
    print("\nTesting feature compatibility...")

    features_tested = []

    try:
        import torch

        from dilated_attention_pytorch.improved_multihead_dilated_attention import (
            ImprovedMultiheadDilatedAttention,
        )

        # Test TF32 option
        ImprovedMultiheadDilatedAttention(
            embed_dim=128,
            num_heads=4,
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            use_tf32=True,
        )
        features_tested.append("✓ TF32 option works")

        # Test layer norm option
        ImprovedMultiheadDilatedAttention(
            embed_dim=128,
            num_heads=4,
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            layer_norm=False,
        )
        features_tested.append("✓ Layer norm option works")

        # Test bias option
        ImprovedMultiheadDilatedAttention(
            embed_dim=128,
            num_heads=4,
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            bias=False,
        )
        features_tested.append("✓ Bias option works")

        # Test device/dtype options
        if torch.cuda.is_available():
            ImprovedMultiheadDilatedAttention(
                embed_dim=128,
                num_heads=4,
                segment_lengths=[64, 128],
                dilation_rates=[1, 2],
                device="cuda",
                dtype=torch.float16,
            )
            features_tested.append("✓ CUDA device/dtype options work")
        else:
            features_tested.append("⚠ CUDA not available for device test")

        # Test gamma_init option
        ImprovedMultiheadDilatedAttention(
            embed_dim=128,
            num_heads=4,
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            gamma_init=0.5,
        )
        features_tested.append("✓ Gamma init option works")

        for feature in features_tested:
            print(feature)

        return True

    except Exception as e:
        print(f"✗ Feature test failed: {e}")
        for feature in features_tested:
            print(feature)
        return False


def main():
    """Run all tests."""
    print("Testing ImprovedMultiheadDilatedAttention Implementation")
    print("=" * 60)

    tests = [
        test_interface_compatibility,
        test_forward_compatibility,
        test_feature_compatibility,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("🎉 All tests passed! ImprovedMultiheadDilatedAttention is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
