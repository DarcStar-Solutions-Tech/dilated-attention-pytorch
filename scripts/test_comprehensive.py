#!/usr/bin/env python3
"""Comprehensive test script to verify all components."""

import torch
import sys
from typing import Dict, Tuple


def test_component(name: str, test_func) -> Tuple[bool, str]:
    """Test a component and return status."""
    try:
        test_func()
        return True, "PASSED"
    except Exception as e:
        return False, str(e)


def run_tests() -> Dict[str, Tuple[bool, str]]:
    """Run all component tests."""
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: Basic imports
    def test_imports():
        pass

    results["Basic imports"] = test_component("Basic imports", test_imports)

    # Test 2: DilatedAttention
    def test_dilated_attention():
        from dilated_attention_pytorch import DilatedAttention

        model = DilatedAttention(segment_lengths=[256, 512], dilation_rates=[1, 2]).to(
            device
        )
        x = torch.randn(2, 1024, 8, 64, device=device)
        out = model(x, x, x)
        assert out.shape == x.shape

    results["DilatedAttention"] = test_component(
        "DilatedAttention", test_dilated_attention
    )

    # Test 3: MultiheadDilatedAttention
    def test_multihead_dilated_attention():
        from dilated_attention_pytorch import MultiheadDilatedAttention

        mha = MultiheadDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
        ).to(device)
        x = torch.randn(2, 1024, 512, device=device)
        out = mha(x, x, x)
        assert out.shape == x.shape

    results["MultiheadDilatedAttention"] = test_component(
        "MultiheadDilatedAttention", test_multihead_dilated_attention
    )

    # Test 4: ImprovedDilatedAttention
    def test_improved_dilated_attention():
        from dilated_attention_pytorch import ImprovedDilatedAttention

        model = ImprovedDilatedAttention(
            segment_lengths=[256, 512], dilation_rates=[1, 2]
        ).to(device)
        x = torch.randn(2, 1024, 8, 64, device=device)
        out = model(x, x, x)
        assert out.shape == x.shape

    results["ImprovedDilatedAttention"] = test_component(
        "ImprovedDilatedAttention", test_improved_dilated_attention
    )

    # Test 5: ImprovedMultiheadDilatedAttention (returns tuple)
    def test_improved_multihead_dilated_attention():
        from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention

        mha = ImprovedMultiheadDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
        ).to(device)
        x = torch.randn(2, 1024, 512, device=device)
        out = mha(x, x, x)
        # Handle both tuple and tensor returns
        if isinstance(out, tuple):
            assert out[0].shape == x.shape
        else:
            assert out.shape == x.shape

    results["ImprovedMultiheadDilatedAttention"] = test_component(
        "ImprovedMultiheadDilatedAttention", test_improved_multihead_dilated_attention
    )

    # Test 6: Factory functions
    def test_factory_functions():
        from dilated_attention_pytorch import create_multihead_dilated_attention

        attn = create_multihead_dilated_attention(
            "improved",
            embed_dim=512,
            num_heads=8,
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
        ).to(device)
        x = torch.randn(2, 1024, 512, device=device)
        out = attn(x, x, x)
        # Handle both tuple and tensor returns
        if isinstance(out, tuple):
            assert out[0].shape == x.shape
        else:
            assert out.shape == x.shape

    results["Factory functions"] = test_component(
        "Factory functions", test_factory_functions
    )

    # Test 7: LongNet
    def test_longnet():
        from dilated_attention_pytorch import LongNet

        # LongNet is the base transformer (not LM)
        model = LongNet(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
        ).to(device)

        # Input is embeddings, not token ids
        x = torch.randn(2, 1024, 512, device=device)
        out = model(x)
        assert out.shape == x.shape

    results["LongNet"] = test_component("LongNet", test_longnet)

    # Test 8: Edge cases
    def test_edge_cases():
        from dilated_attention_pytorch import DilatedAttention

        # Test with minimum sequence length
        model = DilatedAttention(segment_lengths=[64], dilation_rates=[1]).to(device)
        x = torch.randn(1, 64, 4, 32, device=device)
        out = model(x, x, x)
        assert out.shape == x.shape

        # Test with causal mask
        out_causal = model(x, x, x, is_causal=True)
        assert out_causal.shape == x.shape

    results["Edge cases"] = test_component("Edge cases", test_edge_cases)

    # Test 9: Error handling
    def test_error_handling():
        from dilated_attention_pytorch import DilatedAttention

        # Test mismatched lengths
        try:
            model = DilatedAttention(
                segment_lengths=[256, 512],
                dilation_rates=[1],  # Should match segment_lengths
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

        # Test invalid sequence length
        model = DilatedAttention(segment_lengths=[256], dilation_rates=[1]).to(device)

        try:
            x = torch.randn(1, 100, 4, 32, device=device)  # Not divisible by 256
            _ = model(x, x, x)
            raise AssertionError("Should have raised error for invalid sequence length")
        except (ValueError, RuntimeError):
            pass

    results["Error handling"] = test_component("Error handling", test_error_handling)

    # Test 10: Backward compatibility
    def test_backward_compatibility():
        from dilated_attention_pytorch import RingDilatedAttention

        # Should be an alias for RingDilatedAttentionHybrid
        assert RingDilatedAttention.__name__ == "RingDilatedAttentionHybrid"

    results["Backward compatibility"] = test_component(
        "Backward compatibility", test_backward_compatibility
    )

    return results


def main():
    """Run all tests and print results."""
    print("Running comprehensive test suite...")
    print("=" * 60)

    results = run_tests()

    # Print results
    passed = 0
    failed = 0

    for test_name, (status, message) in results.items():
        if status:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"✗ {test_name}: FAILED - {message}")
            failed += 1

    print("=" * 60)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed / len(results) * 100:.1f}%")

    # Return exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
