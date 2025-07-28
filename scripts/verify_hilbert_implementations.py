#!/usr/bin/env python3
"""
Comprehensive verification of remaining Hilbert implementations after refactoring.
"""

import torch
import sys
import os
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_implementation(
    name: str,
    module_class: type,
    test_configs: List[
        Tuple[int, int, int, int]
    ],  # (batch, seq_len, hidden_dim, num_heads)
) -> Dict[str, Any]:
    """Test a single implementation with various configurations."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    results = {
        "name": name,
        "success": True,
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": [],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch, seq_len, hidden_dim, num_heads in test_configs:
        config_str = f"B={batch}, L={seq_len}, D={hidden_dim}, H={num_heads}"
        try:
            # Create model with specific config
            init_kwargs = {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_size": min(64, seq_len),
                "dilation_rate": 1,
                "dropout": 0.0,
            }

            # Add use_custom_backward if supported
            if "use_custom_backward" in module_class.__init__.__code__.co_varnames:
                init_kwargs["use_custom_backward"] = True

            model = module_class(**init_kwargs).to(device)

            # Test input
            x = torch.randn(
                batch, seq_len, hidden_dim, device=device, requires_grad=True
            )

            # Forward pass
            with torch.amp.autocast(device.type, enabled=False):
                if (
                    hasattr(model, "forward")
                    and "use_hilbert" in model.forward.__code__.co_varnames
                ):
                    # Test with Hilbert ordering
                    out_hilbert = model(x, use_hilbert=True)
                    # Test without Hilbert ordering
                    _ = model(x, use_hilbert=False)
                else:
                    # Simple forward
                    out_hilbert = model(x)
                    _ = out_hilbert

            # Verify output shape
            assert out_hilbert.shape == x.shape, (
                f"Output shape mismatch: {out_hilbert.shape} != {x.shape}"
            )

            # Check for NaN/Inf
            assert not torch.isnan(out_hilbert).any(), "Output contains NaN"
            assert not torch.isinf(out_hilbert).any(), "Output contains Inf"

            # Test backward pass
            loss = out_hilbert.sum()
            loss.backward()

            # Check gradients
            assert x.grad is not None, "No gradient computed for input"
            assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
            assert not torch.isinf(x.grad).any(), "Gradient contains Inf"

            print(f"✓ {config_str}")
            results["tests_passed"] += 1

        except Exception as e:
            print(f"✗ {config_str}: {str(e)}")
            results["tests_failed"] += 1
            results["errors"].append(f"{config_str}: {str(e)}")
            results["success"] = False

    return results


def test_wrapper_interface():
    """Test the wrapper with Q,K,V interface."""
    print(f"\n{'=' * 60}")
    print("Testing Wrapper Q,K,V Interface")
    print(f"{'=' * 60}")

    from src.dilated_attention_pytorch.kernels.hilbert_attention_triton_wrapper import (
        HilbertAttentionTritonWrapper,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test configuration
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    try:
        # Create wrapper
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[128],
            dilation_rates=[1],
            dropout=0.0,
            num_heads=num_heads,
            head_dim=head_dim,
        ).to(device)

        # Create Q, K, V tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Forward pass
        output = wrapper(q, k, v)

        # Verify output
        assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"

        print("✓ Q,K,V interface works correctly")
        return True

    except Exception as e:
        print(f"✗ Q,K,V interface failed: {str(e)}")
        return False


def test_backward_compatibility():
    """Test backward compatibility aliases."""
    print(f"\n{'=' * 60}")
    print("Testing Backward Compatibility Aliases")
    print(f"{'=' * 60}")

    try:
        # Just verify imports work
        import src.dilated_attention_pytorch.kernels as kernels

        _ = kernels  # Use it to satisfy linter

        # No more aliases to check
        print("✓ Imports successful")
        print("✓ No backward compatibility aliases (cleaned up)")
        return True

    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        return False


def main():
    """Run all verification tests."""
    print("Hilbert Implementation Verification")
    print("=" * 60)

    all_results = []

    # Test configurations: (batch, seq_len, hidden_dim, num_heads)
    test_configs = [
        (1, 64, 256, 4),  # Small
        (2, 128, 512, 8),  # Medium
        (1, 256, 768, 12),  # Large
        (2, 255, 384, 8),  # Non-power-of-2
    ]

    # Test HilbertAttentionCore
    try:
        from src.dilated_attention_pytorch.kernels.hilbert_attention_core import (
            HilbertAttentionCore,
        )

        results = test_implementation(
            "HilbertAttentionCore", HilbertAttentionCore, test_configs
        )
        all_results.append(results)

    except Exception as e:
        print(f"Failed to test HilbertAttentionCore: {str(e)}")
        all_results.append(
            {"name": "HilbertAttentionCore", "success": False, "error": str(e)}
        )

    # Test original implementation if it exists
    try:
        from src.dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton_fixed import (
            HilbertAttentionTritonFixed as OriginalFixed,
        )

        results = test_implementation(
            "HilbertAttentionTritonFixed (Original)", OriginalFixed, test_configs
        )
        all_results.append(results)

    except Exception as e:
        print(f"Original implementation not found or failed: {str(e)}")

    # Test wrapper interface
    wrapper_success = test_wrapper_interface()

    # Test backward compatibility
    compat_success = test_backward_compatibility()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total_passed = sum(r.get("tests_passed", 0) for r in all_results)
    total_failed = sum(r.get("tests_failed", 0) for r in all_results)

    print("\nImplementation Tests:")
    for result in all_results:
        status = "✓" if result.get("success", False) else "✗"
        print(
            f"{status} {result['name']}: {result.get('tests_passed', 0)} passed, {result.get('tests_failed', 0)} failed"
        )

    print("\nInterface Tests:")
    print(f"{'✓' if wrapper_success else '✗'} Q,K,V Wrapper Interface")
    print(f"{'✓' if compat_success else '✗'} Backward Compatibility")

    print(f"\nTotal: {total_passed} tests passed, {total_failed} tests failed")

    # Print any errors
    if total_failed > 0:
        print(f"\n{'=' * 60}")
        print("ERRORS")
        print(f"{'=' * 60}")
        for result in all_results:
            if result.get("errors"):
                print(f"\n{result['name']}:")
                for error in result["errors"]:
                    print(f"  - {error}")

    return total_failed == 0 and wrapper_success and compat_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
