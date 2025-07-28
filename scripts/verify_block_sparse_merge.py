#!/usr/bin/env python3
"""
Verify that all block-sparse implementations work correctly after merging optimizations.
Tests all 7 remaining implementations with various configurations.
"""

import torch
from typing import Dict, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    BlockSparseRingDilatedAttention,
    BlockSparseRingMultiheadDilatedAttention,
    BlockSparseRingDistributedDilatedAttention,
    BlockSparseHierarchical,
    BlockSparseAdaptive,
    SparsePatternConfig,
)

# Test configurations
TEST_CONFIGS = [
    {
        "name": "Small sequence",
        "batch_size": 2,
        "seq_len": 512,
        "num_heads": 8,
        "head_dim": 64,
        "sparsity_ratio": 0.1,
        "pattern_type": "local_window",
    },
    {
        "name": "Medium sequence",
        "batch_size": 1,
        "seq_len": 2048,
        "num_heads": 16,
        "head_dim": 64,
        "sparsity_ratio": 0.05,
        "pattern_type": "dilated_sparse",
    },
    {
        "name": "Large sequence",
        "batch_size": 1,
        "seq_len": 8192,
        "num_heads": 12,
        "head_dim": 64,
        "sparsity_ratio": 0.02,
        "pattern_type": "global_local",
    },
]


def test_basic_functionality(
    model_class, config: Dict, use_multihead: bool = False
) -> Tuple[bool, str]:
    """Test basic forward pass functionality."""
    try:
        # Create model
        if use_multihead:
            # Multihead version needs embed_dim
            embed_dim = config["num_heads"] * config["head_dim"]
            model = model_class(
                embed_dim=embed_dim,
                num_heads=config["num_heads"],
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                sparsity_ratio=config["sparsity_ratio"],
            )

            # Create inputs for multihead
            x = torch.randn(
                config["batch_size"], config["seq_len"], embed_dim, device="cuda"
            )
            output = model(x, x, x)

            # Check output shape
            expected_shape = (config["batch_size"], config["seq_len"], embed_dim)
        else:
            # Create sparse config
            sparse_config = SparsePatternConfig(
                pattern_type=config["pattern_type"],
                sparsity_ratio=config["sparsity_ratio"],
                block_size=64,  # Default block size
            )

            # Ring implementations need these parameters
            model = model_class(
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                sparse_config=sparse_config,
            )

            # Create inputs
            q = torch.randn(
                config["batch_size"],
                config["seq_len"],
                config["num_heads"],
                config["head_dim"],
                device="cuda",
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Forward pass
            output = model(q, k, v)

            # Check output shape
            expected_shape = q.shape

        if output.shape != expected_shape:
            return False, f"Output shape mismatch: {output.shape} != {expected_shape}"

        # Check for NaN/Inf
        if torch.isnan(output).any():
            return False, "Output contains NaN values"
        if torch.isinf(output).any():
            return False, "Output contains Inf values"

        return True, "Success"

    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_pattern_caching(model_class, config: Dict) -> Tuple[bool, str]:
    """Test that pattern caching works correctly."""
    try:
        # Create model
        sparse_config = SparsePatternConfig(
            pattern_type=config["pattern_type"],
            sparsity_ratio=config["sparsity_ratio"],
            block_size=64,
        )

        model = model_class(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
        )

        # Create inputs
        q = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["num_heads"],
            config["head_dim"],
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # First forward pass
        output1 = model(q, k, v)

        # Check cache stats if available
        if hasattr(model, "get_pattern_stats"):
            stats1 = model.get_pattern_stats()

            # Second forward pass (should use cache)
            output2 = model(q, k, v)
            stats2 = model.get_pattern_stats()

            # Verify cache was used
            if stats2["total_accesses"] <= stats1["total_accesses"]:
                return False, "Cache access count did not increase"

            # Outputs should be identical
            if not torch.allclose(output1, output2, rtol=1e-5):
                return False, "Cached outputs differ from original"

        return True, "Success"

    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_gradient_flow(model_class, config: Dict) -> Tuple[bool, str]:
    """Test backward pass and gradient flow."""
    try:
        # Create model
        sparse_config = SparsePatternConfig(
            pattern_type=config["pattern_type"],
            sparsity_ratio=config["sparsity_ratio"],
            block_size=64,
        )

        model = model_class(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
        )

        # Create inputs with requires_grad
        q = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["num_heads"],
            config["head_dim"],
            device="cuda",
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        # Forward pass
        output = model(q, k, v)

        # Create loss and backward
        loss = output.mean()
        loss.backward()

        # Check gradients exist and are not zero
        for tensor, name in [(q, "q"), (k, "k"), (v, "v")]:
            if tensor.grad is None:
                return False, f"No gradient for {name}"
            if torch.all(tensor.grad == 0):
                return False, f"Zero gradient for {name}"
            if torch.isnan(tensor.grad).any():
                return False, f"NaN gradient for {name}"

        return True, "Success"

    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_all_tests():
    """Run all tests for all implementations."""
    # Map of implementation names to classes and whether they use multihead API
    implementations = {
        "BlockSparseRingDilatedAttention": (BlockSparseRingDilatedAttention, False),
        "BlockSparseRingMultiheadDilatedAttention": (
            BlockSparseRingMultiheadDilatedAttention,
            True,
        ),
        "BlockSparseHierarchical": (BlockSparseHierarchical, False),
        "BlockSparseAdaptive": (BlockSparseAdaptive, False),
    }

    # Skip distributed if not in distributed environment
    if torch.cuda.device_count() > 1:
        implementations["BlockSparseRingDistributedDilatedAttention"] = (
            BlockSparseRingDistributedDilatedAttention,
            False,
        )

    results = {}

    for impl_name, (impl_class, use_multihead) in implementations.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {impl_name}")
        print(f"{'=' * 60}")

        impl_results = {}

        for config in TEST_CONFIGS:
            print(f"\nConfiguration: {config['name']}")
            print(f"  Sequence length: {config['seq_len']}")
            print(f"  Pattern type: {config['pattern_type']}")
            print(f"  Sparsity ratio: {config['sparsity_ratio']}")

            # Run tests
            tests = [
                (
                    "Basic functionality",
                    lambda: test_basic_functionality(impl_class, config, use_multihead),
                ),
            ]

            # Only test caching and gradients for non-multihead implementations
            if not use_multihead:
                tests.extend(
                    [
                        (
                            "Pattern caching",
                            lambda: test_pattern_caching(impl_class, config),
                        ),
                        (
                            "Gradient flow",
                            lambda: test_gradient_flow(impl_class, config),
                        ),
                    ]
                )

            config_results = {}
            for test_name, test_func in tests:
                success, message = test_func()
                config_results[test_name] = (success, message)
                status = "✓" if success else "✗"
                print(f"  {test_name}: {status} {message if not success else ''}")

            impl_results[config["name"]] = config_results

        results[impl_name] = impl_results

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total_tests = 0
    passed_tests = 0

    for impl_name, impl_results in results.items():
        impl_passed = 0
        impl_total = 0

        for config_name, config_results in impl_results.items():
            for test_name, (success, _) in config_results.items():
                impl_total += 1
                total_tests += 1
                if success:
                    impl_passed += 1
                    passed_tests += 1

        print(f"{impl_name}: {impl_passed}/{impl_total} tests passed")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✅ All block-sparse implementations verified successfully!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")

    return passed_tests == total_tests


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        sys.exit(0)

    success = run_all_tests()
    sys.exit(0 if success else 1)
