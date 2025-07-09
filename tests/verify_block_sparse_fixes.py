#!/usr/bin/env python3
"""
Verify that block-sparse fixes work correctly.
Tests factory patterns, API consistency, and basic functionality.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    get_block_sparse_preset,
    create_hierarchical_block_sparse,
    create_adaptive_block_sparse,
    create_multihead_block_sparse,
)
from dilated_attention_pytorch import SparsePatternConfig


def test_factory_creation():
    """Test that all factory methods work."""
    print("Testing factory creation...")

    tests = [
        # Basic factory
        ("Base", lambda: create_block_sparse_attention("base")),
        ("Hierarchical", lambda: create_block_sparse_attention("hierarchical")),
        ("Adaptive", lambda: create_block_sparse_attention("adaptive")),
        (
            "Multihead",
            lambda: create_block_sparse_attention(
                "multihead", embed_dim=768, num_heads=12
            ),
        ),
        # Auto selection
        ("Auto", lambda: create_block_sparse_attention("auto")),
        # Presets
        ("Preset Local", lambda: get_block_sparse_preset("local")),
        ("Preset Hierarchical", lambda: get_block_sparse_preset("hierarchical_long")),
        ("Preset Ultra Sparse", lambda: get_block_sparse_preset("ultra_sparse")),
        # Convenience functions
        ("Hierarchical Preset", lambda: create_hierarchical_block_sparse("standard")),
        (
            "Adaptive Convenience",
            lambda: create_adaptive_block_sparse(base_sparsity=0.95),
        ),
        ("Multihead Convenience", lambda: create_multihead_block_sparse(768, 12)),
    ]

    results = []
    for name, create_fn in tests:
        try:
            _ = create_fn()
            results.append((name, "✓ Success", None))
            print(f"  {name}: ✓")
        except Exception as e:
            results.append((name, "✗ Failed", str(e)))
            print(f"  {name}: ✗ {str(e)}")

    return results


def test_api_consistency():
    """Test that all implementations have consistent APIs."""
    print("\nTesting API consistency...")

    # Create test inputs
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if torch.cuda.is_available():
        q, k, v = q.cuda(), k.cuda(), v.cuda()

    implementations = [
        ("Base", create_block_sparse_attention("base")),
        ("Hierarchical", create_block_sparse_attention("hierarchical")),
        ("Adaptive", create_block_sparse_attention("adaptive")),
    ]

    results = []
    for name, model in implementations:
        try:
            if torch.cuda.is_available():
                model = model.cuda()

            # Test forward pass
            output = model(q, k, v)

            # Check shape
            if output.shape != q.shape:
                raise ValueError(f"Output shape mismatch: {output.shape} != {q.shape}")

            # Test with return_attention_weights
            output2, weights = model(q, k, v, return_attention_weights=True)

            results.append((name, "✓ Success", None))
            print(f"  {name}: ✓")

        except Exception as e:
            results.append((name, "✗ Failed", str(e)))
            print(f"  {name}: ✗ {str(e)}")

    return results


def test_multihead_device_fix():
    """Test that multihead implementation handles devices correctly."""
    print("\nTesting multihead device fix...")

    if not torch.cuda.is_available():
        print("  Skipped (no CUDA)")
        return [("Multihead Device", "Skipped", "No CUDA available")]

    try:
        # Create on CPU first
        model = create_multihead_block_sparse(
            embed_dim=768, num_heads=12, sparsity_ratio=0.1
        )

        # Move to CUDA
        model = model.cuda()

        # Create inputs on CUDA
        batch_size = 2
        seq_len = 512
        embed_dim = 768

        x = torch.randn(batch_size, seq_len, embed_dim, device="cuda")

        # Forward pass
        output = model(x, x, x)

        # Check output is on correct device
        if output.device != x.device:
            raise ValueError(f"Output on wrong device: {output.device} != {x.device}")

        print("  Multihead Device: ✓")
        return [("Multihead Device", "✓ Success", None)]

    except Exception as e:
        print(f"  Multihead Device: ✗ {str(e)}")
        return [("Multihead Device", "✗ Failed", str(e))]


def test_hierarchical_stats_fix():
    """Test that hierarchical get_pattern_stats works without seq_len."""
    print("\nTesting hierarchical stats fix...")

    try:
        model = create_block_sparse_attention("hierarchical")

        # Test without seq_len (should use default)
        stats1 = model.get_pattern_stats()

        # Test with explicit seq_len
        _ = model.get_pattern_stats(seq_len=4096)

        # Check stats have expected keys
        expected_keys = ["total_blocks", "active_blocks", "sparsity", "levels"]
        for key in expected_keys:
            if key not in stats1:
                raise ValueError(f"Missing key in stats: {key}")

        print("  Hierarchical Stats: ✓")
        return [("Hierarchical Stats", "✓ Success", None)]

    except Exception as e:
        print(f"  Hierarchical Stats: ✗ {str(e)}")
        return [("Hierarchical Stats", "✗ Failed", str(e))]


def test_adaptive_lazy_init():
    """Test that adaptive implementation works with lazy initialization."""
    print("\nTesting adaptive lazy initialization...")

    try:
        # Create without specifying dimensions
        model = create_block_sparse_attention("adaptive")

        # Create inputs (dimensions inferred from here)
        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        if torch.cuda.is_available():
            model = model.cuda()
            q, k, v = q.cuda(), k.cuda(), v.cuda()

        # First forward pass should initialize
        output = model(q, k, v)

        # Check output shape
        if output.shape != q.shape:
            raise ValueError(f"Output shape mismatch: {output.shape} != {q.shape}")

        # Second forward pass should work normally
        _ = model(q, k, v)

        print("  Adaptive Lazy Init: ✓")
        return [("Adaptive Lazy Init", "✓ Success", None)]

    except Exception as e:
        print(f"  Adaptive Lazy Init: ✗ {str(e)}")
        return [("Adaptive Lazy Init", "✗ Failed", str(e))]


def test_sparse_config_override():
    """Test that sparse config parameters can be overridden."""
    print("\nTesting sparse config override...")

    try:
        # Test with sparsity_ratio override
        model1 = create_block_sparse_attention("base", sparsity_ratio=0.01)
        if model1.sparse_config.sparsity_ratio != 0.01:
            raise ValueError("Sparsity ratio not set correctly")

        # Test with block_size override
        model2 = create_block_sparse_attention("base", block_size=128)
        if model2.sparse_config.block_size != 128:
            raise ValueError("Block size not set correctly")

        # Test with full sparse_config
        config = SparsePatternConfig(
            pattern_type="global_local",
            sparsity_ratio=0.05,
            block_size=256,
            global_tokens=128,
        )
        model3 = create_block_sparse_attention("base", sparse_config=config)
        if model3.sparse_config.pattern_type != "global_local":
            raise ValueError("Pattern type not set correctly")

        print("  Sparse Config Override: ✓")
        return [("Sparse Config Override", "✓ Success", None)]

    except Exception as e:
        print(f"  Sparse Config Override: ✗ {str(e)}")
        return [("Sparse Config Override", "✗ Failed", str(e))]


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Block-Sparse Fixes Verification")
    print("=" * 60)

    all_results = []

    # Run all tests
    all_results.extend(test_factory_creation())
    all_results.extend(test_api_consistency())
    all_results.extend(test_multihead_device_fix())
    all_results.extend(test_hierarchical_stats_fix())
    all_results.extend(test_adaptive_lazy_init())
    all_results.extend(test_sparse_config_override())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status, _ in all_results if "Success" in status)
    total = len(all_results)

    print(f"\nTotal: {passed}/{total} tests passed")

    # Show failures
    failures = [
        (name, error) for name, status, error in all_results if "Failed" in status
    ]
    if failures:
        print("\nFailures:")
        for name, error in failures:
            print(f"  - {name}: {error}")

    if passed == total:
        print("\n✅ All fixes verified successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
