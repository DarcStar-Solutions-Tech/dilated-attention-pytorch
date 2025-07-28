#!/usr/bin/env python3
"""
Basic test for UnifiedHilbertAttention to verify initialization issues are fixed.
"""

import torch
import sys

# Add project to path
sys.path.insert(
    0, "/home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch/src"
)

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    create_hilbert_attention,
)


def test_basic_initialization():
    """Test basic initialization of UnifiedHilbertAttention."""
    print("Testing basic initialization...")

    # Test with default parameters
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
    )
    print("✓ Default initialization successful")

    # Test with memory optimization
    _ = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        memory_mode="aggressive",
    )
    print("✓ Memory mode initialization successful")

    # Test with sparse optimization
    _ = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        dilation_rate=4,
        sparse_mode="direct",
    )
    print("✓ Sparse mode initialization successful")

    # Test with combined optimizations
    _ = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=2,
        memory_mode="optimized",
        sparse_mode="selective",
        access_mode="strided",
        backend="pytorch",
    )
    print("✓ Combined optimization initialization successful")


def test_forward_pass():
    """Test forward pass through UnifiedHilbertAttention."""
    print("\nTesting forward pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=512,
        num_heads=8,
        segment_size=128,
        dilation_rate=2,
        sparse_mode="direct",
    ).to(device)

    # Test input
    batch_size = 2
    seq_len = 256
    hidden_dim = 512
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Forward pass
    with torch.no_grad():
        out = module(x)

    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} != {x.shape}"
    print("✓ Forward pass successful")

    # Test with gradient
    x.requires_grad = True
    out = module(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient computed for input"
    print("✓ Backward pass successful")


def test_factory_function():
    """Test the factory function for creating implementations."""
    print("\nTesting factory function...")

    implementations = ["standard", "memory_optimized", "sparse", "strided", "simple"]

    for impl in implementations:
        module = create_hilbert_attention(
            impl,
            hidden_dim=768,
            num_heads=12,
        )
        print(f"✓ Created '{impl}' implementation")

        # Test it works
        x = torch.randn(1, 128, 768)
        out = module(x)
        assert out.shape == x.shape


def test_configuration():
    """Test configuration methods."""
    print("\nTesting configuration...")

    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=256,
        dilation_rate=4,
        dropout=0.1,
        memory_mode="optimized",
        sparse_mode="direct",
        access_mode="strided",
        backend="pytorch",
    )

    config = module.get_config()
    assert config["hidden_dim"] == 768
    assert config["num_heads"] == 12
    assert config["segment_size"] == 256
    assert config["dilation_rate"] == 4
    assert config["dropout"] == 0.1
    assert config["memory_mode"] == "optimized"
    assert config["sparse_mode"] == "direct"
    assert config["access_mode"] == "strided"
    assert config["backend"] == "pytorch"

    print("✓ Configuration retrieval successful")


def test_memory_estimation():
    """Test memory usage estimation."""
    print("\nTesting memory estimation...")

    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,
        sparse_mode="direct",
    )

    mem_info = module.estimate_memory_usage(seq_len=2048, batch_size=4)

    assert "parameters_mb" in mem_info
    assert "cache_mb" in mem_info
    assert mem_info["parameters_mb"] > 0

    print(f"✓ Memory estimation successful: {mem_info}")


if __name__ == "__main__":
    print("Testing UnifiedHilbertAttention implementation...")
    print("=" * 50)

    test_basic_initialization()
    test_forward_pass()
    test_factory_function()
    test_configuration()
    test_memory_estimation()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
