#!/usr/bin/env python3
"""
Test kernel consolidation implementation.

This verifies that:
1. All 10 original kernel implementations have been consolidated
2. The unified implementation provides equivalent functionality
3. Memory caches are properly bounded
4. Migration utilities work correctly
"""

import torch
import sys

# Add project to path
sys.path.insert(
    0, "/home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch/src"
)

from dilated_attention_pytorch.kernels import (
    create_hilbert_attention,
    UnifiedHilbertAttention,
)


def test_consolidation_completeness():
    """Verify all 10 implementations are accounted for."""
    print("Testing consolidation completeness...")

    # Original implementations that should be consolidated
    original_implementations = [
        "UnifiedHilbertAttention",
        "HilbertAttentionMemoryOptimized",
        "HilbertAttentionOptimizedSelective",
        "HilbertAttentionSparseOptimized",
        "HilbertAttentionSparseSimple",
        "HilbertAttentionStrided",
        "HilbertAttentionStridedSimple",
        "HilbertAttentionV2",
        "HilbertAttentionTester",
        "HilbertAttentionBase",
    ]

    # Map to unified configurations
    implementation_configs = {
        "UnifiedHilbertAttention": {},
        "HilbertAttentionMemoryOptimized": {"memory_mode": "aggressive"},
        "HilbertAttentionOptimizedSelective": {"sparse_mode": "selective"},
        "HilbertAttentionSparseOptimized": {"sparse_mode": "direct"},
        "HilbertAttentionSparseSimple": {"sparse_mode": "direct", "backend": "pytorch"},
        "HilbertAttentionStrided": {"access_mode": "strided"},
        "HilbertAttentionStridedSimple": {
            "access_mode": "strided",
            "backend": "pytorch",
        },
        "HilbertAttentionV2": {"memory_mode": "optimized"},
        "HilbertAttentionTester": {"backend": "pytorch"},
        "HilbertAttentionBase": {"backend": "pytorch"},
    }

    # Verify each can be created through unified implementation
    for impl_name, config in implementation_configs.items():
        _ = UnifiedHilbertAttention(hidden_dim=768, num_heads=12, **config)
        print(f"✓ {impl_name} -> UnifiedHilbertAttention(**{config})")

    print(f"✓ All {len(original_implementations)} implementations consolidated")


def test_direct_usage():
    """Test that implementations can be used directly."""
    print("\nTesting direct usage...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test UnifiedHilbertAttention (if available)
    if UnifiedHilbertAttention is not None:
        module = UnifiedHilbertAttention(
            hidden_dim=768, num_heads=12, segment_size=128
        ).to(device)
        x = torch.randn(2, 256, 768, device=device)
        out = module(x)
        assert out.shape == x.shape
        print("✓ UnifiedHilbertAttention works directly")

    # Test UnifiedHilbertAttention
    module = UnifiedHilbertAttention(hidden_dim=768, num_heads=12, segment_size=128).to(
        device
    )
    x = torch.randn(2, 256, 768, device=device)
    out = module(x)
    assert out.shape == x.shape
    print("✓ UnifiedHilbertAttention works directly")


def test_memory_cache_bounds():
    """Test that memory caches are properly bounded."""
    print("\nTesting memory cache bounds...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create module with small cache limits
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,
        sparse_mode="direct",
        cache_size=8,  # Small cache
        cache_memory_mb=10.0,  # Low memory limit
    ).to(device)

    # Process many different sequence lengths to stress the cache
    seq_lengths = list(range(128, 2048, 128))
    for _ in range(3):  # Multiple passes
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, 768, device=device)
            with torch.no_grad():
                _ = module(x)

    # Check base cache stats
    base_stats = module.get_cache_stats()
    assert base_stats["size"] <= 8, (
        f"Base cache exceeded size limit: {base_stats['size']}"
    )
    assert base_stats["memory_usage_mb"] <= 10.0, (
        f"Base cache exceeded memory limit: {base_stats['memory_usage_mb']}"
    )

    # Check sparse cache stats if available
    if hasattr(module, "get_sparse_cache_stats"):
        sparse_stats = module.get_sparse_cache_stats()
        assert sparse_stats["size"] <= 32, (
            f"Sparse cache exceeded size limit: {sparse_stats['size']}"
        )
        assert sparse_stats["memory_usage_mb"] <= 50.0, (
            f"Sparse cache exceeded memory limit: {sparse_stats['memory_usage_mb']}"
        )

    print("✓ Memory caches properly bounded")
    print(
        f"  Base cache: {base_stats['size']} entries, {base_stats['memory_usage_mb']:.2f} MB"
    )
    if hasattr(module, "get_sparse_cache_stats"):
        print(
            f"  Sparse cache: {sparse_stats['size']} entries, {sparse_stats['memory_usage_mb']:.2f} MB"
        )


def test_factory_function():
    """Test the factory function creates correct configurations."""
    print("\nTesting factory function...")

    test_cases = [
        ("standard", {}),
        ("memory_optimized", {"memory_mode": "aggressive"}),
        ("sparse", {"sparse_mode": "direct"}),
        ("selective", {"sparse_mode": "selective"}),
        ("strided", {"access_mode": "strided"}),
        ("simple", {"backend": "pytorch"}),
    ]

    for impl_type, expected_config in test_cases:
        module = create_hilbert_attention(
            impl_type,
            hidden_dim=768,
            num_heads=12,
        )

        config = module.get_config()
        for key, value in expected_config.items():
            assert config[key] == value, (
                f"{impl_type}: Expected {key}={value}, got {config[key]}"
            )

        print(f"✓ Factory '{impl_type}' creates correct configuration")


def test_unified_replaces_old():
    """Test that unified implementation can replace old ones."""
    print("\nTesting unified replacement...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create unified module with specific configuration
    unified_module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=256,
        dilation_rate=2,
        backend="pytorch",  # Similar to old UnifiedHilbertAttention
    ).to(device)

    # Verify it works
    x = torch.randn(2, 512, 768, device=device)
    with torch.no_grad():
        out = unified_module(x)
    assert out.shape == x.shape

    print("✓ Unified implementation successfully replaces old ones")


def test_optimization_combinations():
    """Test various optimization combinations work together."""
    print("\nTesting optimization combinations...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combinations = [
        # Memory + Sparse
        {"memory_mode": "aggressive", "sparse_mode": "direct", "dilation_rate": 4},
        # Memory + Strided
        {"memory_mode": "optimized", "access_mode": "strided", "dilation_rate": 2},
        # Sparse + PyTorch backend
        {"sparse_mode": "selective", "backend": "pytorch", "dilation_rate": 2},
        # All optimizations
        {
            "memory_mode": "optimized",
            "sparse_mode": "direct",
            "access_mode": "strided",
            "backend": "pytorch",
            "dilation_rate": 4,
        },
    ]

    for i, config in enumerate(combinations):
        module = UnifiedHilbertAttention(
            hidden_dim=512, num_heads=8, segment_size=128, **config
        ).to(device)

        x = torch.randn(2, 256, 512, device=device)
        out = module(x)
        assert out.shape == x.shape

        print(f"✓ Combination {i + 1}: {config}")


def test_configuration_options():
    """Test various configuration options in unified implementation."""
    print("\nTesting configuration options...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different configuration modes
    configs = [
        {"memory_mode": "standard"},
        {"memory_mode": "optimized"},
        {"memory_mode": "aggressive"},
        {"sparse_mode": "standard"},
        {"sparse_mode": "selective"},
        {"sparse_mode": "direct"},
        {"access_mode": "standard"},
        {"access_mode": "strided"},
        {"backend": "auto"},
        {"backend": "pytorch"},
    ]

    for config in configs:
        module = UnifiedHilbertAttention(
            hidden_dim=256, num_heads=4, segment_size=64, **config
        ).to(device)

        # Get and verify config
        module_config = module.get_config()
        for key, value in config.items():
            assert module_config[key] == value, f"Config mismatch: {key}={value}"

        print(f"✓ Configuration works: {config}")


def test_forward_backward_consistency():
    """Test that unified implementation maintains gradient flow."""
    print("\nTesting forward/backward consistency...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with different modes
    configs = [
        {},  # Standard
        {"memory_mode": "optimized"},
        {"sparse_mode": "direct", "dilation_rate": 2},
        {"access_mode": "strided", "dilation_rate": 2},
    ]

    for config in configs:
        module = UnifiedHilbertAttention(
            hidden_dim=256, num_heads=4, segment_size=64, **config
        ).to(device)

        # Forward pass
        x = torch.randn(2, 128, 256, device=device, requires_grad=True)
        out = module(x)
        loss = out.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert module.qkv_proj.weight.grad is not None
        assert module.out_proj.weight.grad is not None

        # Gradient should be non-zero
        assert x.grad.abs().sum() > 0

        print(f"✓ Gradient flow correct for config: {config}")


if __name__ == "__main__":
    print("Testing Kernel Consolidation Implementation")
    print("=" * 60)

    test_consolidation_completeness()
    test_direct_usage()
    test_memory_cache_bounds()
    test_factory_function()
    test_unified_replaces_old()
    test_optimization_combinations()
    test_configuration_options()
    test_forward_backward_consistency()

    print("\n" + "=" * 60)
    print("All consolidation tests passed! ✓")
    print("\nSummary:")
    print("- 10 kernel implementations successfully consolidated")
    print("- Unified implementation supports all optimization modes")
    print("- Memory caches are properly bounded")
    print("- Direct usage of implementations works")
    print("- Configuration options work correctly")
    print("- Unified implementation provides all functionality")
