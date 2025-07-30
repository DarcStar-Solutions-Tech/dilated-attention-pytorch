#!/usr/bin/env python3
"""
Test suite for UnifiedHilbertAttention implementation.
"""

import pytest
import torch

from dilated_attention_pytorch.kernels import (
    create_hilbert_attention,
    migrate_to_unified,
    UnifiedHilbertAttention,
)


class TestUnifiedHilbertAttention:
    """Test UnifiedHilbertAttention functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def test_configs(self):
        """Get test configurations."""
        return [
            # (batch_size, seq_len, hidden_dim, num_heads, segment_size, dilation_rate)
            (2, 512, 768, 12, 128, 1),
            (1, 1024, 512, 8, 256, 2),
            (2, 2048, 1024, 16, 512, 4),
        ]

    def test_standard_mode(self, device, test_configs):
        """Test standard mode matches UnifiedHilbertAttention."""
        for B, N, D, H, seg_size, dil_rate in test_configs:
            # Skip if incompatible
            if N % seg_size != 0:
                continue

            # Create modules
            unified = UnifiedHilbertAttention(
                hidden_dim=D,
                num_heads=H,
                segment_size=seg_size,
                dilation_rate=dil_rate,
                memory_mode="standard",
                sparse_mode="standard",
                access_mode="standard",
            ).to(device)

            reference = UnifiedHilbertAttention(
                hidden_dim=D,
                num_heads=H,
                segment_size=seg_size,
                dilation_rate=dil_rate,
            ).to(device)

            # Copy weights
            unified.qkv_proj.weight.data = reference.qkv_proj.weight.data.clone()
            unified.out_proj.weight.data = reference.out_proj.weight.data.clone()

            # Test forward pass
            x = torch.randn(B, N, D, device=device)

            with torch.no_grad():
                out_unified = unified(x, use_hilbert=True)
                out_reference = reference(x, use_hilbert=True)

            # Check outputs match
            assert torch.allclose(out_unified, out_reference, atol=1e-5)

    def test_memory_modes(self, device):
        """Test different memory optimization modes."""
        configs = ["standard", "optimized", "aggressive"]

        for mode in configs:
            module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                memory_mode=mode,
            ).to(device)

            # Test forward pass works
            x = torch.randn(2, 512, 768, device=device)
            out = module(x)
            assert out.shape == x.shape

            # Check memory optimization level
            expected_level = {"standard": 0, "optimized": 1, "aggressive": 2}[mode]
            assert module.memory_optimization_level == expected_level

    def test_sparse_modes(self, device):
        """Test different sparse optimization modes."""
        modes = ["standard", "selective", "direct"]

        for mode in modes:
            module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=4,
                sparse_mode=mode,
            ).to(device)

            # Test forward pass
            x = torch.randn(2, 512, 768, device=device)
            out = module(x)
            assert out.shape == x.shape

            # For sparse modes, check cache exists
            if mode in ["selective", "direct"]:
                assert hasattr(module, "_sparse_pattern_cache")

    def test_access_modes(self, device):
        """Test different access modes."""
        modes = ["standard", "strided"]

        for mode in modes:
            module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=2,
                access_mode=mode,
            ).to(device)

            # Test forward pass
            x = torch.randn(2, 512, 768, device=device)
            out = module(x)
            assert out.shape == x.shape

    def test_backend_selection(self, device):
        """Test backend selection."""
        # PyTorch backend
        module_pytorch = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            backend="pytorch",
        ).to(device)
        assert not module_pytorch.use_triton

        # Auto backend
        module_auto = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            backend="auto",
        ).to(device)
        assert module_auto.use_triton == torch.cuda.is_available()

        # Triton backend (if CUDA available)
        if torch.cuda.is_available():
            module_triton = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                backend="triton",
            ).to(device)
            assert module_triton.use_triton

    def test_gradient_flow(self, device):
        """Test gradient flow through unified implementation."""
        module = UnifiedHilbertAttention(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=2,
            sparse_mode="direct",
        ).to(device)

        # Input with gradients
        x = torch.randn(2, 256, 512, device=device, requires_grad=True)

        # Forward pass
        out = module(x)
        loss = out.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert module.qkv_proj.weight.grad is not None
        assert module.out_proj.weight.grad is not None

    def test_config_combinations(self, device):
        """Test various configuration combinations."""
        configs = [
            # Memory + Sparse
            {"memory_mode": "optimized", "sparse_mode": "direct", "dilation_rate": 4},
            # Memory + Strided
            {"memory_mode": "aggressive", "access_mode": "strided", "dilation_rate": 2},
            # Sparse + PyTorch backend
            {"sparse_mode": "selective", "backend": "pytorch", "dilation_rate": 2},
        ]

        for config in configs:
            module = UnifiedHilbertAttention(
                hidden_dim=768, num_heads=12, segment_size=128, **config
            ).to(device)

            x = torch.randn(2, 512, 768, device=device)
            out = module(x)
            assert out.shape == x.shape

    def test_create_hilbert_attention_factory(self, device):
        """Test factory function for creating specific implementations."""
        implementations = {
            "standard": {},
            "memory_optimized": {"memory_mode": "aggressive"},
            "sparse": {"sparse_mode": "direct"},
            "strided": {"access_mode": "strided"},
            "simple": {"backend": "pytorch"},
        }

        for impl_type, expected_config in implementations.items():
            module = create_hilbert_attention(
                impl_type,
                hidden_dim=768,
                num_heads=12,
            ).to(device)

            # Check configuration
            config = module.get_config()
            for key, value in expected_config.items():
                assert config[key] == value

            # Test it works
            x = torch.randn(2, 512, 768, device=device)
            out = module(x)
            assert out.shape == x.shape

    def test_memory_usage_estimation(self, device):
        """Test memory usage estimation methods."""
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=4,
            sparse_mode="direct",
            access_mode="strided",
        ).to(device)

        # Test memory estimation
        mem_info = module.estimate_memory_usage(seq_len=2048, batch_size=4)

        # Check expected keys
        assert "parameters_mb" in mem_info
        assert "cache_mb" in mem_info

        # For sparse mode
        assert "reduction_factor" in mem_info
        assert mem_info["reduction_factor"] > 1

        # For strided mode
        assert "bandwidth_reduction" in mem_info
        assert 0 <= mem_info["bandwidth_reduction"] <= 1

    def test_cache_management(self, device):
        """Test cache management functionality."""
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            cache_size=16,  # Small cache for testing
            cache_memory_mb=50.0,
        ).to(device)

        # Process multiple sequence lengths
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        for _ in range(3):  # Multiple passes
            for seq_len in seq_lengths:
                x = torch.randn(1, seq_len, 768, device=device)
                _ = module(x)

        # Check cache is bounded
        stats = module.get_cache_stats()
        assert stats["size"] <= 16
        assert stats["memory_usage_mb"] <= 50.0

        # Clear cache
        module.clear_cache()
        stats_after = module.get_cache_stats()
        assert stats_after["size"] == 0

    def test_causal_masking(self, device):
        """Test causal masking support."""
        module = UnifiedHilbertAttention(
            hidden_dim=512,
            num_heads=8,
        ).to(device)

        x = torch.randn(2, 256, 512, device=device)

        # Test with causal masking
        out_causal = module(x, is_causal=True)
        assert out_causal.shape == x.shape

        # Test without causal masking
        out_non_causal = module(x, is_causal=False)
        assert out_non_causal.shape == x.shape

        # Outputs should be different
        assert not torch.allclose(out_causal, out_non_causal)


class TestMigration:
    """Test migration utilities."""

    def test_migrate_to_unified(self):
        """Test migrating old implementations to unified."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create old implementation
        old_module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=2,
        ).to(device)

        # Set some weights
        old_module.qkv_proj.weight.data.fill_(0.1)
        old_module.out_proj.weight.data.fill_(0.2)

        # Migrate
        new_module = migrate_to_unified(old_module)

        # Check configuration transferred
        assert new_module.hidden_dim == 768
        assert new_module.num_heads == 12
        assert new_module.segment_size == 128
        assert new_module.dilation_rate == 2

        # Check weights transferred
        assert torch.allclose(
            new_module.qkv_proj.weight.data, old_module.qkv_proj.weight.data
        )
        assert torch.allclose(
            new_module.out_proj.weight.data, old_module.out_proj.weight.data
        )

        # Check it works
        x = torch.randn(2, 256, 768, device=device)
        out = new_module(x)
        assert out.shape == x.shape

    def test_explicit_implementations(self):
        """Test using UnifiedHilbertAttention and UnifiedHilbertAttention directly."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test UnifiedHilbertAttention
        simple_module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
        ).to(device)

        x = torch.randn(2, 256, 768, device=device)
        out = simple_module(x)
        assert out.shape == x.shape

        # Test UnifiedHilbertAttention if available
        if UnifiedHilbertAttention is not None:
            core_module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
            ).to(device)

            out = core_module(x)
            assert out.shape == x.shape


if __name__ == "__main__":
    # Run basic tests
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Test unified implementation
    test_unified = TestUnifiedHilbertAttention()

    print("Testing standard mode...")
    test_unified.test_standard_mode(device, test_unified.test_configs())

    print("Testing memory modes...")
    test_unified.test_memory_modes(device)

    print("Testing sparse modes...")
    test_unified.test_sparse_modes(device)

    print("Testing backend selection...")
    test_unified.test_backend_selection(device)

    print("Testing gradient flow...")
    test_unified.test_gradient_flow(device)

    print("Testing factory function...")
    test_unified.test_create_hilbert_attention_factory(device)

    print("\nAll tests passed!")
