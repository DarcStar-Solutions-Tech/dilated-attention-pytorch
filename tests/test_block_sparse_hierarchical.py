"""
Test suite for Block-Sparse Hierarchical Attention Patterns
"""

import pytest
import torch

from dilated_attention_pytorch.block_sparse_hierarchical import (
    BlockSparseHierarchical,
    HierarchicalConfig,
    create_hierarchical_attention,
    get_hierarchical_presets,
)


class TestBlockSparseHierarchical:
    """Test hierarchical block-sparse attention implementation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def default_config(self):
        return {
            "segment_lengths": [1024, 2048, 4096],
            "dilation_rates": [1, 2, 4],
        }

    def test_initialization(self, default_config, device):
        """Test module initialization."""
        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        assert model is not None
        assert model.hierarchical_config is not None
        assert model.hierarchical_config.num_levels == 3

    def test_hierarchical_pattern_generation(self, default_config, device):
        """Test hierarchical pattern generation."""
        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        seq_len = 1024
        num_heads = 8

        row_indices, col_indices = model._generate_hierarchical_pattern(
            seq_len, num_heads, device
        )

        # Check pattern properties
        assert row_indices.device.type == device.type
        assert col_indices.device.type == device.type
        assert len(row_indices) == len(col_indices)
        assert len(row_indices) > 0

        # Verify indices are within bounds
        num_blocks = seq_len // model.block_size
        assert torch.all(row_indices >= 0)
        assert torch.all(row_indices < num_blocks)
        assert torch.all(col_indices >= 0)
        assert torch.all(col_indices < num_blocks)

    def test_forward_pass(self, default_config, device):
        """Test forward pass with hierarchical patterns."""
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        output = model(q, k, v)

        assert output.shape == q.shape
        assert output.dtype == q.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_pattern_stats(self, default_config, device):
        """Test pattern statistics calculation."""
        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        seq_len = 2048
        stats = model.get_pattern_stats(seq_len)

        assert "total_blocks" in stats
        assert "active_blocks" in stats
        assert "sparsity" in stats
        assert "levels" in stats

        # Check sparsity is reasonable
        assert 0.0 <= stats["sparsity"] <= 1.0
        assert stats["active_blocks"] <= stats["total_blocks"]

        # Check level stats
        assert len(stats["levels"]) == model.hierarchical_config.num_levels
        for level_stat in stats["levels"]:
            assert "stride" in level_stat
            assert "window_size" in level_stat
            assert "coverage" in level_stat

    def test_different_hierarchical_configs(self, default_config, device):
        """Test different hierarchical configurations."""
        presets = get_hierarchical_presets()

        batch_size = 1
        seq_len = 1024
        num_heads = 4
        head_dim = 64

        for preset_name, hierarchical_config in presets.items():
            model = BlockSparseHierarchical(
                **default_config,
                hierarchical_config=hierarchical_config,
                device=device,
                dtype=torch.float32,
            )

            # Create inputs
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Forward pass
            output = model(q, k, v)

            assert output.shape == q.shape
            assert not torch.isnan(output).any()

            # Check pattern stats
            stats = model.get_pattern_stats(seq_len)
            print(f"\n{preset_name} preset - Sparsity: {stats['sparsity']:.1%}")

    def test_pattern_visualization(self, default_config, device):
        """Test pattern visualization."""
        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        # Generate visualization
        viz = model.visualize_pattern(seq_len=512)

        assert isinstance(viz, str)
        assert "Hierarchical Attention Pattern" in viz
        assert "Legend" in viz
        assert "Sparsity" in viz

        # Print for visual inspection
        print("\n" + viz)

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    def test_different_sequence_lengths(self, default_config, device, seq_len):
        """Test with different sequence lengths."""
        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 1
        num_heads = 4
        head_dim = 64

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        output = model(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, default_config, device):
        """Test gradient flow through hierarchical attention."""
        model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 1
        seq_len = 512
        num_heads = 4
        head_dim = 64

        # Create inputs with gradient tracking
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        # Forward pass
        output = model(q, k, v)

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Check gradients exist and are not zero
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        assert not torch.isnan(q.grad).any()
        assert not torch.all(q.grad == 0)

    def test_memory_efficiency(self, default_config, device):
        """Test memory efficiency compared to dense attention."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")

        batch_size = 1
        seq_len = 4096
        num_heads = 8
        head_dim = 64

        # Create hierarchical model
        hierarchical_model = BlockSparseHierarchical(
            **default_config,
            device=device,
            dtype=torch.float16,
        )

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Measure hierarchical memory
        torch.cuda.reset_peak_memory_stats()
        _ = hierarchical_model(q, k, v)
        hierarchical_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        # Get pattern stats
        stats = hierarchical_model.get_pattern_stats(seq_len)

        print(f"\nHierarchical attention memory: {hierarchical_memory:.2f} MB")
        print(f"Sparsity: {stats['sparsity']:.1%}")
        print(f"Active blocks: {stats['active_blocks']}/{stats['total_blocks']}")

    def test_factory_function(self, device):
        """Test factory function for creating hierarchical attention."""
        model = create_hierarchical_attention(
            embed_dim=512,
            num_heads=8,
            device=device,
            dtype=torch.float32,
        )

        assert isinstance(model, BlockSparseHierarchical)

        # Test forward pass
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        output = model(q, k, v)
        assert output.shape == q.shape

    def test_custom_hierarchical_config(self, default_config, device):
        """Test with custom hierarchical configuration."""
        # Create custom 4-level hierarchy
        custom_config = HierarchicalConfig(
            level_configs=[
                {"stride": 1, "window_size": 64, "block_size": 16},
                {"stride": 4, "window_size": 256, "block_size": 32},
                {"stride": 16, "window_size": 1024, "block_size": 64},
                {"stride": 64, "window_size": -1, "block_size": 128},
            ]
        )

        model = BlockSparseHierarchical(
            **default_config,
            hierarchical_config=custom_config,
            device=device,
            dtype=torch.float32,
        )

        assert model.hierarchical_config.num_levels == 4

        # Test pattern generation
        stats = model.get_pattern_stats(2048)
        assert len(stats["levels"]) == 4

        # Visualize for inspection
        print("\nCustom 4-level hierarchy:")
        print(model.visualize_pattern(seq_len=512))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
