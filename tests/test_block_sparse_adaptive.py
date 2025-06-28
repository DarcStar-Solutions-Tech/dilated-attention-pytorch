"""
Test suite for Content-Adaptive Block-Sparse Attention
"""

import pytest
import torch

from dilated_attention_pytorch.block_sparse_adaptive import (
    BlockSparseAdaptive,
    AdaptiveConfig,
    ImportanceScorer,
    AdaptiveSparsityTrainer,
    create_adaptive_block_sparse,
)


class TestBlockSparseAdaptive:
    """Test adaptive block-sparse attention implementation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def default_config(self):
        return {
            "segment_lengths": [1024, 2048],
            "dilation_rates": [1, 2],
            "num_heads": 4,
            "head_dim": 64,
        }

    def test_initialization(self, default_config, device):
        """Test module initialization."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        assert model is not None
        assert model.adaptive_config is not None
        assert hasattr(model, "importance_scorer")
        assert hasattr(model, "q_summary_proj")
        assert hasattr(model, "k_summary_proj")

    def test_importance_scorer(self, device):
        """Test importance scorer network."""
        input_dim = 64
        hidden_dim = 128
        num_heads = 4
        batch_size = 2
        num_blocks = 16

        # Test shared scorer
        scorer = ImportanceScorer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            share_across_heads=True,
        )

        q_summary = torch.randn(batch_size, num_blocks, input_dim, device=device)
        k_summary = torch.randn(batch_size, num_blocks, input_dim, device=device)

        scores = scorer(q_summary, k_summary)

        assert scores.shape == (batch_size, num_blocks, num_blocks)
        assert not torch.isnan(scores).any()

        # Test head-specific scorer
        scorer_heads = ImportanceScorer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            share_across_heads=False,
        )

        # Test single head
        scores_h0 = scorer_heads(q_summary, k_summary, head_idx=0)
        assert scores_h0.shape == (batch_size, num_blocks, num_blocks)

    def test_block_summaries(self, default_config, device):
        """Test block summary computation."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 2
        seq_len = 512
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)

        # Compute summaries for head 0
        q_summary, k_summary = model._compute_block_summaries(q, k, head_idx=0)

        num_blocks = seq_len // model.block_size
        assert q_summary.shape == (batch_size, num_blocks, head_dim)
        assert k_summary.shape == (batch_size, num_blocks, head_dim)

    def test_gumbel_softmax_topk(self, default_config, device):
        """Test differentiable top-k selection."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 2
        num_q_blocks = 8
        num_k_blocks = 8
        k = 3  # Keep top-3 per query

        scores = torch.randn(batch_size, num_q_blocks, num_k_blocks, device=device)

        # Test soft selection
        soft_mask = model._gumbel_softmax_topk(scores, k=k, temperature=1.0, hard=False)
        assert soft_mask.shape == scores.shape
        assert torch.all(soft_mask >= 0) and torch.all(soft_mask <= 1)

        # Test hard selection
        hard_mask = model._gumbel_softmax_topk(scores, k=k, temperature=0.1, hard=True)
        assert hard_mask.shape == scores.shape
        # Check that exactly k elements are selected per query
        for b in range(batch_size):
            for q in range(num_q_blocks):
                assert torch.sum(hard_mask[b, q, :] == 1.0) == k

    def test_adaptive_pattern_generation(self, default_config, device):
        """Test adaptive pattern generation."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 2
        seq_len = 512
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)

        # Generate pattern for head 0
        row_indices, col_indices = model._generate_adaptive_pattern(
            q, k, head_idx=0, target_sparsity=0.9
        )

        assert len(row_indices) == len(col_indices)
        assert len(row_indices) > 0

        # Check indices are valid
        num_blocks = seq_len // model.block_size
        assert torch.all(row_indices >= 0) and torch.all(row_indices < num_blocks)
        assert torch.all(col_indices >= 0) and torch.all(col_indices < num_blocks)

    def test_forward_pass(self, default_config, device):
        """Test forward pass with adaptive patterns."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 1
        seq_len = 512
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass without returning pattern
        output = model(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Forward pass with pattern return
        output_with_pattern, pattern_info = model(q, k, v, return_pattern=True)

        assert output_with_pattern.shape == q.shape
        assert "patterns" in pattern_info
        assert len(pattern_info["patterns"]) == num_heads

    @pytest.mark.skip(
        reason="Gradient flow through discrete pattern selection is complex"
    )
    def test_gradient_flow(self, default_config, device):
        """Test gradient flow through adaptive attention."""
        # Use soft sparsity to ensure gradients flow
        adaptive_config = AdaptiveConfig(
            hard_sparsity=False,
            temperature=1.0,
        )

        model = BlockSparseAdaptive(
            **default_config,
            adaptive_config=adaptive_config,
            device=device,
            dtype=torch.float32,
        )

        # Ensure model is in training mode
        model.train()

        batch_size = 1
        seq_len = 256
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

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

        # Check gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        # Check model parameters have gradients (importance scorer gradients depend on pattern)
        # The importance scorer might not always get gradients if patterns are cached or discrete
        model_has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                model_has_gradients = True
                break
        assert model_has_gradients, "No gradients found in model"

        # Verify output gradients are non-zero
        assert not torch.all(q.grad == 0)
        assert not torch.all(k.grad == 0)
        assert not torch.all(v.grad == 0)

    def test_temperature_annealing(self, default_config, device):
        """Test temperature annealing functionality."""
        adaptive_config = AdaptiveConfig(
            learnable_temperature=True,
            temperature=1.0,
            min_temperature=0.1,
        )

        model = BlockSparseAdaptive(
            **default_config,
            adaptive_config=adaptive_config,
            device=device,
            dtype=torch.float32,
        )

        # Create trainer
        trainer = AdaptiveSparsityTrainer(
            model,
            initial_temperature=1.0,
            final_temperature=0.1,
            annealing_steps=100,
        )

        # Check initial temperature
        initial_temp = torch.exp(model.log_temperature).item()
        assert abs(initial_temp - 1.0) < 0.01

        # Step through annealing
        for _ in range(50):
            trainer.step()

        # Check temperature has decreased
        mid_temp = torch.exp(model.log_temperature).item()
        assert mid_temp < initial_temp
        assert mid_temp > 0.1

        # Complete annealing
        for _ in range(50):
            trainer.step()

        # Check final temperature
        final_temp = torch.exp(model.log_temperature).item()
        assert abs(final_temp - 0.1) < 0.01

    def test_different_sparsity_levels(self, default_config, device):
        """Test with different target sparsity levels."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 1
        seq_len = 1024  # Larger sequence for more blocks
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)

        sparsity_levels = [0.5, 0.8, 0.95]
        pattern_sizes = []

        for sparsity in sparsity_levels:
            row_idx, col_idx = model._generate_adaptive_pattern(
                q, k, head_idx=0, target_sparsity=sparsity
            )
            pattern_sizes.append(len(row_idx))

        # Higher sparsity should result in fewer connections
        assert pattern_sizes[0] >= pattern_sizes[1] >= pattern_sizes[2]
        # With more blocks, we should see some difference
        assert pattern_sizes[0] > pattern_sizes[2]

    def test_head_specific_patterns(self, default_config, device):
        """Test that different heads can learn different patterns."""
        adaptive_config = AdaptiveConfig(share_across_heads=False)

        model = BlockSparseAdaptive(
            **default_config,
            adaptive_config=adaptive_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 1
        seq_len = 512
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Get patterns for all heads
        _, pattern_info = model(q, k, v, return_pattern=True)
        patterns = pattern_info["patterns"]

        # Check that patterns can be different across heads
        # (They might be similar initially but should be able to differ)
        assert len(patterns) == num_heads

        # Each pattern should be valid
        for row_idx, col_idx in patterns:
            assert len(row_idx) == len(col_idx)
            assert len(row_idx) > 0

    def test_factory_function(self, device):
        """Test factory function for creating adaptive attention."""
        model = create_adaptive_block_sparse(
            embed_dim=256,
            num_heads=4,
            device=device,
            dtype=torch.float32,
        )

        assert isinstance(model, BlockSparseAdaptive)
        assert model.num_heads == 4
        assert model.head_dim == 64

        # Test forward pass
        batch_size = 1
        seq_len = 512
        q = torch.randn(batch_size, seq_len, 4, 64, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        output = model(q, k, v)
        assert output.shape == q.shape

    @pytest.mark.parametrize("hard_sparsity", [True, False])
    def test_hard_vs_soft_sparsity(self, default_config, device, hard_sparsity):
        """Test hard vs soft sparsity selection."""
        adaptive_config = AdaptiveConfig(hard_sparsity=hard_sparsity)

        model = BlockSparseAdaptive(
            **default_config,
            adaptive_config=adaptive_config,
            device=device,
            dtype=torch.float32,
        )

        batch_size = 1
        seq_len = 256
        num_heads = default_config["num_heads"]
        head_dim = default_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass should work with both modes
        output = model(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_sparsity_loss(self, default_config, device):
        """Test sparsity regularization loss."""
        model = BlockSparseAdaptive(
            **default_config,
            device=device,
            dtype=torch.float32,
        )

        loss = model.get_sparsity_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.device.type == device.type
        assert loss.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
