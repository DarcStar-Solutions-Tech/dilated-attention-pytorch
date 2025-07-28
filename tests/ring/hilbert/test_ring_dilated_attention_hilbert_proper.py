"""
Unit tests for HilbertRingAttention implementation.

Tests verify:
- Correct integration of ring communication, dilated attention, and Hilbert SFC
- Per-segment processing (not global)
- Gradient flow
- Numerical stability
"""

import pytest
import torch
import torch.distributed as dist

from dilated_attention_pytorch.ring import HilbertRingAttention, RingAttentionConfig


class TestHilbertRingAttention:
    """Test suite for HilbertRingAttention."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def config(self):
        """Standard configuration."""
        return RingAttentionConfig(
            segment_lengths=[128, 256, 512],
            dilation_rates=[1, 2, 4],
            dropout=0.1,
            use_hilbert=True,
            hilbert_curve_level=8,
        )

    def test_initialization(self, config, device):
        """Test model initialization."""
        model = HilbertRingAttention(config, device=device)

        assert model.segment_lengths == config.segment_lengths
        assert model.dilation_rates == config.dilation_rates
        assert model.use_hilbert == config.use_hilbert
        assert model.hilbert_curve_level == config.hilbert_curve_level

    def test_forward_shape(self, config, device):
        """Test forward pass output shape."""
        model = HilbertRingAttention(config, device=device)
        batch_size, seq_len = 2, 512
        num_heads = 8
        head_dim = 32

        # Create Q, K, V tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)

        output = model(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("seq_len", [256, 512, 1024, 2048])
    def test_variable_sequence_lengths(self, config, device, seq_len):
        """Test with different sequence lengths."""
        model = HilbertRingAttention(config, device=device)
        batch_size = 2
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)

        output = model(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    def test_causal_masking(self, config, device):
        """Test causal masking is applied correctly."""
        model = HilbertRingAttention(config, device=device)
        model.eval()

        seq_len = 256
        batch_size = 1
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)

        with torch.no_grad():
            # Get outputs with and without causal masking
            out_causal = model(q, k, v, is_causal=True)
            out_non_causal = model(q, k, v, is_causal=False)

        # Outputs should be different
        assert not torch.allclose(out_causal, out_non_causal)

    def test_gradient_flow(self, config, device):
        """Test gradient flow through the model."""
        model = HilbertRingAttention(config, device=device)

        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 32

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, requires_grad=True
        ).to(device)
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, requires_grad=True
        ).to(device)
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, requires_grad=True
        ).to(device)

        output = model(q, k, v)
        loss = output.mean()
        loss.backward()

        # Check input gradients
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()
        assert q.grad.abs().mean() > 0
        assert k.grad.abs().mean() > 0
        assert v.grad.abs().mean() > 0

    def test_hilbert_ordering_effect(self, device):
        """Test that Hilbert ordering has an effect."""
        # Create two configs with and without Hilbert
        config_hilbert = RingAttentionConfig(
            segment_lengths=[128, 256, 512],
            dilation_rates=[1, 2, 4],
            dropout=0.0,
            use_hilbert=True,
            hilbert_curve_level=8,
        )

        config_no_hilbert = RingAttentionConfig(
            segment_lengths=[128, 256, 512],
            dilation_rates=[1, 2, 4],
            dropout=0.0,
            use_hilbert=False,
        )

        model_hilbert = HilbertRingAttention(config_hilbert, device=device)
        model_no_hilbert = HilbertRingAttention(config_no_hilbert, device=device)

        # Set to eval mode to avoid dropout
        model_hilbert.eval()
        model_no_hilbert.eval()

        # Create input
        batch_size = 1
        seq_len = 512
        num_heads = 8
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)

        with torch.no_grad():
            out_hilbert = model_hilbert(q, k, v)
            out_no_hilbert = model_no_hilbert(q, k, v)

        # Outputs should be different due to different ordering
        assert not torch.allclose(out_hilbert, out_no_hilbert, rtol=1e-5)

    @pytest.mark.skipif(
        not torch.distributed.is_initialized(), reason="Distributed not initialized"
    )
    def test_distributed_ring_communication(self, config, device):
        """Test ring communication in distributed setting."""
        model = HilbertRingAttention(config, device=device)

        batch_size = 1
        seq_len = 1024
        num_heads = 8
        head_dim = 32

        # Get world size and rank
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Create full sequence
        torch.manual_seed(42)  # Same seed on all ranks
        q_full = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        k_full = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)
        v_full = torch.randn(batch_size, seq_len, num_heads, head_dim).to(device)

        # Split for this rank
        chunk_size = seq_len // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size

        q_local = q_full[:, start_idx:end_idx].contiguous()
        k_local = k_full[:, start_idx:end_idx].contiguous()
        v_local = v_full[:, start_idx:end_idx].contiguous()

        # Forward with already_split=True
        output_local = model(q_local, k_local, v_local, already_split=True)

        # Check output shape
        assert output_local.shape == (batch_size, chunk_size, num_heads, head_dim)
        assert not torch.isnan(output_local).any()
        assert not torch.isinf(output_local).any()

    def test_attention_pattern_consistency(self, config, device):
        """Test that attention patterns are consistent."""
        model = HilbertRingAttention(config, device=device)
        model.eval()

        batch_size = 1
        seq_len = 256
        num_heads = 8
        head_dim = 32

        # Create input with specific pattern
        q = torch.eye(seq_len, seq_len).unsqueeze(0).unsqueeze(2).to(device)
        q = q.expand(batch_size, seq_len, num_heads, head_dim).contiguous()
        k = q.clone()
        v = torch.ones(batch_size, seq_len, num_heads, head_dim).to(device)

        with torch.no_grad():
            output = model(q, k, v)

        # Check output characteristics
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
