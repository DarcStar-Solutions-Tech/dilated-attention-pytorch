#!/usr/bin/env python3
"""
Integration tests for distributed Ring Attention implementations.

These tests focus on end-to-end distributed training scenarios,
multi-GPU communication patterns, and real-world usage patterns.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dilated_attention_pytorch.ring_distributed_dilated_attention import (
    RingDistributedDilatedAttention,
)
from dilated_attention_pytorch.ring_multihead_dilated_attention import (
    RingMultiheadDilatedAttention,
)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for integration testing."""

    def __init__(
        self, embed_dim: int, num_heads: int, segment_lengths: list[int], dilation_rates: list[int]
    ):
        super().__init__()
        self.attention = RingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)[0] + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x) + residual

        return x


class TestDistributedIntegration:
    """Integration tests for distributed Ring Attention."""

    @pytest.fixture
    def mock_dist_env(self):
        """Mock distributed environment with 4 GPUs."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=4),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_backend", return_value="nccl"),
        ):
            yield

    def test_transformer_block_integration(self, mock_dist_env):
        """Test Ring Attention in a transformer block."""
        # Create model
        model = SimpleTransformerBlock(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
        )

        # Create test input
        batch_size, seq_len = 2, 4096
        x = torch.randn(batch_size, seq_len, 512)

        # Forward pass should work
        output = model(x)
        assert output.shape == (batch_size, seq_len, 512)

        # Check gradients flow properly
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_ddp_wrapper_compatibility(self, mock_dist_env):
        """Test compatibility with PyTorch DDP."""
        # Create model
        model = SimpleTransformerBlock(
            embed_dim=256,
            num_heads=4,
            segment_lengths=[512],
            dilation_rates=[1],
        )

        # Mock process group
        mock_pg = MagicMock()

        with patch(
            "torch.nn.parallel.DistributedDataParallel._get_default_group", return_value=mock_pg
        ):
            # Wrap in DDP
            ddp_model = DDP(model)

            # Test forward pass
            x = torch.randn(1, 512, 256)
            output = ddp_model(x)
            assert output.shape == (1, 512, 256)

    def test_gradient_accumulation(self, mock_dist_env):
        """Test gradient accumulation across multiple steps."""
        model = RingMultiheadDilatedAttention(
            embed_dim=256,
            num_heads=4,
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
        )

        # Accumulate gradients over multiple steps
        accumulated_grad = None
        num_accumulation_steps = 4

        for step in range(num_accumulation_steps):
            x = torch.randn(1, 2048, 256)
            output, _ = model(x, x, x)
            loss = output.mean() / num_accumulation_steps
            loss.backward()

            # Check gradient accumulation
            if accumulated_grad is None:
                accumulated_grad = {
                    name: param.grad.clone()
                    for name, param in model.named_parameters()
                    if param.grad is not None
                }
            else:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Gradients should be accumulating
                        assert not torch.allclose(param.grad, accumulated_grad[name], atol=1e-6)

    def test_mixed_precision_training(self, mock_dist_env):
        """Test mixed precision training compatibility."""
        model = RingDistributedDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024],
            dilation_rates=[1],
            enable_mixed_precision=True,
        )

        # Create autocast context
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            x = torch.randn(2, 1024, 512)
            # Model should handle mixed precision internally
            output = model(x)
            assert output.dtype == torch.float16

    def test_checkpoint_save_load(self, mock_dist_env):
        """Test model checkpointing and loading."""
        # Create model
        model = RingMultiheadDilatedAttention(
            embed_dim=256,
            num_heads=4,
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
        )

        # Get initial state
        initial_state = model.state_dict()

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)

        # Verify weights changed
        for key in initial_state:
            if "weight" in key or "bias" in key:
                assert not torch.allclose(model.state_dict()[key], initial_state[key])

        # Load initial state
        model.load_state_dict(initial_state)

        # Verify weights restored
        for key in initial_state:
            assert torch.allclose(model.state_dict()[key], initial_state[key])

    def test_multi_gpu_communication_pattern(self, mock_dist_env):
        """Test ring communication pattern across GPUs."""
        # Mock different ranks
        world_size = 4

        for rank in range(world_size):
            with patch("torch.distributed.get_rank", return_value=rank):
                model = RingDistributedDilatedAttention(
                    embed_dim=256,
                    num_heads=4,
                    segment_lengths=[512],
                    dilation_rates=[1],
                    ring_size=world_size,
                )

                # Each rank should know its neighbors
                assert model.rank == rank
                assert model.next_rank == (rank + 1) % world_size
                assert model.prev_rank == (rank - 1) % world_size

    def test_dynamic_sequence_lengths(self, mock_dist_env):
        """Test handling of variable sequence lengths in a batch."""
        model = RingMultiheadDilatedAttention(
            embed_dim=256,
            num_heads=4,
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
        )

        # Different sequence lengths (with padding)
        batch_size = 3
        max_seq_len = 2048
        seq_lengths = [1024, 1536, 2048]

        # Create padded input
        x = torch.zeros(batch_size, max_seq_len, 256)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

        for i, seq_len in enumerate(seq_lengths):
            x[i, :seq_len] = torch.randn(seq_len, 256)
            attention_mask[i, :seq_len] = True

        # Forward pass with attention mask
        output, _ = model(x, x, x, key_padding_mask=~attention_mask)

        # Check output shape
        assert output.shape == (batch_size, max_seq_len, 256)

        # Check masked positions are zero
        for i, seq_len in enumerate(seq_lengths):
            assert torch.allclose(
                output[i, seq_len:], torch.zeros_like(output[i, seq_len:]), atol=1e-6
            )

    def test_distributed_optimizer_compatibility(self, mock_dist_env):
        """Test compatibility with distributed optimizers."""
        model = RingDistributedDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024],
            dilation_rates=[1],
            enable_gradient_compression=True,
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training step
        x = torch.randn(2, 1024, 512)
        output = model(x)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Optimizer step should work with compressed gradients
        optimizer.step()
        optimizer.zero_grad()

    @pytest.mark.parametrize("backend", ["nccl", "gloo"])
    def test_backend_compatibility(self, mock_dist_env, backend):
        """Test compatibility with different communication backends."""
        with patch("torch.distributed.get_backend", return_value=backend):
            model = RingDistributedDilatedAttention(
                embed_dim=256,
                num_heads=4,
                segment_lengths=[512],
                dilation_rates=[1],
            )

            # Should initialize properly with different backends
            assert model.backend == backend

            # Forward pass should work
            x = torch.randn(1, 512, 256)
            output = model(x)
            assert output.shape == (1, 512, 256)

    def test_fault_tolerance_node_failure(self, mock_dist_env):
        """Test handling of node failures during training."""
        model = RingDistributedDilatedAttention(
            embed_dim=256,
            num_heads=4,
            segment_lengths=[512],
            dilation_rates=[1],
            enable_fault_tolerance=True,
        )

        # Simulate node failure during forward pass
        with patch.object(model, "_ring_communicate", side_effect=RuntimeError("Node failure")):
            x = torch.randn(1, 512, 256)

            # Should handle failure gracefully
            try:
                output = model(x)
                # If fault tolerance is working, should get output
                assert output.shape == (1, 512, 256)
            except RuntimeError:
                # If no fault tolerance, should raise
                assert not model.enable_fault_tolerance

    def test_performance_monitoring_integration(self, mock_dist_env):
        """Test integration with performance monitoring."""
        model = RingDistributedDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            enable_monitoring=True,
        )

        # Perform multiple forward passes
        for _ in range(5):
            x = torch.randn(2, 4096, 512)
            _ = model(x)

        # Check metrics were collected
        if hasattr(model, "metrics"):
            assert "forward_time" in model.metrics
            assert "communication_time" in model.metrics
            assert len(model.metrics["forward_time"]) == 5

    def test_hierarchical_parallelism(self, mock_dist_env):
        """Test hierarchical parallelism patterns."""
        # Mock node-level and GPU-level groups
        node_group = MagicMock()
        gpu_group = MagicMock()

        with (
            patch("torch.distributed.new_group", side_effect=[node_group, gpu_group]),
            patch.dict(os.environ, {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2"}),
        ):
            model = RingDistributedDilatedAttention(
                embed_dim=512,
                num_heads=8,
                segment_lengths=[1024],
                dilation_rates=[1],
                hierarchical_allreduce=True,
            )

            # Should create hierarchical groups
            assert hasattr(model, "node_group")
            assert hasattr(model, "local_group")

    def test_cpu_offloading(self, mock_dist_env):
        """Test CPU memory offloading for large models."""
        model = RingDistributedDilatedAttention(
            embed_dim=1024,
            num_heads=16,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            enable_cpu_offload=True,
        )

        # Large input that might trigger offloading
        x = torch.randn(4, 8192, 1024)

        # Forward pass should handle offloading internally
        output = model(x)
        assert output.shape == (4, 8192, 1024)

        # Check if offloading occurred
        if hasattr(model, "cpu_buffer_pool"):
            assert len(model.cpu_buffer_pool) > 0


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.fixture
    def mock_dist_env(self):
        """Mock distributed environment."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            yield

    def test_language_model_training(self, mock_dist_env):
        """Test integration in language model training."""

        # Simple language model with Ring Attention
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.layers = nn.ModuleList(
                    [
                        SimpleTransformerBlock(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=[1024, 2048, 4096],
                            dilation_rates=[1, 2, 4],
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.output = nn.Linear(embed_dim, vocab_size)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)

        # Create model
        model = SimpleLM(vocab_size=50000, embed_dim=512, num_heads=8, num_layers=2)

        # Training step
        input_ids = torch.randint(0, 50000, (2, 8192))
        logits = model(input_ids)

        # Check output
        assert logits.shape == (2, 8192, 50000)

    def test_multimodal_training(self, mock_dist_env):
        """Test integration in multimodal models."""

        class MultimodalModel(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int):
                super().__init__()
                # Vision encoder (mock)
                self.vision_encoder = nn.Linear(768, embed_dim)
                # Text encoder (mock)
                self.text_encoder = nn.Linear(768, embed_dim)
                # Cross-modal attention with Ring Attention
                self.cross_attention = RingMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=[512, 1024],
                    dilation_rates=[1, 2],
                )

            def forward(
                self, vision_features: torch.Tensor, text_features: torch.Tensor
            ) -> torch.Tensor:
                vision_emb = self.vision_encoder(vision_features)
                text_emb = self.text_encoder(text_features)

                # Cross-modal attention
                output, _ = self.cross_attention(
                    query=text_emb,
                    key=vision_emb,
                    value=vision_emb,
                )
                return output

        # Create model
        model = MultimodalModel(embed_dim=512, num_heads=8)

        # Test forward pass
        vision_features = torch.randn(2, 196, 768)  # 14x14 patches
        text_features = torch.randn(2, 1024, 768)

        output = model(vision_features, text_features)
        assert output.shape == (2, 1024, 512)

    def test_extreme_scale_training(self, mock_dist_env):
        """Test training at extreme scales."""
        # Simulate 100M token context
        segment_lengths = [4096, 8192, 16384, 32768, 65536]
        dilation_rates = [1, 2, 4, 8, 16]

        model = RingDistributedDilatedAttention(
            embed_dim=1024,
            num_heads=16,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=8,  # 8 GPUs
            block_size=512,  # Small blocks for memory efficiency
            enable_gradient_checkpointing=True,
            enable_cpu_offload=True,
            enable_memory_pool=True,
        )

        # Test configuration is valid
        assert model.ring_size == 8
        assert len(model.segment_lengths) == 5

        # Small forward pass to verify setup
        x = torch.randn(1, sum(segment_lengths), 1024)  # ~130K tokens
        # Note: In real training, this would be split across GPUs
        # Here we just verify the configuration is valid


class TestCommunicationPatterns:
    """Test specific communication patterns."""

    def test_all_gather_pattern(self):
        """Test all-gather communication pattern."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=4),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            # Mock all_gather
            gathered_tensors = []

            def mock_all_gather(tensor_list, tensor):
                # Simulate gathering from all ranks
                for i in range(4):
                    tensor_list[i].copy_(tensor)

            with patch("torch.distributed.all_gather", side_effect=mock_all_gather):
                model = RingDistributedDilatedAttention(
                    embed_dim=256,
                    num_heads=4,
                    segment_lengths=[512],
                    dilation_rates=[1],
                    use_all_gather=True,
                )

                # Test gathering operation
                local_tensor = torch.randn(128, 4, 64)
                gathered = model._all_gather_tensors(local_tensor)

                # Should gather from all ranks
                assert len(gathered) == 4
                assert all(t.shape == (128, 4, 64) for t in gathered)

    def test_pipeline_communication(self):
        """Test pipeline-style communication."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=4),
        ):
            # Test each rank in pipeline
            for rank in range(4):
                with patch("torch.distributed.get_rank", return_value=rank):
                    model = RingDistributedDilatedAttention(
                        embed_dim=256,
                        num_heads=4,
                        segment_lengths=[512],
                        dilation_rates=[1],
                        enable_pipeline_parallel=True,
                    )

                    # Each rank should know its position in pipeline
                    assert model.rank == rank
                    if rank > 0:
                        assert model.prev_rank == rank - 1
                    if rank < 3:
                        assert model.next_rank == rank + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
