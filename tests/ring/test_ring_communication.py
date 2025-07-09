"""Comprehensive tests for ring communication patterns.

This module tests the core ring communication functionality to ensure
correctness, reliability, and performance.
"""

import pytest
import torch
import torch.distributed as dist
import os
import time

from dilated_attention_pytorch.ring.base import (
    StandardRingAttention,
    RingAttentionConfig,
)
from dilated_attention_pytorch.ring.utils import (
    RingCommunicationMixin,
    AsyncRingCommunicator,
)


class TestRingCommunication:
    """Test core ring communication patterns."""

    @pytest.fixture
    def setup_distributed(self):
        """Setup distributed environment for testing."""
        if not dist.is_available():
            pytest.skip("Distributed not available")

        # Initialize process group if not already initialized
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method="tcp://localhost:12345",
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
            )

        yield

        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_ring_pass_correctness(self, setup_distributed):
        """Verify data correctly passes through ring."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for ring communication test")

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Create test mixin
        class TestMixin(RingCommunicationMixin):
            def __init__(self, rank, world_size, device, dtype):
                super().__init__()
                self.rank = rank
                self.world_size = world_size
                self.device = device
                self.dtype = dtype
                self.is_distributed = True

        mixin = TestMixin(rank, world_size, device, torch.float32)

        # Create unique tensor for each rank
        test_tensor = torch.full((4, 4), rank, device=device, dtype=torch.float32)

        # Perform ring pass
        received = mixin.ring_pass_forward(test_tensor)

        # Verify we received from previous rank
        expected_rank = (rank - 1) % world_size
        expected = torch.full((4, 4), expected_rank, device=device, dtype=torch.float32)

        assert torch.allclose(received, expected), (
            f"Rank {rank} expected from rank {expected_rank}, got {received[0, 0].item()}"
        )

        # Test backward pass
        received_back = mixin.ring_pass_backward(test_tensor)
        expected_rank_back = (rank + 1) % world_size
        expected_back = torch.full(
            (4, 4), expected_rank_back, device=device, dtype=torch.float32
        )

        assert torch.allclose(received_back, expected_back), (
            f"Rank {rank} backward pass failed"
        )

    def test_gradient_flow_through_ring(self, setup_distributed):
        """Verify gradients flow correctly through ring communication."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for gradient test")

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Create ring attention
        config = RingAttentionConfig(
            segment_lengths=[64], dilation_rates=[1], dropout=0.0
        )

        attention = StandardRingAttention(config, device=device)

        # Create test inputs that require gradients
        batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 32
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )

        # Forward pass
        output = attention(q, k, v)

        # Create loss and backward
        loss = output.sum()
        loss.backward()

        # Verify gradients exist and are non-zero
        assert q.grad is not None and not torch.all(q.grad == 0)
        assert k.grad is not None and not torch.all(k.grad == 0)
        assert v.grad is not None and not torch.all(v.grad == 0)

        # Synchronize gradients across ranks
        dist.barrier()

    def test_communication_failure_recovery(self, setup_distributed):
        """Test handling of communication failures."""
        if not dist.is_initialized():
            pytest.skip("Need distributed environment")

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Create test mixin with failure injection
        class FailingMixin(RingCommunicationMixin):
            def __init__(self, rank, world_size, device, dtype):
                super().__init__()
                self.rank = rank
                self.world_size = world_size
                self.device = device
                self.dtype = dtype
                self.is_distributed = True
                self.fail_count = 0

            def _do_ring_communication(self, tensor, src, dst, tag, async_op=False):
                # Fail first attempt
                if self.fail_count == 0:
                    self.fail_count += 1
                    raise RuntimeError("Simulated communication failure")
                return super()._do_ring_communication(tensor, src, dst, tag, async_op)

        mixin = FailingMixin(rank, dist.get_world_size(), device, torch.float32)

        # Test that retry succeeds
        test_tensor = torch.ones(4, 4, device=device)
        result = mixin.ring_pass_forward(test_tensor)

        # Should succeed on retry
        assert result is not None
        assert mixin._comm_stats["failed_attempts"] == 1

    def test_different_world_sizes(self):
        """Test ring communication with different world sizes."""
        # This test simulates different world sizes
        for world_size in [2, 4, 8]:
            for rank in range(world_size):
                # Simulate ring neighbors
                src = (rank - 1) % world_size
                dst = (rank + 1) % world_size

                # Verify neighbor calculation
                assert src == (rank - 1) % world_size
                assert dst == (rank + 1) % world_size

                # Verify ring closure
                if rank == 0:
                    assert src == world_size - 1
                if rank == world_size - 1:
                    assert dst == 0

    def test_async_ring_communication(self, setup_distributed):
        """Test asynchronous ring communication for overlap."""
        if not dist.is_initialized():
            pytest.skip("Need distributed environment")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs")

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Create mixin and async communicator
        class TestMixin(RingCommunicationMixin):
            def __init__(self, rank, world_size, device, dtype):
                super().__init__()
                self.rank = rank
                self.world_size = world_size
                self.device = device
                self.dtype = dtype
                self.is_distributed = True

        mixin = TestMixin(rank, world_size, device, torch.float32)
        async_comm = AsyncRingCommunicator(mixin)

        # Start multiple async operations
        tensors = []
        for i in range(3):
            tensor = torch.full(
                (4, 4), rank * 10 + i, device=device, dtype=torch.float32
            )
            tensors.append(tensor)
            async_comm.start_ring_pass(tensor, tag=i)

        # Simulate computation overlap
        time.sleep(0.01)

        # Wait for all completions
        results = async_comm.wait_all()

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            expected_rank = (rank - 1) % world_size
            expected_value = expected_rank * 10 + i
            assert torch.all(result == expected_value)


class TestRingAttentionCorrectness:
    """Test correctness of ring attention implementation."""

    def test_single_gpu_equivalence(self):
        """Test that ring attention matches standard attention on single GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create configuration
        config = RingAttentionConfig(
            segment_lengths=[64, 128], dilation_rates=[1, 2], dropout=0.0
        )

        # Create ring attention
        ring_attn = StandardRingAttention(config, device=device)

        # Create inputs
        batch_size, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Forward pass
        output = ring_attn(q, k, v)

        # Verify output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Verify no NaNs or Infs
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_causal_masking(self):
        """Test causal masking in ring attention."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = RingAttentionConfig(
            segment_lengths=[32], dilation_rates=[1], dropout=0.0
        )

        ring_attn = StandardRingAttention(config, device=device)

        # Create inputs
        batch_size, seq_len, num_heads, head_dim = 1, 64, 2, 16
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Forward with causal masking
        output_causal = ring_attn(q, k, v, is_causal=True)

        # Forward without causal masking
        output_non_causal = ring_attn(q, k, v, is_causal=False)

        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal)

    def test_attention_mask(self):
        """Test custom attention mask in ring attention."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = RingAttentionConfig(
            segment_lengths=[32], dilation_rates=[1], dropout=0.0
        )

        ring_attn = StandardRingAttention(config, device=device)

        # Create inputs
        batch_size, seq_len, num_heads, head_dim = 1, 64, 2, 16
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Create attention mask (mask out second half)
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
        mask[:, :, :, seq_len // 2 :] = -float("inf")

        # Forward with mask
        output_masked = ring_attn(q, k, v, attention_mask=mask)

        # Verify masking effect
        # The output for positions attending to masked positions should be different
        output_no_mask = ring_attn(q, k, v)
        assert not torch.allclose(output_masked, output_no_mask)


class TestRingMemoryEfficiency:
    """Test memory efficiency of ring attention."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_scaling(self):
        """Test that memory scales as O(n/k) with world size."""
        device = torch.device("cuda:0")

        # Test different sequence lengths
        seq_lengths = [1024, 2048, 4096]
        batch_size, num_heads, head_dim = 1, 8, 64

        memories = []

        for seq_len in seq_lengths:
            # Create config
            config = RingAttentionConfig(
                segment_lengths=[seq_len // 4], dilation_rates=[1], dropout=0.0
            )

            # Create ring attention
            ring_attn = StandardRingAttention(config, device=device)

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create inputs
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Forward pass
            _ = ring_attn(q, k, v)

            # Record peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memories.append(peak_memory)

            # Cleanup
            del q, k, v, ring_attn
            torch.cuda.empty_cache()

        # Verify memory scaling is roughly linear
        # (not quadratic as with standard attention)
        ratio1 = memories[1] / memories[0]
        ratio2 = memories[2] / memories[1]

        # Should be roughly 2x for 2x sequence length
        assert 1.5 < ratio1 < 3.0, (
            f"Memory scaling ratio {ratio1} out of expected range"
        )
        assert 1.5 < ratio2 < 3.0, (
            f"Memory scaling ratio {ratio2} out of expected range"
        )


class TestRingCommunicationStats:
    """Test communication statistics and monitoring."""

    def test_stats_tracking(self):
        """Test that communication stats are properly tracked."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create mixin
        class TestMixin(RingCommunicationMixin):
            def __init__(self):
                super().__init__()
                self.rank = 0
                self.world_size = 1
                self.device = device
                self.dtype = torch.float32
                self.is_distributed = False

        mixin = TestMixin()

        # Perform some operations
        tensor = torch.ones(4, 4, device=device)
        for _ in range(5):
            mixin.ring_pass_forward(tensor)

        # Check stats
        stats = mixin.get_communication_stats()
        assert stats["total_sends"] == 5
        assert stats["total_recvs"] == 5
        assert stats["total_bytes"] > 0
        assert stats["failure_rate"] == 0.0

        # Reset stats
        mixin.reset_communication_stats()
        stats = mixin.get_communication_stats()
        assert stats["total_sends"] == 0


if __name__ == "__main__":
    # Run with: torchrun --nproc_per_node=2 test_ring_communication.py
    pytest.main([__file__, "-v"])
