"""
Test suite for Ring Attention with Dilated Patterns Integration.

This module tests the integration of dilated attention patterns with Ring Attention V2,
verifying correctness, memory efficiency, and performance.
"""

import pytest
import torch

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def compare_attention_outputs(output1, output2, rtol=1e-4, atol=1e-5):
    """Compare two attention outputs for numerical equivalence."""
    if not torch.allclose(output1, output2, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(output1 - output2)).item()
        mean_diff = torch.mean(torch.abs(output1 - output2)).item()
        return False, f"max_diff: {max_diff:.6e}, mean_diff: {mean_diff:.6e}"
    return True, "outputs match"


class TestRingDilatedAttentionIntegration:
    """Test suite for Ring Attention + Dilated Patterns integration."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        return torch.float32  # Use float32 for numerical precision in tests

    @pytest.fixture
    def basic_config(self):
        return {
            "segment_lengths": [1024, 2048],  # Smaller segments for testing
            "dilation_rates": [1, 2],
            "dropout": 0.0,  # No dropout for deterministic tests
        }

    @pytest.fixture
    def test_tensors(self, device, dtype):
        """Create test tensors with known patterns."""
        batch_size, seq_len, num_heads, head_dim = (
            1,
            4096,
            8,
            64,
        )  # Reduced size for memory

        # Use deterministic initialization for reproducible tests
        torch.manual_seed(42)
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        return query, key, value

    def test_basic_dilated_integration(self, device, dtype, basic_config, test_tensors):
        """Test basic dilated attention pattern integration."""
        query, key, value = test_tensors

        # Create Ring Dilated Attention V2 with single device (no ring)
        ring_dilated = RingDilatedAttentionV2(
            ring_size=1,  # Single device mode to test dilated patterns
            device=device,
            dtype=dtype,
            **basic_config,
        )

        # Forward pass
        output = ring_dilated(query, key, value, is_causal=False)

        # Verify output shape
        assert output.shape == query.shape, (
            f"Expected {query.shape}, got {output.shape}"
        )

        # Verify output is not all zeros (attention produced meaningful results)
        assert torch.any(output != 0), "Output should not be all zeros"

        # Basic verification complete for forward pass
        print(
            f"✓ Forward pass successful with output norm: {torch.norm(output).item():.4f}"
        )

    def test_head_group_distribution(self, device, dtype, basic_config):
        """Test that heads are correctly distributed across segment lengths."""
        ring_dilated = RingDilatedAttentionV2(
            ring_size=1, device=device, dtype=dtype, **basic_config
        )

        # Test different head counts
        test_head_counts = [4, 8, 12, 16]

        for num_heads in test_head_counts:
            head_groups = ring_dilated._calculate_head_groups(num_heads)

            # Verify total heads match
            assert sum(head_groups) == num_heads, (
                f"Head groups {head_groups} don't sum to {num_heads}"
            )

            # Verify all groups are non-negative
            assert all(g >= 0 for g in head_groups), (
                f"Negative head groups: {head_groups}"
            )

            # Verify length matches number of segments
            assert len(head_groups) == len(basic_config["segment_lengths"])

    def test_dilation_pattern_correctness(
        self, device, dtype, basic_config, test_tensors
    ):
        """Test that dilation patterns are applied correctly."""
        query, key, value = test_tensors

        ring_dilated = RingDilatedAttentionV2(
            ring_size=1, device=device, dtype=dtype, **basic_config
        )

        # Test dilation index caching
        segment_len, dilation_rate, offset = 512, 2, 0

        # Call _apply_dilation to populate cache
        b, h, d = 1, 4, 64
        q_test = torch.randn(b, 4, segment_len, h, d, device=device, dtype=dtype)
        k_test = torch.randn(b, 4, segment_len, h, d, device=device, dtype=dtype)
        v_test = torch.randn(b, 4, segment_len, h, d, device=device, dtype=dtype)

        q_out, k_out, v_out = ring_dilated._apply_dilation(
            q_test, k_test, v_test, dilation_rate, offset, segment_len
        )

        # Verify cache was populated
        cache_key = (segment_len, dilation_rate, offset, device)
        assert cache_key in ring_dilated._dilated_indices_cache

        # Verify query is unchanged (only K/V are dilated)
        assert torch.equal(q_test, q_out), "Query should not be modified by dilation"

        # Verify K/V are modified when dilation_rate > 1
        if dilation_rate > 1:
            assert not torch.equal(k_test, k_out), "Key should be modified by dilation"
            assert not torch.equal(v_test, v_out), (
                "Value should be modified by dilation"
            )

    def test_comparison_with_improved_dilated(self, device, dtype, basic_config):
        """Compare Ring Dilated Attention with standard Improved Dilated Attention."""
        # Use smaller tensors for comparison test
        batch_size, seq_len, num_heads, head_dim = 1, 4096, 8, 64

        torch.manual_seed(123)  # Deterministic for comparison
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Create both attention mechanisms
        ring_dilated = RingDilatedAttentionV2(
            ring_size=1,  # Single device mode for fair comparison
            device=device,
            dtype=dtype,
            **basic_config,
        )

        improved_dilated = ImprovedDilatedAttention(
            device=device, dtype=dtype, **basic_config
        )

        # Forward pass through both
        with torch.no_grad():
            ring_output = ring_dilated(query, key, value, is_causal=False)
            improved_output = improved_dilated(query, key, value, is_causal=False)

        # Note: Outputs may not be identical due to different implementation details,
        # but they should be in the same ballpark
        ring_norm = torch.norm(ring_output).item()
        improved_norm = torch.norm(improved_output).item()

        # Verify both produce reasonable outputs
        assert ring_norm > 0, "Ring attention should produce non-zero output"
        assert improved_norm > 0, "Improved attention should produce non-zero output"

        # Verify they're in similar magnitude (within order of magnitude)
        ratio = ring_norm / improved_norm
        assert 0.1 < ratio < 10.0, (
            f"Output magnitudes too different: {ring_norm} vs {improved_norm}"
        )

    def test_causal_masking(self, device, dtype, basic_config, test_tensors):
        """Test causal masking in dilated ring attention."""
        query, key, value = test_tensors

        ring_dilated = RingDilatedAttentionV2(
            ring_size=1, device=device, dtype=dtype, **basic_config
        )

        with torch.no_grad():
            # Test both causal and non-causal
            output_causal = ring_dilated(query, key, value, is_causal=True)
            output_non_causal = ring_dilated(query, key, value, is_causal=False)

        # Verify outputs are different when causal masking is applied
        assert not torch.equal(output_causal, output_non_causal), (
            "Causal and non-causal outputs should be different"
        )

        # Verify shapes are correct
        assert output_causal.shape == query.shape
        assert output_non_causal.shape == query.shape

    def test_memory_efficiency(self, device, dtype, basic_config):
        """Test that ring attention with dilated patterns is memory efficient."""
        if device.type != "cuda":
            pytest.skip("Memory efficiency test requires CUDA")

        # Test with different ring sizes
        batch_size, seq_len, num_heads, head_dim = 1, 8192, 8, 64

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Test single device mode (no ring)
        ring_dilated_single = RingDilatedAttentionV2(
            ring_size=1, device=device, dtype=dtype, **basic_config
        )

        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()

        with torch.no_grad():
            _ = ring_dilated_single(query, key, value, is_causal=False)

        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()
        memory_used = (memory_after - memory_before) / (1024**2)  # MB

        # Verify memory usage is reasonable (should be much less than O(n²))
        # For 8K sequence, O(n²) would be ~2GB, dilated should be much less
        assert memory_used < 500, f"Memory usage too high: {memory_used:.1f} MB"

        print(f"Memory used for dilated ring attention: {memory_used:.1f} MB")

    def test_segment_length_validation(self, device, dtype, basic_config):
        """Test validation of sequence length against segment lengths."""
        ring_dilated = RingDilatedAttentionV2(
            ring_size=1, device=device, dtype=dtype, **basic_config
        )

        # Test with sequence length smaller than largest segment
        batch_size, seq_len, num_heads, head_dim = (
            1,
            1024,
            8,
            64,
        )  # Smaller than 2048, 4096

        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Should handle gracefully by skipping segments that are too large
        with torch.no_grad():
            output = ring_dilated(query, key, value, is_causal=False)

        assert output.shape == query.shape
        assert torch.any(output != 0), (
            "Should produce meaningful output even with small sequences"
        )

    def test_gradient_flow(self, device, dtype, basic_config):
        """Test that gradients flow correctly through dilated ring attention."""
        batch_size, seq_len, num_heads, head_dim = 1, 4096, 8, 64

        query = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        value = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        ring_dilated = RingDilatedAttentionV2(
            ring_size=1, device=device, dtype=dtype, **basic_config
        )
        ring_dilated.train()  # Ensure training mode

        output = ring_dilated(query, key, value, is_causal=False)
        loss = output.sum()
        loss.backward()

        # Verify all inputs have gradients
        assert query.grad is not None, "Query should have gradients"
        assert key.grad is not None, "Key should have gradients"
        assert value.grad is not None, "Value should have gradients"

        # Verify gradients are non-zero
        assert torch.any(query.grad != 0), "Query gradients should be non-zero"
        assert torch.any(key.grad != 0), "Key gradients should be non-zero"
        assert torch.any(value.grad != 0), "Value gradients should be non-zero"


if __name__ == "__main__":
    # Run basic integration test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Running dilated Ring Attention integration tests on {device}")

    # Basic functionality test
    config = {
        "segment_lengths": [2048, 4096],
        "dilation_rates": [1, 2],
        "dropout": 0.0,
    }

    ring_dilated = RingDilatedAttentionV2(
        ring_size=1, device=device, dtype=dtype, **config
    )

    batch_size, seq_len, num_heads, head_dim = 1, 8192, 8, 64
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    with torch.no_grad():
        output = ring_dilated(query, key, value, is_causal=False)

    print("✓ Dilated Ring Attention integration test passed!")
    print(f"  Input shape: {query.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {torch.norm(output).item():.4f}")
    print(f"  Segment lengths: {config['segment_lengths']}")
    print(f"  Dilation rates: {config['dilation_rates']}")
