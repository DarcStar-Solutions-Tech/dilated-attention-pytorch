"""
Comprehensive tests for Ring Dilated Attention V2.

Tests include:
1. Correctness - output matches standard attention
2. Memory scaling - verifies O(n/ring_size) memory usage
3. Single vs multi-GPU operation
4. Gradient flow
"""

import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2


class TestRingAttentionV2:
    """Test suite for Ring Attention V2."""

    def test_correctness_single_gpu(self):
        """Verify output matches standard attention on single GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32  # Use float32 for precision

        seq_len = 2048
        batch_size = 2
        num_heads = 8
        head_dim = 64

        # Create test tensors
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Reference: standard attention
        scores = torch.matmul(
            q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)
        ) / (head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        expected = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2)

        # Test different ring sizes
        for ring_size in [1, 2, 4, 8]:
            if seq_len % ring_size != 0:
                continue

            ring_attn = RingDilatedAttentionV2(
                segment_lengths=[512, 1024],
                dilation_rates=[1, 2],
                ring_size=ring_size,
                device=device,
                dtype=dtype,
            )

            with torch.no_grad():
                actual = ring_attn(q, k, v)

            # Check shapes
            assert actual.shape == expected.shape

            # Check values (with some tolerance for floating point)
            max_diff = torch.max(torch.abs(actual - expected)).item()
            mean_diff = torch.mean(torch.abs(actual - expected)).item()

            print(
                f"Ring size {ring_size}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
            )

            assert max_diff < 1e-5, f"Output mismatch for ring_size={ring_size}"

    def test_memory_scaling(self):
        """Verify memory scales as O(n/ring_size)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory tests")

        device = torch.device("cuda")
        dtype = torch.float16

        # Test configuration
        seq_len = 4096
        batch_size = 1
        num_heads = 8
        head_dim = 64

        results = []

        for ring_size in [1, 2, 4, 8]:
            # Clear memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create module
            ring_attn = RingDilatedAttentionV2(
                segment_lengths=[1024],
                dilation_rates=[1],
                ring_size=ring_size,
                device=device,
                dtype=dtype,
            )

            # Create tensors
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Measure memory before forward
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward pass
            with torch.no_grad():
                output = ring_attn(q, k, v)

            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() - mem_before

            # Get theoretical estimate
            estimate = ring_attn.get_memory_estimate(
                seq_len, batch_size, num_heads, head_dim
            )

            results.append(
                {
                    "ring_size": ring_size,
                    "measured_gb": peak_memory / (1024**3),
                    "estimated_gb": estimate["total_per_device_gb"],
                    "reduction_factor": estimate["memory_reduction_factor"],
                }
            )

            print(f"\nRing size {ring_size}:")
            print(f"  Measured: {peak_memory / (1024**3):.3f} GB")
            print(f"  Estimated: {estimate['total_per_device_gb']:.3f} GB")
            print(f"  Reduction factor: {estimate['memory_reduction_factor']:.1f}x")

            # Cleanup
            del q, k, v, output
            torch.cuda.empty_cache()

        # Verify scaling
        if len(results) > 1:
            baseline = results[0]["measured_gb"]
            for r in results[1:]:
                expected_reduction = r["reduction_factor"]
                actual_reduction = baseline / r["measured_gb"]
                print(
                    f"\nRing {r['ring_size']}: Expected {expected_reduction:.1f}x reduction, got {actual_reduction:.1f}x"
                )
                # Allow some variance due to PyTorch overhead
                assert actual_reduction > expected_reduction * 0.7

    def test_gradient_flow(self):
        """Verify gradients flow correctly through ring attention."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        seq_len = 512
        batch_size = 2
        num_heads = 4
        head_dim = 32

        # Create test tensors with gradients
        q = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        # Ring attention
        ring_attn = RingDilatedAttentionV2(
            segment_lengths=[256],
            dilation_rates=[1],
            ring_size=4,
            device=device,
            dtype=dtype,
        )

        output = ring_attn(q, k, v)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        assert torch.any(q.grad != 0)
        assert torch.any(k.grad != 0)
        assert torch.any(v.grad != 0)

        print("✓ Gradients flow correctly")

    def test_comparison_with_ring_attention_correct(self):
        """Compare V2 with RingAttentionCorrectV2 implementation."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        seq_len = 1024
        batch_size = 1
        num_heads = 8
        head_dim = 64
        ring_size = 4

        # Create test tensors
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # RingAttentionCorrectV2 (reference with proper normalization)
        from dilated_attention_pytorch.ring_attention_correct_v2 import (
            RingAttentionCorrectV2,
        )

        correct_impl = RingAttentionCorrectV2(device=device, dtype=dtype)
        with torch.no_grad():
            expected = correct_impl(q, k, v, ring_size=ring_size)

        # RingDilatedAttentionV2
        v2_impl = RingDilatedAttentionV2(
            segment_lengths=[256, 512],
            dilation_rates=[1, 1],  # No dilation for fair comparison
            ring_size=ring_size,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            actual = v2_impl(q, k, v)

        # Compare outputs
        max_diff = torch.max(torch.abs(actual - expected)).item()
        mean_diff = torch.mean(torch.abs(actual - expected)).item()

        print("\nComparison with RingAttentionCorrectV2:")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        assert max_diff < 1e-5, "V2 output doesn't match RingAttentionCorrectV2"

    def test_extreme_sequence_simulation(self):
        """Test with extreme sequence lengths in simulation mode."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Test configuration for extreme sequences
        test_configs = [
            (8192, 8),  # 8K tokens, ring=8
            (32768, 32),  # 32K tokens, ring=32
            (131072, 128),  # 128K tokens, ring=128
        ]

        for seq_len, ring_size in test_configs:
            print(f"\nTesting {seq_len:,} tokens with ring_size={ring_size}")

            # Get memory estimate
            ring_attn = RingDilatedAttentionV2(
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                ring_size=ring_size,
                device=device,
                dtype=dtype,
            )

            estimate = ring_attn.get_memory_estimate(
                seq_len, batch_size=1, num_heads=8, head_dim=64
            )

            print(f"  Mode: {estimate['mode']}")
            print(f"  Memory per device: {estimate['total_per_device_gb']:.2f} GB")
            print(f"  Memory reduction: {estimate['memory_reduction_factor']:.1f}x")

            # Skip actual allocation if too large
            if estimate["total_per_device_gb"] > 6.0:
                print("  Skipping actual test (would require too much memory)")
                continue

            # Actually test with smaller sequence
            test_len = min(seq_len, 8192)
            q = torch.randn(1, test_len, 8, 64, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            with torch.no_grad():
                output = ring_attn(q, k, v)

            assert output.shape == q.shape
            print(f"  ✓ Successfully processed {test_len} tokens")


def run_all_tests():
    """Run all tests."""
    test = TestRingAttentionV2()

    print("Testing correctness...")
    test.test_correctness_single_gpu()

    print("\nTesting memory scaling...")
    test.test_memory_scaling()

    print("\nTesting gradient flow...")
    test.test_gradient_flow()

    print("\nComparing with RingAttentionCorrect...")
    test.test_comparison_with_ring_attention_correct()

    print("\nTesting extreme sequences...")
    test.test_extreme_sequence_simulation()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
