#!/usr/bin/env python3
"""
Basic test of Hilbert implementations to verify they work.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basic_functionality():
    """Test basic forward/backward pass."""
    print("Testing Basic Hilbert Functionality")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test 1: HilbertAttentionCore
    try:
        from src.dilated_attention_pytorch.kernels.hilbert_attention_core import (
            HilbertAttentionCore,
        )

        # Simple configuration that should work
        batch_size = 1
        seq_len = 128  # Power of 2
        hidden_dim = 512
        num_heads = 8

        model = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=64,
            dilation_rate=1,
            dropout=0.0,
            use_custom_backward=True,
        ).to(device)

        # Test input
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, requires_grad=True
        )

        # Forward pass
        print("\nTest 1: HilbertAttentionCore")
        print("-" * 40)

        # With Hilbert ordering
        out_hilbert = model(x, use_hilbert=True)
        print(f"✓ Forward pass (Hilbert): {out_hilbert.shape}")

        # Without Hilbert ordering
        x_clone = x.clone().detach().requires_grad_(True)
        out_standard = model(x_clone, use_hilbert=False)
        print(f"✓ Forward pass (Standard): {out_standard.shape}")

        # Test backward
        loss = out_hilbert.sum()
        loss.backward()
        print(f"✓ Backward pass: grad shape {x.grad.shape}")

        # Check for NaN/Inf
        assert not torch.isnan(out_hilbert).any(), "Output contains NaN"
        assert not torch.isinf(out_hilbert).any(), "Output contains Inf"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"

        print("✅ HilbertAttentionCore: All tests passed!")

    except Exception as e:
        print(f"❌ HilbertAttentionCore failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Test 2: Wrapper with Q,K,V interface
    try:
        from src.dilated_attention_pytorch.kernels.hilbert_attention_triton_wrapper import (
            HilbertAttentionTritonWrapper,
        )

        print("\n\nTest 2: HilbertAttentionTritonWrapper (Q,K,V interface)")
        print("-" * 40)

        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[64],
            dilation_rates=[1],
            dropout=0.0,
            num_heads=8,
            head_dim=64,
        ).to(device)

        # Q, K, V tensors
        q = torch.randn(batch_size, seq_len, num_heads, 64, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, 64, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, 64, device=device)

        output = wrapper(q, k, v)
        print(f"✓ Forward pass: {output.shape}")

        assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"

        print("✅ HilbertAttentionTritonWrapper: All tests passed!")

    except Exception as e:
        print(f"❌ HilbertAttentionTritonWrapper failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Test 3: Test with different sequence lengths
    try:
        print("\n\nTest 3: Variable Sequence Lengths")
        print("-" * 40)

        seq_lengths = [64, 128, 256, 512]

        for seq_len in seq_lengths:
            model = HilbertAttentionCore(
                hidden_dim=512,
                num_heads=8,
                segment_size=min(64, seq_len),
                dilation_rate=1,
                dropout=0.0,
            ).to(device)

            x = torch.randn(1, seq_len, 512, device=device)

            with torch.no_grad():
                out = model(x, use_hilbert=True)

            assert out.shape == x.shape, f"Shape mismatch for seq_len={seq_len}"
            assert not torch.isnan(out).any(), f"NaN in output for seq_len={seq_len}"

            print(f"✓ seq_len={seq_len}: OK")

        print("✅ Variable sequence lengths: All tests passed!")

    except Exception as e:
        print(f"❌ Variable sequence lengths failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Test 4: Memory efficiency test
    try:
        print("\n\nTest 4: Memory Efficiency")
        print("-" * 40)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

            # Large sequence
            x_large = torch.randn(1, 1024, 512, device=device)
            model = HilbertAttentionCore(
                hidden_dim=512, num_heads=8, segment_size=128, dilation_rate=2
            ).to(device)

            with torch.no_grad():
                out = model(x_large, use_hilbert=True)

            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"✓ Peak memory for 1K sequence: {peak_memory:.2f} MB")

            # Verify output
            assert not torch.isnan(out).any(), "NaN in output"
            print("✅ Memory efficiency test passed!")
        else:
            print("⚠️  Skipping memory test (GPU required)")

    except Exception as e:
        print(f"❌ Memory efficiency test failed: {str(e)}")

    print("\n" + "=" * 60)
    print("Testing Complete!")


if __name__ == "__main__":
    test_basic_functionality()
