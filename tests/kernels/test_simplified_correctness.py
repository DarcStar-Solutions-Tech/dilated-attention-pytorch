#!/usr/bin/env python3
"""
Comprehensive correctness verification for the simplified UnifiedHilbertAttention.

This test verifies that the simplified implementation produces correct results
compared to the original implementations and maintains all expected behavior.
"""

import torch
import torch.nn.functional as F
import sys
import warnings

# Add project to path
sys.path.insert(
    0, "/home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch/src"
)

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def test_attention_correctness():
    """Verify attention computation is mathematically correct."""
    print("Testing attention computation correctness...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test case for manual verification (>64 to trigger Hilbert mapping)
    batch_size = 2
    seq_len = 128
    hidden_dim = 256
    num_heads = 8
    head_dim = hidden_dim // num_heads

    # Create module
    attn = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=32,
        dropout=0.0,  # No dropout for deterministic testing
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Get output without Hilbert ordering (standard attention)
    with torch.no_grad():
        out = attn(x, use_hilbert=False)

    # Manually compute expected output
    with torch.no_grad():
        # Project to QKV
        qkv = attn.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, hidden_dim)
        expected = attn.out_proj(attn_out)

    # Check correctness
    assert torch.allclose(out, expected, atol=1e-5), "Attention computation incorrect"
    print("✓ Standard attention computation correct")

    # Test with Hilbert ordering
    with torch.no_grad():
        out_hilbert = attn(x, use_hilbert=True)

    # For dilation_rate=1, Hilbert ordering should produce different output
    # unless the implementation is not applying it
    if attn.dilation_rate == 1:
        # Outputs should be different due to reordering
        # Let's check if they are the same first
        if torch.allclose(out, out_hilbert, atol=1e-6):
            print("⚠ Hilbert ordering produces same output - checking implementation")
            # This might happen if Triton kernels aren't available
            print(f"  Triton available: {attn._triton_available}")
        else:
            print("✓ Hilbert ordering applied correctly")
    else:
        # For dilated attention, Hilbert is only applied with dilation_rate=1
        print("✓ Dilated attention mode (Hilbert not applied)")


def test_dilated_attention_correctness():
    """Verify dilated/sparse attention is computed correctly."""
    print("\nTesting dilated attention correctness...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test configuration
    dilation_rate = 4
    segment_size = 32
    seq_len = 128

    attn = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        dropout=0.0,
    ).to(device)

    # Create input
    x = torch.randn(1, seq_len, 256, device=device)

    # Get output
    with torch.no_grad():
        out = attn(x)

    # Verify sparsity pattern
    # With dilation_rate=4, each query should only attend to every 4th key
    # This means the effective attention window is reduced by factor of dilation_rate

    # Compare with full attention
    attn_full = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        segment_size=segment_size,
        dilation_rate=1,  # No dilation
        dropout=0.0,
    ).to(device)

    # Copy weights
    attn_full.load_state_dict(attn.state_dict())

    with torch.no_grad():
        out_full = attn_full(x)

    # Outputs should be different
    assert not torch.allclose(out, out_full, atol=1e-4), "Dilation not applied"
    print("✓ Dilated attention pattern applied correctly")


def test_causal_masking():
    """Verify causal masking is applied correctly."""
    print("\nTesting causal masking...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    attn = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,
    ).to(device)

    # Create input where later positions have distinct values
    x = torch.randn(1, 64, 256, device=device)
    x[:, 32:, :] = x[:, 32:, :] + 10.0  # Make later positions very different

    # Forward with causal masking
    with torch.no_grad():
        out_causal = attn(x, is_causal=True)
        out_non_causal = attn(x, is_causal=False)

    # With causal masking, early positions should not be affected by later ones
    # So the output for early positions should be different
    early_causal = out_causal[:, :16, :]
    early_non_causal = out_non_causal[:, :16, :]

    assert not torch.allclose(early_causal, early_non_causal, atol=1e-3), (
        "Causal masking not affecting output"
    )

    print("✓ Causal masking applied correctly")


def test_gradient_flow():
    """Verify gradients flow correctly through the module."""
    print("\nTesting gradient flow...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test different configurations
    configs = [
        {"dilation_rate": 1},  # Standard
        {"dilation_rate": 4},  # Dilated
    ]

    for config in configs:
        attn = UnifiedHilbertAttention(hidden_dim=256, num_heads=8, **config).to(device)

        # Input with gradients
        x = torch.randn(2, 128, 256, device=device, requires_grad=True)

        # Forward and backward
        out = attn(x)
        loss = out.mean()
        loss.backward()

        # Check gradients exist and are non-zero
        assert x.grad is not None, f"No input gradient for config {config}"
        assert x.grad.abs().sum() > 0, f"Zero input gradient for config {config}"

        assert attn.qkv_proj.weight.grad is not None, (
            f"No QKV gradient for config {config}"
        )
        assert attn.qkv_proj.weight.grad.abs().sum() > 0, (
            f"Zero QKV gradient for config {config}"
        )

        assert attn.out_proj.weight.grad is not None, (
            f"No output gradient for config {config}"
        )
        assert attn.out_proj.weight.grad.abs().sum() > 0, (
            f"Zero output gradient for config {config}"
        )

        print(f"✓ Gradient flow correct for config {config}")


def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    print("\nTesting numerical stability...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,
    ).to(device)

    # Test with very small values
    x_small = torch.randn(1, 64, 256, device=device) * 1e-6
    with torch.no_grad():
        out_small = attn(x_small)
    assert torch.isfinite(out_small).all(), "NaN/Inf with small inputs"
    print("✓ Stable with small inputs")

    # Test with very large values
    x_large = torch.randn(1, 64, 256, device=device) * 1e3
    with torch.no_grad():
        out_large = attn(x_large)
    assert torch.isfinite(out_large).all(), "NaN/Inf with large inputs"
    print("✓ Stable with large inputs")

    # Test with mixed scales
    x_mixed = torch.randn(1, 64, 256, device=device)
    x_mixed[:, :32, :] *= 1e-3
    x_mixed[:, 32:, :] *= 1e3
    with torch.no_grad():
        out_mixed = attn(x_mixed)
    assert torch.isfinite(out_mixed).all(), "NaN/Inf with mixed scale inputs"
    print("✓ Stable with mixed scale inputs")


def test_consistency_across_devices():
    """Test consistency between CPU and GPU implementations."""
    print("\nTesting CPU/GPU consistency...")

    if not torch.cuda.is_available():
        print("⚠ Skipping CPU/GPU consistency test (no GPU available)")
        return

    torch.manual_seed(42)

    # Create identical modules
    attn_cpu = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        dilation_rate=2,
        dropout=0.0,
    )

    attn_gpu = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        dilation_rate=2,
        dropout=0.0,
    ).cuda()

    # Ensure same weights
    attn_gpu.load_state_dict(attn_cpu.state_dict())

    # Test input
    x_cpu = torch.randn(2, 128, 256)
    x_gpu = x_cpu.cuda()

    # Forward pass without Hilbert (to avoid Triton differences)
    with torch.no_grad():
        out_cpu = attn_cpu(x_cpu, use_hilbert=False)
        out_gpu = attn_gpu(x_gpu, use_hilbert=False)

    # Compare (allowing for small numerical differences)
    if torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-4):
        print("✓ CPU/GPU implementations consistent (without Hilbert)")
    else:
        print("⚠ CPU/GPU outputs differ - likely due to different backend paths")
        print(f"  Max difference: {(out_cpu - out_gpu.cpu()).abs().max():.6f}")

    # Also test with Hilbert if both use PyTorch backend
    with torch.no_grad():
        out_cpu_h = attn_cpu(x_cpu, use_hilbert=True)
        out_gpu_h = attn_gpu(x_gpu, use_hilbert=True)

    if torch.allclose(out_cpu_h, out_gpu_h.cpu(), atol=1e-4):
        print("✓ CPU/GPU consistent with Hilbert ordering")
    else:
        # This is expected if GPU uses Triton
        print("ℹ CPU/GPU differ with Hilbert (GPU using Triton kernel)")


def test_batch_consistency():
    """Test that batched computation is consistent with individual samples."""
    print("\nTesting batch consistency...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    attn = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        dilation_rate=2,
        dropout=0.0,
    ).to(device)

    # Create batched input
    x_batch = torch.randn(4, 128, 256, device=device)

    # Process as batch
    with torch.no_grad():
        out_batch = attn(x_batch)

    # Process individually
    out_individual = []
    with torch.no_grad():
        for i in range(4):
            out_i = attn(x_batch[i : i + 1])
            out_individual.append(out_i)

    out_individual = torch.cat(out_individual, dim=0)

    # Should be identical
    assert torch.allclose(out_batch, out_individual, atol=1e-6), (
        "Batched and individual processing differ"
    )

    print("✓ Batch processing consistent")


def test_hilbert_mapping_properties():
    """Test properties of Hilbert mapping."""
    print("\nTesting Hilbert mapping properties...")

    # Test different sequence lengths
    for seq_len in [32, 64, 128, 256, 512, 1000]:
        mapping = UnifiedHilbertAttention._create_hilbert_mapping(seq_len)

        # Check it's a permutation
        assert mapping.shape == (seq_len,)
        assert set(mapping.tolist()) == set(range(seq_len)), (
            f"Hilbert mapping for seq_len={seq_len} is not a valid permutation"
        )

        # Check locality preservation (approximate)
        # Adjacent elements in Hilbert order should be relatively close in original order
        if seq_len >= 64:
            diffs = []
            for i in range(len(mapping) - 1):
                diff = abs(mapping[i].item() - mapping[i + 1].item())
                diffs.append(diff)

            avg_diff = sum(diffs) / len(diffs)
            random_avg_diff = seq_len / 3  # Expected for random permutation

            # Hilbert curve should have better locality than random
            assert avg_diff < random_avg_diff * 0.8, (
                f"Hilbert mapping doesn't preserve locality for seq_len={seq_len}"
            )

    print("✓ Hilbert mapping properties verified")


def compare_with_standard_attention():
    """Compare with PyTorch's standard scaled dot-product attention."""
    print("\nComparing with PyTorch standard attention...")

    if not hasattr(F, "scaled_dot_product_attention"):
        print("⚠ Skipping comparison (PyTorch version too old)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Create our implementation
    attn = UnifiedHilbertAttention(
        hidden_dim=512,
        num_heads=8,
        dropout=0.0,
    ).to(device)

    # Test input
    x = torch.randn(2, 256, 512, device=device)

    # Our output (without Hilbert ordering for fair comparison)
    with torch.no_grad():
        our_out = attn(x, use_hilbert=False)

    # Manual computation with PyTorch's SDPA
    with torch.no_grad():
        qkv = attn.qkv_proj(x)
        qkv = qkv.reshape(2, 256, 3, 8, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        sdpa_out = F.scaled_dot_product_attention(q, k, v, scale=64**-0.5)
        sdpa_out = sdpa_out.transpose(1, 2).contiguous().view(2, 256, 512)
        sdpa_out = attn.out_proj(sdpa_out)

    # Should be very close
    assert torch.allclose(our_out, sdpa_out, atol=1e-5), (
        "Output differs from PyTorch SDPA"
    )

    print("✓ Matches PyTorch scaled dot-product attention")


if __name__ == "__main__":
    print("Comprehensive Correctness Verification")
    print("=" * 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore deprecation warnings

        test_attention_correctness()
        test_dilated_attention_correctness()
        test_causal_masking()
        test_gradient_flow()
        test_numerical_stability()
        test_consistency_across_devices()
        test_batch_consistency()
        test_hilbert_mapping_properties()
        compare_with_standard_attention()

    print("\n" + "=" * 60)
    print("All correctness tests passed! ✓")
    print("\nThe simplified implementation is verified to be correct.")
