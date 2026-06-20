#!/usr/bin/env python3
"""
Test the simplified UnifiedHilbertAttention implementation.
"""

import torch
import sys

# Add project to path
sys.path.insert(
    0, "/home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch/src"
)

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def test_basic_functionality():
    """Test basic forward pass functionality."""
    print("Testing basic functionality...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test configurations
    configs = [
        # (batch_size, seq_len, hidden_dim, num_heads, segment_size, dilation_rate)
        (2, 512, 768, 12, 128, 1),  # Standard attention
        (1, 1024, 512, 8, 256, 2),  # Dilated with rate 2
        (2, 2048, 1024, 16, 512, 4),  # Dilated with rate 4
        (1, 256, 256, 4, 64, 1),  # Small model
    ]

    for B, N, D, H, seg_size, dil_rate in configs:
        print(
            f"\nTesting B={B}, N={N}, D={D}, H={H}, seg_size={seg_size}, dil_rate={dil_rate}"
        )

        # Create module
        attn = UnifiedHilbertAttention(
            hidden_dim=D,
            num_heads=H,
            segment_size=seg_size,
            dilation_rate=dil_rate,
            dropout=0.1,
        ).to(device)

        # Test input
        x = torch.randn(B, N, D, device=device)

        # Forward pass
        with torch.no_grad():
            # Test with Hilbert ordering
            out_hilbert = attn(x, use_hilbert=True)
            assert out_hilbert.shape == x.shape

            # Test without Hilbert ordering
            out_no_hilbert = attn(x, use_hilbert=False)
            assert out_no_hilbert.shape == x.shape

            # Test with causal masking
            out_causal = attn(x, use_hilbert=True, is_causal=True)
            assert out_causal.shape == x.shape

        print("✓ Forward pass successful")

        # Test gradient flow
        x.requires_grad = True
        out = attn(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert attn.qkv_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None

        print("✓ Gradient flow successful")


def test_cache_management():
    """Test cache management functionality."""
    print("\n\nTesting cache management...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create module with small cache
    attn = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        cache_size=4,  # Small cache for testing
        cache_memory_mb=1.0,
    ).to(device)

    # Process multiple sequence lengths
    seq_lengths = [128, 256, 384, 512, 640, 768]

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, 256, device=device)
        with torch.no_grad():
            _ = attn(x)

    # Check cache stats
    stats = attn.get_cache_stats()
    print(f"Cache stats: {stats}")

    assert stats["size"] <= 4, f"Cache exceeded size limit: {stats['size']}"
    assert stats["memory_usage_mb"] <= 1.0, (
        f"Cache exceeded memory limit: {stats['memory_usage_mb']}"
    )

    # Clear cache
    attn.clear_cache()
    stats_after = attn.get_cache_stats()
    assert stats_after["size"] == 0

    print("✓ Cache management successful")


def test_sparse_attention():
    """Test sparse attention with different dilation rates."""
    print("\n\nTesting sparse attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different dilation rates
    dilation_rates = [1, 2, 4, 8]

    for dil_rate in dilation_rates:
        print(f"\nTesting dilation_rate={dil_rate}")

        attn = UnifiedHilbertAttention(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=dil_rate,
        ).to(device)

        # Test input
        x = torch.randn(2, 512, 512, device=device)

        # Forward pass
        with torch.no_grad():
            out = attn(x)
            assert out.shape == x.shape

        # Check that output is different for different dilation rates
        if dil_rate > 1:
            attn_standard = UnifiedHilbertAttention(
                hidden_dim=512,
                num_heads=8,
                segment_size=128,
                dilation_rate=1,
            ).to(device)

            # Copy weights
            attn_standard.load_state_dict(attn.state_dict())

            with torch.no_grad():
                out_standard = attn_standard(x)
                # Outputs should be different due to sparse pattern
                assert not torch.allclose(out, out_standard, atol=1e-4)

        print(f"✓ Dilation rate {dil_rate} working correctly")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n\nTesting edge cases...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test non-divisible sequence lengths
    attn = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        segment_size=64,
    ).to(device)

    # Test various sequence lengths that aren't multiples of segment_size
    seq_lengths = [63, 65, 127, 129, 255, 257]

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, 256, device=device)
        with torch.no_grad():
            out = attn(x)
            assert out.shape == x.shape, f"Failed for seq_len={seq_len}"

    print("✓ Edge cases handled correctly")

    # Test very small sequences
    for seq_len in [1, 2, 4, 8, 16]:
        x = torch.randn(1, seq_len, 256, device=device)
        with torch.no_grad():
            out = attn(x)
            assert out.shape == x.shape

    print("✓ Small sequences handled correctly")


def test_hilbert_mapping():
    """Test Hilbert mapping generation."""
    print("\n\nTesting Hilbert mapping...")

    # Test various sequence lengths
    seq_lengths = [16, 32, 64, 128, 256, 512, 1024]

    for seq_len in seq_lengths:
        mapping = UnifiedHilbertAttention._create_hilbert_mapping(seq_len)

        # Check properties
        assert mapping.shape == (seq_len,)
        assert mapping.min() == 0
        assert mapping.max() == seq_len - 1
        assert len(torch.unique(mapping)) == seq_len  # All indices present

        print(f"✓ Hilbert mapping for seq_len={seq_len} valid")


def compare_performance():
    """Compare performance of simplified vs original implementation."""
    print("\n\nComparing performance...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("Skipping performance comparison (requires CUDA)")
        return

    import time

    # Configuration
    batch_size = 4
    seq_len = 2048
    hidden_dim = 768
    num_heads = 12

    # Create module
    attn = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=256,
        dilation_rate=4,
    ).to(device)

    # Warmup
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = attn(x)

    # Time forward pass
    torch.cuda.synchronize()
    start = time.time()

    num_iterations = 20
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = attn(x)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_time = elapsed / num_iterations
    print(f"Average forward pass time: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")


if __name__ == "__main__":
    print("Testing Simplified UnifiedHilbertAttention Implementation")
    print("=" * 60)

    test_basic_functionality()
    test_cache_management()
    test_sparse_attention()
    test_edge_cases()
    test_hilbert_mapping()
    compare_performance()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
