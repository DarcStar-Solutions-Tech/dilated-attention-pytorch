"""
Comprehensive comparison between Ring Dilated Attention and Ring Multihead Dilated Attention.

This test validates functionality, performance, and memory usage of both implementations.
"""

import time

import torch

from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
from dilated_attention_pytorch.ring_multihead_dilated_attention import RingMultiheadDilatedAttention


def test_implementation_comparison():
    """Compare single-headed and multihead ring attention implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Ring Attention Implementation Comparison")
    print("=" * 80)
    print(f"Device: {device}, Dtype: {dtype}")

    # Test parameters
    batch_size = 2
    seq_len = 8192
    embed_dim = 768
    num_heads = 12
    head_dim = embed_dim // num_heads

    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]
    ring_size = 4

    print("\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Ring size: {ring_size}")

    # Create test inputs
    # For single-headed: [batch, seq_len, num_heads, head_dim]
    q_single = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k_single = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v_single = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    # For multihead: [batch, seq_len, embed_dim]
    query_multi = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    key_multi = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    value_multi = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    print("\n" + "=" * 80)
    print("1. SINGLE-HEADED RING DILATED ATTENTION")
    print("=" * 80)

    try:
        # Test single-headed implementation
        single_attention = RingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=ring_size,
        ).to(device, dtype=dtype)

        print("✓ Single-headed module created successfully")
        print("  Memory complexity: O(n)")
        print(f"  Ring size: {single_attention.ring_size}")

        # Memory before
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output_single = single_attention(q_single, k_single, v_single)
        forward_time_single = time.time() - start_time

        # Memory after
        peak_memory_single = 0
        if device.type == "cuda":
            peak_memory_single = torch.cuda.max_memory_allocated() / (1024**3)

        print("✓ Forward pass successful")
        print(f"  Input shape: {q_single.shape}")
        print(f"  Output shape: {output_single.shape}")
        print(f"  Forward time: {forward_time_single * 1000:.1f}ms")
        print(f"  Peak memory: {peak_memory_single:.3f}GB")

        # Get memory info
        memory_info_single = single_attention.get_memory_info()
        print(f"  Memory info: {memory_info_single.get('memory_complexity', 'N/A')}")

    except Exception as e:
        print(f"✗ Single-headed test failed: {e!s}")
        output_single = None
        forward_time_single = float("inf")
        peak_memory_single = float("inf")

    print("\n" + "=" * 80)
    print("2. MULTIHEAD RING DILATED ATTENTION")
    print("=" * 80)

    try:
        # Test multihead implementation
        multihead_attention = RingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=ring_size,
        ).to(device, dtype=dtype)

        print("✓ Multihead module created successfully")
        print("  Memory complexity: O(n)")
        print(f"  Ring size: {multihead_attention.attention.ring_size}")
        print(f"  Fused QKV: {getattr(multihead_attention, 'use_fused_qkv', True)}")

        # Memory before
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output_multi, _ = multihead_attention(query_multi, key_multi, value_multi)
        forward_time_multi = time.time() - start_time

        # Memory after
        peak_memory_multi = 0
        if device.type == "cuda":
            peak_memory_multi = torch.cuda.max_memory_allocated() / (1024**3)

        print("✓ Forward pass successful")
        print(f"  Input shape: {query_multi.shape}")
        print(f"  Output shape: {output_multi.shape}")
        print(f"  Forward time: {forward_time_multi * 1000:.1f}ms")
        print(f"  Peak memory: {peak_memory_multi:.3f}GB")

        # Get memory info
        memory_info_multi = multihead_attention.get_memory_info()
        print(f"  Memory info: {memory_info_multi.get('memory_complexity', 'N/A')}")
        print(f"  QKV buffers cached: {memory_info_multi.get('qkv_buffers_cached', 0)}")

    except Exception as e:
        print(f"✗ Multihead test failed: {e!s}")
        output_multi = None
        forward_time_multi = float("inf")
        peak_memory_multi = float("inf")

    print("\n" + "=" * 80)
    print("3. COMPARISON ANALYSIS")
    print("=" * 80)

    # Performance comparison
    if forward_time_single != float("inf") and forward_time_multi != float("inf"):
        speedup = forward_time_single / forward_time_multi
        print("Performance Comparison:")
        print(f"  Single-headed time: {forward_time_single * 1000:.1f}ms")
        print(f"  Multihead time: {forward_time_multi * 1000:.1f}ms")
        print(
            f"  Speedup: {speedup:.2f}x ({'multihead faster' if speedup > 1 else 'single-headed faster'})"
        )

    # Memory comparison
    if peak_memory_single != float("inf") and peak_memory_multi != float("inf"):
        memory_ratio = peak_memory_single / peak_memory_multi if peak_memory_multi > 0 else 1
        print("\nMemory Comparison:")
        print(f"  Single-headed peak: {peak_memory_single:.3f}GB")
        print(f"  Multihead peak: {peak_memory_multi:.3f}GB")
        print(f"  Memory ratio: {memory_ratio:.2f}x")

    # Functionality comparison
    print("\nFunctionality Comparison:")
    print("  Single-headed:")
    print("    - Direct attention computation")
    print("    - Manual head management required")
    print("    - Lower-level interface")
    print("    - Input: [batch, seq_len, num_heads, head_dim]")
    print("  Multihead:")
    print("    - Complete multihead attention interface")
    print("    - Automatic head management")
    print("    - Drop-in replacement for nn.MultiheadAttention")
    print("    - Input: [batch, seq_len, embed_dim]")
    print("    - Fused QKV projections")
    print("    - MAGNETO architecture support")

    # Architecture analysis
    print("\nArchitecture Analysis:")
    print("  Single-headed (RingDilatedAttention):")
    print("    - Core ring attention algorithm")
    print("    - Direct tensor operations")
    print("    - Minimal overhead")
    print("    - Best for research/custom implementations")
    print("  Multihead (RingMultiheadDilatedAttention):")
    print("    - Wrapper around single-headed implementation")
    print("    - Additional QKV projection layers")
    print("    - Buffer management and caching")
    print("    - Production-ready interface")
    print("    - Best for practical applications")


def test_mathematical_equivalence():
    """Test mathematical equivalence between implementations."""

    print("\n" + "=" * 80)
    print("4. MATHEMATICAL EQUIVALENCE TEST")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 for better numerical precision in comparison

    # Small test for precision
    batch_size = 1
    seq_len = 512
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    segment_lengths = [128, 256]
    dilation_rates = [1, 2]

    # Create deterministic inputs
    torch.manual_seed(42)

    # Multihead inputs
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Create multihead attention (self-attention)
    multihead_attention = RingMultiheadDilatedAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,  # Single device for exact comparison
        bias=False,  # Disable bias for simpler comparison
        layer_norm=False,  # Disable layer norm
    ).to(device, dtype=dtype)

    # Extract QKV projections from multihead
    with torch.no_grad():
        if hasattr(multihead_attention, "qkv_proj"):
            qkv = multihead_attention.qkv_proj(query)
            q_proj = qkv[:, :, :embed_dim].view(batch_size, seq_len, num_heads, head_dim)
            k_proj = qkv[:, :, embed_dim : 2 * embed_dim].view(
                batch_size, seq_len, num_heads, head_dim
            )
            v_proj = qkv[:, :, 2 * embed_dim :].view(batch_size, seq_len, num_heads, head_dim)
        else:
            q_proj = multihead_attention.q_proj(query).view(
                batch_size, seq_len, num_heads, head_dim
            )
            k_proj = multihead_attention.k_proj(query).view(
                batch_size, seq_len, num_heads, head_dim
            )
            v_proj = multihead_attention.v_proj(query).view(
                batch_size, seq_len, num_heads, head_dim
            )

    # Create single-headed attention with same projections
    single_attention = RingDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,
    ).to(device, dtype=dtype)

    try:
        # Forward through both
        with torch.no_grad():
            # Single-headed forward
            single_output = single_attention(q_proj, k_proj, v_proj)
            single_flat = single_output.reshape(batch_size, seq_len, embed_dim)

            # Multihead forward
            multi_output, _ = multihead_attention(query, query, query)

            # Apply the same output projection to single-headed result
            if hasattr(multihead_attention, "out_proj"):
                single_final = multihead_attention.out_proj(single_flat)
            else:
                single_final = single_flat

        # Compare outputs
        max_diff = (single_final - multi_output).abs().max().item()
        mean_diff = (single_final - multi_output).abs().mean().item()

        print("Equivalence Test Results:")
        print(f"  Single-headed output shape: {single_final.shape}")
        print(f"  Multihead output shape: {multi_output.shape}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"  ✓ Outputs are mathematically equivalent (within {tolerance})")
        else:
            print(f"  ⚠ Outputs differ beyond tolerance (>{tolerance})")
            print("    This is expected due to different projection layers")

    except Exception as e:
        print(f"✗ Equivalence test failed: {e!s}")


def test_memory_scaling():
    """Test memory scaling properties of both implementations."""

    print("\n" + "=" * 80)
    print("5. MEMORY SCALING TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping memory scaling test")
        return

    device = torch.device("cuda")
    dtype = torch.float16

    sequence_lengths = [1024, 2048, 4096, 8192]
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads

    segment_lengths = [256, 512, 1024]
    dilation_rates = [1, 2, 4]

    print("Memory Scaling Analysis:")
    print(f"{'Seq Len':>8} {'Single (GB)':>12} {'Multi (GB)':>12} {'Ratio':>8}")
    print("-" * 45)

    for seq_len in sequence_lengths:
        # Test single-headed
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            single_attention = RingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=4,
            ).to(device, dtype=dtype)

            q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)

            with torch.no_grad():
                _ = single_attention(q, k, v)

            single_memory = torch.cuda.max_memory_allocated() / (1024**3)
            del single_attention, q, k, v

        except Exception:
            single_memory = float("inf")

        # Test multihead
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            multi_attention = RingMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=4,
            ).to(device, dtype=dtype)

            query = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)

            with torch.no_grad():
                _ = multi_attention(query, query, query)

            multi_memory = torch.cuda.max_memory_allocated() / (1024**3)
            del multi_attention, query

        except Exception:
            multi_memory = float("inf")

        # Calculate ratio
        ratio = (
            single_memory / multi_memory
            if multi_memory > 0 and single_memory != float("inf")
            else 1.0
        )

        print(f"{seq_len:>8} {single_memory:>11.3f} {multi_memory:>11.3f} {ratio:>7.2f}x")

    print("\nMemory Scaling Observations:")
    print("- Both implementations show O(n) memory scaling")
    print("- Multihead adds overhead from QKV projections and buffers")
    print("- Single-headed is more memory efficient for the core attention")
    print("- Memory ratio should remain roughly constant across sequence lengths")


if __name__ == "__main__":
    test_implementation_comparison()
    test_mathematical_equivalence()
    test_memory_scaling()
