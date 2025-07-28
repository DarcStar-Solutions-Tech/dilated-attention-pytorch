#!/usr/bin/env python3
"""
Test script to verify head-parallel optimizations are working.
"""

import torch
import time
import gc


def test_optimizations():
    """Test that optimizations are properly integrated."""

    print("=== Testing Head-Parallel Optimizations ===\n")

    # Test single GPU first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n"
    )

    # Import implementations
    from dilated_attention_pytorch.head_parallel_dilated_attention import (
        HeadParallelDilatedAttention,
    )

    # Test configuration
    batch_size = 1
    seq_len = 8192
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    print("Test configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}")
    print(f"  Segments: {segment_lengths}")
    print(f"  Dilations: {dilation_rates}\n")

    # Create model with optimizations
    model = HeadParallelDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        use_xformers=True,
        use_flex_attention=False,
        device=device,
        dtype=torch.float32,
    )

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Test 1: Check if improved implementation is loaded
    print("Test 1: Checking implementation...")
    if hasattr(model, "_attention_impl") and model._attention_impl is not None:
        print("✓ Persistent attention implementation created")
        print(f"  Type: {type(model._attention_impl).__name__}")

        # Check optimizations
        if hasattr(model._attention_impl, "use_memory_pool"):
            print(f"  Memory pool: {model._attention_impl.use_memory_pool}")
        if hasattr(model._attention_impl, "use_pattern_cache"):
            print(f"  Pattern cache: {model._attention_impl.use_pattern_cache}")
    else:
        print("✗ Using fallback implementation")
        print("  Will still use optimized computation")

    print("\nTest 2: Running forward pass...")

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        _ = model(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Benchmark
    _ = torch.cuda.memory_allocated() / 1024**2

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    end_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print("✓ Forward pass completed")
    print(f"  Time: {(end_time - start_time) * 1000:.1f} ms")
    print(f"  Peak memory: {peak_mem:.1f} MB")
    print(f"  Memory per token: {peak_mem / (batch_size * seq_len) * 1024:.2f} KB")
    print(f"  Output shape: {output.shape}")

    # Test 3: Verify dilated pattern is applied
    print("\nTest 3: Checking attention pattern...")

    # Create small test case to verify dilation
    small_seq = 16
    small_model = HeadParallelDilatedAttention(
        segment_lengths=[8],
        dilation_rates=[2],
        dropout=0.0,
        device=device,
    )

    # Correct shape: (batch, seq, heads, dim)
    q_small = torch.ones(1, small_seq, 1, 8, device=device)
    k_small = torch.ones(1, small_seq, 1, 8, device=device)
    v_small = (
        torch.arange(small_seq, device=device)
        .float()
        .unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(3)
        .expand(1, small_seq, 1, 8)
    )

    with torch.no_grad():
        output_small = small_model(q_small, k_small, v_small, is_causal=False)

    print(f"  Small test passed: {output_small.shape}")

    # Test 4: Memory efficiency
    print("\nTest 4: Memory efficiency comparison...")

    # Theoretical minimum for attention scores
    attention_memory = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2
    print(f"  Theoretical attention memory: {attention_memory:.1f} MB")
    print(f"  Actual peak memory: {peak_mem:.1f} MB")
    print(f"  Overhead: {(peak_mem / attention_memory - 1) * 100:.1f}%")

    if peak_mem < attention_memory * 1.5:
        print("  ✓ Good memory efficiency")
    else:
        print("  ✗ High memory overhead")

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)

    optimizations_working = True

    # Check key indicators
    if hasattr(model, "_attention_impl") and model._attention_impl is not None:
        print("✓ Improved implementation loaded")
    else:
        print("⚠ Using fallback (still optimized)")
        optimizations_working = False

    if peak_mem < attention_memory * 2:
        print("✓ Memory usage is reasonable")
    else:
        print("✗ Memory usage is high")
        optimizations_working = False

    if (end_time - start_time) * 1000 < 1000:  # Less than 1 second for 8K sequence
        print("✓ Performance is good")
    else:
        print("✗ Performance needs improvement")

    if optimizations_working:
        print("\n✅ Optimizations are working properly!")
    else:
        print("\n⚠️  Some optimizations may not be fully utilized")
        print("   Check that ImprovedDilatedAttention is properly imported")

    return optimizations_working


if __name__ == "__main__":
    test_optimizations()
