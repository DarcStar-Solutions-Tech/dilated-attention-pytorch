#!/usr/bin/env python3
"""
Verification script for Ring Attention V2Collective fixes.

This script tests:
1. Causal mask caching correctness
2. Unified attention method performance
3. O(n²) fix verification
"""

import time
import torch
import numpy as np

# Import the Ring Attention implementations
try:
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )
    from dilated_attention_pytorch.improved_dilated_attention import (
        ImprovedDilatedAttention,
    )
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    return result, end_time - start_time


def test_causal_mask_caching():
    """Test that causal mask caching works correctly."""
    print("=" * 70)
    print("Test 1: Causal Mask Caching")
    print("=" * 70)

    # Create Ring Attention instance
    ring_attn = RingDilatedAttentionV2Collective(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        dropout=0.0,
        use_pattern_cache=True,
    )

    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # First forward pass - should create cache
    print("Running first forward pass (should create cache)...")
    output1, time1 = measure_execution_time(ring_attn, q, k, v, is_causal=True)
    print(f"  Time: {time1:.4f}s")

    # Check if cache was created
    cache_created = (
        hasattr(ring_attn, "_causal_mask_cache")
        and ring_attn._causal_mask_cache is not None
    )
    print(f"  Causal mask cache created: {cache_created}")

    # Second forward pass - should use cache
    print("\nRunning second forward pass (should use cache)...")
    output2, time2 = measure_execution_time(ring_attn, q, k, v, is_causal=True)
    print(f"  Time: {time2:.4f}s")

    # Verify outputs are identical
    max_diff = torch.max(torch.abs(output1 - output2)).item()
    outputs_match = max_diff < 1e-6
    print(f"  Outputs match: {outputs_match} (max diff: {max_diff:.2e})")

    # Check speedup from caching
    speedup = time1 / time2 if time2 > 0 else float("inf")
    print(f"  Speedup from caching: {speedup:.2f}x")

    # Test cache invalidation with different sequence length
    print("\nTesting cache invalidation with different sequence length...")
    q_new = torch.randn(
        batch_size, 2048, num_heads, head_dim
    )  # Use valid sequence length
    k_new = torch.randn(batch_size, 2048, num_heads, head_dim)
    v_new = torch.randn(batch_size, 2048, num_heads, head_dim)

    output3, time3 = measure_execution_time(
        ring_attn, q_new, k_new, v_new, is_causal=True
    )
    print(f"  Time with new seq_len: {time3:.4f}s")

    return {
        "cache_created": cache_created,
        "outputs_match": outputs_match,
        "speedup": speedup,
        "max_diff": max_diff,
    }


def test_unified_attention_method():
    """Test the unified attention method produces correct outputs."""
    print("\n" + "=" * 70)
    print("Test 2: Unified Attention Method")
    print("=" * 70)

    # Create instances
    ring_attn = RingDilatedAttentionV2Collective(
        segment_lengths=[256, 512],
        dilation_rates=[1, 2],
        dropout=0.0,
    )

    baseline_attn = ImprovedDilatedAttention(
        segment_lengths=[256, 512],
        dilation_rates=[1, 2],
        dropout=0.0,
    )

    # Test parameters
    batch_size = 2
    seq_len = 512
    num_heads = 4
    head_dim = 32

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Test both causal and non-causal
    for is_causal in [False, True]:
        print(f"\nTesting is_causal={is_causal}...")

        # Ring attention output
        ring_output = ring_attn(q, k, v, is_causal=is_causal)

        # Baseline output
        baseline_output = baseline_attn(q, k, v, is_causal=is_causal)

        # Compare outputs
        max_diff = torch.max(torch.abs(ring_output - baseline_output)).item()
        outputs_match = max_diff < 1e-4

        print(f"  Outputs match baseline: {outputs_match} (max diff: {max_diff:.2e})")

    return {"causal_match": outputs_match, "max_diff": max_diff}


def test_performance_scaling():
    """Test performance scaling to verify O(n) vs O(n²) fix."""
    print("\n" + "=" * 70)
    print("Test 3: Performance Scaling (O(n) verification)")
    print("=" * 70)

    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    segment_base = 256

    results = []

    for seq_len in seq_lengths:
        print(f"\nTesting seq_len={seq_len}...")

        # Adjust segment lengths for sequence length
        segment_lengths = [segment_base, min(segment_base * 2, seq_len)]
        dilation_rates = [1, 2]

        # Create attention instance
        ring_attn = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            use_pattern_cache=True,
        )

        # Create inputs
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Warm up
        _ = ring_attn(q, k, v, is_causal=True)

        # Measure time over multiple runs
        num_runs = 5
        times = []
        for _ in range(num_runs):
            _, elapsed = measure_execution_time(ring_attn, q, k, v, is_causal=True)
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        results.append({"seq_len": seq_len, "avg_time": avg_time, "std_time": std_time})

        print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")

    # Analyze scaling
    print("\nScaling Analysis:")
    print("  Sequence Length | Time (s) | Relative Time | Expected O(n)")
    print("  " + "-" * 60)

    base_time = results[0]["avg_time"]
    base_len = results[0]["seq_len"]

    for result in results:
        relative_time = result["avg_time"] / base_time
        expected_linear = result["seq_len"] / base_len
        scaling_ratio = relative_time / expected_linear

        print(
            f"  {result['seq_len']:14d} | {result['avg_time']:8.4f} | "
            f"{relative_time:13.2f}x | {expected_linear:13.2f}x "
            f"(ratio: {scaling_ratio:.2f})"
        )

    # Check if scaling is closer to O(n) than O(n²)
    # For O(n), ratio should be close to 1.0
    # For O(n²), ratio would grow linearly with n
    scaling_ratios = []
    for i in range(1, len(results)):
        rel_time = results[i]["avg_time"] / results[0]["avg_time"]
        rel_len = results[i]["seq_len"] / results[0]["seq_len"]
        scaling_ratios.append(rel_time / rel_len)

    avg_scaling_ratio = np.mean(scaling_ratios)
    is_linear = avg_scaling_ratio < 2.0  # Allow some overhead

    print(f"\nAverage scaling ratio: {avg_scaling_ratio:.2f}")
    print(f"Scaling is approximately O(n): {is_linear}")

    return {
        "results": results,
        "avg_scaling_ratio": avg_scaling_ratio,
        "is_linear": is_linear,
    }


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n" + "=" * 70)
    print("Test 4: Memory Efficiency")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return None

    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Test parameters
    batch_size = 1
    seq_len = 8192
    num_heads = 8
    head_dim = 64

    # Create attention instance
    ring_attn = RingDilatedAttentionV2Collective(
        segment_lengths=[1024, 2048, 4096],
        dilation_rates=[1, 2, 4],
        dropout=0.0,
        use_pattern_cache=True,
    ).cuda()

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")

    # Measure memory before
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    # Forward pass
    output = ring_attn(q, k, v, is_causal=True)

    # Measure memory after
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()

    mem_used = (mem_after - mem_before) / (1024**2)  # Convert to MB

    print(f"  Sequence length: {seq_len}")
    print(f"  Memory used: {mem_used:.2f} MB")
    print(f"  Memory per token: {mem_used * 1024 / seq_len:.2f} KB")

    # Clean up
    del q, k, v, output, ring_attn
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "memory_mb": mem_used,
        "memory_per_token_kb": mem_used * 1024 / seq_len,
    }


def main():
    """Run all verification tests."""
    print("Ring Attention V2Collective Verification Suite")
    print("=" * 70)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    print(f"Running on device: {device}")

    # Run tests
    test_results = {}

    # Test 1: Causal mask caching
    test_results["causal_mask_caching"] = test_causal_mask_caching()

    # Test 2: Unified attention method
    test_results["unified_attention"] = test_unified_attention_method()

    # Test 3: Performance scaling
    test_results["performance_scaling"] = test_performance_scaling()

    # Test 4: Memory efficiency (GPU only)
    if torch.cuda.is_available():
        test_results["memory_efficiency"] = test_memory_efficiency()

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True

    # Check causal mask caching
    if (
        test_results["causal_mask_caching"]["cache_created"]
        and test_results["causal_mask_caching"]["outputs_match"]
    ):
        print("✅ Causal mask caching: PASSED")
    else:
        print("❌ Causal mask caching: FAILED")
        all_passed = False

    # Check unified attention
    if test_results["unified_attention"]["causal_match"]:
        print("✅ Unified attention method: PASSED")
    else:
        print("❌ Unified attention method: FAILED")
        all_passed = False

    # Check performance scaling
    if test_results["performance_scaling"]["is_linear"]:
        print("✅ O(n) performance scaling: PASSED")
    else:
        print("❌ O(n) performance scaling: FAILED")
        all_passed = False

    # Check memory efficiency
    if "memory_efficiency" in test_results and test_results["memory_efficiency"]:
        print(
            f"✅ Memory efficiency: {test_results['memory_efficiency']['memory_per_token_kb']:.2f} KB/token"
        )

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All Ring Attention V2Collective fixes verified successfully!")
    else:
        print("❌ Some fixes need attention. Please review the test output.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
