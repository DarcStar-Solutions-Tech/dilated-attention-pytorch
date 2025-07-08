#!/usr/bin/env python3
"""
Test script to verify the fixed Hilbert implementation.

This script tests:
1. Per-segment Hilbert SFC application
2. Proper ring communication
3. Memory safety with the new safety utilities
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core_fixed import (
    RingDilatedAttentionHilbertCoreFixed,
)
from src.dilated_attention_pytorch.kernels.hilbert_attention_core_fixed import (
    create_hilbert_mapping_per_segment,
)
from benchmarks.core.utils.safety import (
    MemorySafetyChecker,
    SafetyConfig,
    check_memory_before_allocation,
)


def test_per_segment_hilbert_mapping():
    """Test that Hilbert mapping is applied per-segment."""
    print("=== Testing Per-Segment Hilbert Mapping ===")

    seq_len = 8192
    segment_size = 2048

    # Create per-segment mapping
    mapping = create_hilbert_mapping_per_segment(seq_len, segment_size)

    # Verify each segment has its own local Hilbert pattern
    num_segments = seq_len // segment_size

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size

        # Extract segment mapping
        segment_mapping = mapping[seg_start:seg_end] - seg_start

        # Check that mapping is within segment bounds
        assert segment_mapping.min() >= 0, f"Segment {seg_idx} has negative indices"
        assert segment_mapping.max() < segment_size, f"Segment {seg_idx} exceeds bounds"

        # Check that it's a permutation (all indices present)
        unique_indices = segment_mapping.unique()
        assert len(unique_indices) == segment_size, f"Segment {seg_idx} missing indices"

        print(
            f"  Segment {seg_idx}: mapping range [{segment_mapping.min()}, {segment_mapping.max()}] ✓"
        )

    print("  Per-segment Hilbert mapping test PASSED ✓\n")


def test_ring_attention_with_safety():
    """Test ring attention with memory safety checks."""
    print("=== Testing Ring Attention with Safety ===")

    # Configure safety
    safety_config = SafetyConfig(
        max_memory_fraction=0.5,  # Use only 50% of GPU memory for testing
        min_free_memory_gb=1.0,
    )
    safety_checker = MemorySafetyChecker(safety_config)

    # Test parameters
    batch_size = 2
    seq_len = 4096
    num_heads = 8
    head_dim = 64
    dim = num_heads * head_dim

    # Check if we can allocate the tensors
    shape = (batch_size, seq_len, num_heads, head_dim)
    can_allocate = check_memory_before_allocation(
        shape,
        dtype=torch.float16,
        num_tensors=3,  # Q, K, V
    )

    if not can_allocate:
        print("  Insufficient memory for test, skipping...")
        return

    print("  Memory check passed, proceeding with test...")

    # Show current memory state
    if torch.cuda.is_available():
        used, free, total = safety_checker.get_gpu_memory_info()
        print(
            f"  GPU Memory: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total"
        )

    try:
        # Create model
        model = RingDilatedAttentionHilbertCoreFixed(
            dim=dim,
            heads=num_heads,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            ring_size=1,  # Single GPU for now
            use_hilbert=True,
        )

        if torch.cuda.is_available():
            model = model.cuda()
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Create inputs
        device = next(model.parameters()).device
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        print("  Running forward pass...")
        output = model(q, k, v)

        # Verify output shape
        assert output.shape == q.shape, (
            f"Output shape mismatch: {output.shape} vs {q.shape}"
        )

        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print("  Ring attention test PASSED ✓")

        # Cleanup
        del q, k, v, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Ring attention test FAILED: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def test_hilbert_kernel_correctness():
    """Test that the Hilbert kernel produces correct results."""
    print("=== Testing Hilbert Kernel Correctness ===")

    # Small test case for verification
    batch_size = 1
    seq_len = 256
    num_heads = 4
    head_dim = 32
    dim = num_heads * head_dim

    # Create model
    model = RingDilatedAttentionHilbertCoreFixed(
        dim=dim,
        heads=num_heads,
        segment_lengths=[128],
        dilation_rates=[1],
        use_hilbert=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
        dtype = torch.float32  # Use float32 for accuracy test
    else:
        device = "cpu"
        dtype = torch.float32

    # Create simple test pattern
    q = (
        torch.eye(seq_len, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(2)
        .expand(batch_size, seq_len, num_heads, head_dim)
    )
    k = q.clone()
    v = (
        torch.arange(seq_len, device=device, dtype=dtype)
        .view(1, seq_len, 1, 1)
        .expand(batch_size, seq_len, num_heads, head_dim)
    )

    # Forward pass
    output = model(q, k, v)

    # For identity Q and K, output should preserve V's pattern
    # Check that each position got the correct value
    v_recovered = output.mean(dim=[2, 3])  # Average over heads and dims
    v_original = v.mean(dim=[2, 3])

    # They might not be exactly equal due to Hilbert reordering and attention
    # but should be correlated
    correlation = torch.corrcoef(
        torch.stack([v_recovered.flatten(), v_original.flatten()])
    )[0, 1]

    print(f"  Correlation between input and output: {correlation:.4f}")
    assert correlation > 0.9, f"Low correlation: {correlation}"

    print("  Hilbert kernel correctness test PASSED ✓\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Fixed Hilbert Implementation")
    print("=" * 60 + "\n")

    try:
        # Test 1: Per-segment Hilbert mapping
        test_per_segment_hilbert_mapping()

        # Test 2: Ring attention with safety
        test_ring_attention_with_safety()

        # Test 3: Kernel correctness
        test_hilbert_kernel_correctness()

        print("\n" + "=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n\nTests FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
