#!/usr/bin/env python3
"""
Test Ring V3 with bucketed processing.
"""

import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_bucketed_vs_non_bucketed():
    """Compare bucketed vs non-bucketed processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("Testing Bucketed vs Non-Bucketed Processing")
    print("=" * 50)

    # Test configuration
    seq_len = 2048
    batch_size = 1
    num_heads = 4
    head_dim = 32

    # Create models
    model_bucketed = RingDilatedAttentionV3(
        segment_lengths=[1024],
        dilation_rates=[1],
        bucket_size=256,  # Small buckets
        use_bucketed=True,
        device=device,
        dtype=dtype,
    )

    model_standard = RingDilatedAttentionV3(
        segment_lengths=[1024],
        dilation_rates=[1],
        use_bucketed=False,
        device=device,
        dtype=dtype,
    )

    # Create test inputs
    torch.manual_seed(42)
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    print("Testing non-causal mode...")

    # Forward pass - non-causal
    output_bucketed = model_bucketed(q, k, v, is_causal=False)
    output_standard = model_standard(q, k, v, is_causal=False)

    # Compare outputs
    diff = (output_bucketed - output_standard).abs().max().item()
    print(f"  Max difference: {diff:.6e}")

    if diff < 1e-5:
        print("  ✅ Bucketed matches standard!")
    else:
        print("  ❌ Outputs differ significantly")
        print(f"     Bucketed mean: {output_bucketed.mean().item():.6f}")
        print(f"     Standard mean: {output_standard.mean().item():.6f}")

    print("\nTesting causal mode...")

    # Forward pass - causal
    output_bucketed_causal = model_bucketed(q, k, v, is_causal=True)
    output_standard_causal = model_standard(q, k, v, is_causal=True)

    # Compare outputs
    diff_causal = (output_bucketed_causal - output_standard_causal).abs().max().item()
    print(f"  Max difference: {diff_causal:.6e}")

    if diff_causal < 1e-5:
        print("  ✅ Bucketed matches standard!")
    else:
        print("  ❌ Outputs differ")
        print(f"     Bucketed mean: {output_bucketed_causal.mean().item():.6f}")
        print(f"     Standard mean: {output_standard_causal.mean().item():.6f}")


def test_memory_efficiency():
    """Test memory usage with bucketed processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("\nSkipping memory test (no CUDA)")
        return

    print("\n\nTesting Memory Efficiency")
    print("=" * 50)

    # Large sequence to test memory
    seq_len = 8192
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model with small buckets
    model = RingDilatedAttentionV3(
        segment_lengths=[2048],
        dilation_rates=[1],
        bucket_size=512,  # Small buckets for memory efficiency
        use_bucketed=True,
        grad_checkpoint_buckets=False,  # Can enable for more savings
        device=device,
        dtype=torch.float16,  # Use half precision
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    # Get memory before forward
    mem_before = torch.cuda.memory_allocated() / (1024**2)

    try:
        # Forward pass
        _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        mem_after = torch.cuda.memory_allocated() / (1024**2)

        print(f"Sequence length: {seq_len:,}")
        print(f"Bucket size: {model.bucket_size}")
        print(f"Memory before: {mem_before:.1f} MB")
        print(f"Memory after: {mem_after:.1f} MB")
        print(f"Peak memory: {peak_memory:.1f} MB")
        print(f"✅ Forward pass succeeded!")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ Out of memory with bucketed processing")
        else:
            raise


def test_gradient_checkpointing():
    """Test with gradient checkpointing enabled."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n\nTesting Gradient Checkpointing")
    print("=" * 50)

    # Create model with gradient checkpointing
    model = RingDilatedAttentionV3(
        segment_lengths=[512],
        dilation_rates=[1],
        bucket_size=256,
        use_bucketed=True,
        grad_checkpoint_buckets=True,  # Enable gradient checkpointing
        device=device,
        dtype=torch.float32,
    )

    # Enable training mode
    model.train()

    # Small test
    seq_len = 1024
    q = torch.randn(1, seq_len, 4, 32, device=device, requires_grad=True)
    k = torch.randn(1, seq_len, 4, 32, device=device, requires_grad=True)
    v = torch.randn(1, seq_len, 4, 32, device=device, requires_grad=True)

    # Forward pass
    output = model(q, k, v, is_causal=False)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    if q.grad is not None:
        print("✅ Gradients computed successfully")
        print(f"   Q grad norm: {q.grad.norm().item():.6f}")
        print(f"   K grad norm: {k.grad.norm().item():.6f}")
        print(f"   V grad norm: {v.grad.norm().item():.6f}")
    else:
        print("❌ No gradients computed")


def test_different_bucket_sizes():
    """Test performance with different bucket sizes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n\nTesting Different Bucket Sizes")
    print("=" * 50)

    seq_len = 2048
    bucket_sizes = [128, 256, 512, 1024]

    torch.manual_seed(42)
    q = torch.randn(1, seq_len, 4, 32, device=device)
    k = torch.randn(1, seq_len, 4, 32, device=device)
    v = torch.randn(1, seq_len, 4, 32, device=device)

    outputs = []

    for bucket_size in bucket_sizes:
        model = RingDilatedAttentionV3(
            segment_lengths=[1024],
            dilation_rates=[1],
            bucket_size=bucket_size,
            use_bucketed=True,
            device=device,
            dtype=torch.float32,
        )

        output = model(q, k, v, is_causal=False)
        outputs.append(output)

        print(f"Bucket size {bucket_size}: output mean = {output.mean().item():.6f}")

    # Check consistency
    print("\nChecking consistency across bucket sizes:")
    reference = outputs[0]
    for i, output in enumerate(outputs[1:], 1):
        diff = (output - reference).abs().max().item()
        print(f"  Bucket {bucket_sizes[i]} vs {bucket_sizes[0]}: max diff = {diff:.6e}")

        if diff < 1e-5:
            print("    ✅ Consistent")
        else:
            print("    ❌ Inconsistent!")


if __name__ == "__main__":
    test_bucketed_vs_non_bucketed()
    test_memory_efficiency()
    test_gradient_checkpointing()
    test_different_bucket_sizes()
