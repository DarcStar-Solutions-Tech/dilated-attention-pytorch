#!/usr/bin/env python3
"""
Safe benchmark to test Hilbert attention concerns without lockups.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_simple_dilation_mask(
    seq_len: int, segment_length: int, dilation_rate: int
) -> torch.Tensor:
    """Create a simple attention mask for dilated patterns."""
    mask = torch.zeros(seq_len, seq_len)

    # Simple segmented pattern
    num_segments = seq_len // segment_length
    for seg in range(num_segments):
        start = seg * segment_length
        end = start + segment_length

        # Within segment, apply dilation
        for i in range(start, end, dilation_rate):
            for j in range(start, end, dilation_rate):
                mask[i, j] = 1.0

    # Convert to attention mask
    return torch.where(mask == 1, 0.0, float("-inf"))


def test_sdpa_with_mask():
    """Test SDPA with custom dilation mask."""
    print("\n=== Testing SDPA with Dilation Mask ===")

    # Small test case
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    segment_length = 128
    dilation_rate = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create inputs
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Create mask
    mask = create_simple_dilation_mask(seq_len, segment_length, dilation_rate)
    mask = (
        mask.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    )

    print(f"Input shape: {q.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Test SDPA
    try:
        start = time.perf_counter()
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"SDPA Success! Time: {elapsed * 1000:.2f}ms")
        print(f"Output shape: {output.shape}")

        # Check if attention is actually dilated
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            scores_masked = scores + mask
            attn = F.softmax(scores_masked, dim=-1)

            # Check sparsity
            sparsity = (attn[0, 0] == 0).float().mean().item()
            print(f"Attention sparsity: {sparsity * 100:.1f}%")

    except Exception as e:
        print(f"SDPA Error: {e}")
        import traceback

        traceback.print_exc()


def test_manual_vs_sdpa():
    """Compare manual attention with SDPA."""
    print("\n=== Manual vs SDPA Comparison ===")

    batch_size = 1
    seq_len = 256
    num_heads = 4
    head_dim = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Manual attention
    start = time.perf_counter()
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    attn = F.softmax(scores, dim=-1)
    output_manual = torch.matmul(attn, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    manual_time = time.perf_counter() - start

    # SDPA
    start = time.perf_counter()
    output_sdpa = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sdpa_time = time.perf_counter() - start

    print(f"Manual time: {manual_time * 1000:.2f}ms")
    print(f"SDPA time: {sdpa_time * 1000:.2f}ms")
    print(f"Speedup: {manual_time / sdpa_time:.2f}x")

    # Check accuracy
    diff = (output_manual - output_sdpa).abs().max().item()
    print(f"Max difference: {diff:.6f}")


def test_gpu_info():
    """Test GPU capabilities."""
    print("\n=== GPU Information ===")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi-GPU: {'Yes' if torch.cuda.device_count() > 1 else 'No'}")

        # Check for Pascal (compute 6.x)
        if props.major == 6:
            print("  Architecture: Pascal (P100/GTX 10xx) - Using float32")
        elif props.major == 7:
            print("  Architecture: Volta/Turing (V100/RTX 20xx)")
        elif props.major == 8:
            print("  Architecture: Ampere (A100/RTX 30xx)")
        elif props.major == 9:
            print("  Architecture: Hopper (H100)")


def test_hilbert_import():
    """Test if we can import Hilbert implementations."""
    print("\n=== Testing Hilbert Imports ===")

    try:
        from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core import (
            RingDilatedAttentionHilbertCore,
        )

        print("✓ RingDilatedAttentionHilbertCore imported successfully")

        # Try to create a small instance
        model = RingDilatedAttentionHilbertCore(
            embed_dim=256,
            num_heads=4,
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            use_fa3=False,
        )
        print("✓ Model created successfully")

        # Test forward pass with tiny input
        x = torch.randn(1, 256, 256, dtype=torch.float32)
        if torch.cuda.is_available():
            model = model.cuda().float()
            x = x.cuda()

        output = model(x)
        print(f"✓ Forward pass successful, output shape: {output.shape}")

    except Exception as e:
        print(f"✗ Import/execution failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests."""
    print("=" * 80)
    print("Hilbert Attention Safe Benchmark")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    # Run tests
    test_gpu_info()
    test_manual_vs_sdpa()
    test_sdpa_with_mask()
    test_hilbert_import()

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
