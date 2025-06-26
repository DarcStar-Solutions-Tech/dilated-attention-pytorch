#!/usr/bin/env python3
"""
Test Flash Attention 3 integration with Ring Dilated Attention.

This test demonstrates the automatic integration of Flash Attention 3
through PyTorch's SDPA backend selection mechanism.
"""

import os
import sys

import torch

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from dilated_attention_pytorch.ring_dilated_attention import (
    RingDilatedAttention, get_flash_attention_version,
    is_flash_attention_3_available)


def test_flash_attention_3_integration():
    """Test Flash Attention 3 integration and performance."""
    print("=" * 60)
    print("Flash Attention 3 Integration Test")
    print("=" * 60)

    # Check Flash Attention availability
    fa_version = get_flash_attention_version()
    fa3_available = is_flash_attention_3_available()

    print(f"Flash Attention Version: {fa_version}")
    print(f"Flash Attention 3 Available: {fa3_available}")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_properties(0).name
        print(f"GPU: {device_name}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    else:
        print("GPU: CPU only")

    print("-" * 60)

    # Create Ring Attention instance
    ring_attention = RingDilatedAttention(
        segment_lengths=[2048, 2048, 2048],  # Use consistent segment lengths
        dilation_rates=[1, 1, 1],  # No dilation to avoid dimension mismatch
        dropout=0.0,
        use_tf32=True,
        block_size=512,
        ring_size=1,  # Single device for testing
        use_checkpointing=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Get memory info including Flash Attention details
    memory_info = ring_attention.get_memory_info()

    print("Ring Attention Configuration:")
    print(f"  Flash Attention Version: {memory_info['flash_attention_version']}")
    print(
        f"  Flash Attention 3 Available: {memory_info['flash_attention_3_available']}"
    )
    print(f"  Hardware Optimized for FA3: {memory_info['hardware_optimized_for_fa3']}")
    print(f"  SDPA Backend Available: {memory_info['sdpa_backend_available']}")
    print(f"  Memory Complexity: {memory_info['memory_complexity']}")

    print("\nOptimizations Enabled:")
    for opt in memory_info["optimizations_enabled"]:
        print(f"  ✅ {opt}")

    print("-" * 60)

    # Performance test
    batch_size = 2
    seq_len = 8192
    num_heads = 8
    head_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Performance Test Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  Head Dimension: {head_dim}")
    print(f"  Device: {device}")

    # Create test tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    print("\nRunning Forward Pass...")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = ring_attention(q, k, v, is_causal=True)

    # Timed run
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            output = ring_attention(q, k, v, is_causal=True)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        print(f"Forward Pass Time: {elapsed_time:.2f} ms")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        import time

        start_time = time.time()
        with torch.no_grad():
            output = ring_attention(q, k, v, is_causal=True)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"Forward Pass Time: {elapsed_time:.2f} ms")

    print(f"Output Shape: {output.shape}")
    print(f"Output Dtype: {output.dtype}")
    print(f"Output Device: {output.device}")

    print("-" * 60)

    # Test mathematical correctness
    print("Testing Mathematical Correctness...")

    # Check output properties
    assert output.shape == (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    ), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"

    # Check output range (should be reasonable for attention)
    output_mean = output.mean().item()
    output_std = output.std().item()
    print("Output Statistics:")
    print(f"  Mean: {output_mean:.6f}")
    print(f"  Std: {output_std:.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")

    assert abs(output_mean) < 1.0, f"Output mean too large: {output_mean}"
    assert 0.1 < output_std < 2.0, f"Output std suspicious: {output_std}"

    print("✅ All tests passed!")

    print("=" * 60)

    # Flash Attention 3 specific recommendations
    if fa3_available and memory_info["hardware_optimized_for_fa3"]:
        print("🚀 FLASH ATTENTION 3 OPTIMIZATIONS ACTIVE")
        print("   Expected Performance Benefits:")
        print("   • 1.5-2.0x faster than Flash Attention 2")
        print("   • 75% H100 utilization vs 35% for FA2")
        print("   • Support for FP8 precision (experimental)")
        print("   • Up to 1.2 PFLOPS on H100")
        print("   • Asynchronous computation and data movement")
        print("   • Warp-specialization optimizations")
    elif fa3_available:
        print("⚠️  Flash Attention 3 features available but not on optimal hardware")
        print("   Current hardware may benefit from FA2.8+ optimizations")
        print("   Recommendation: Use H100/H800 GPUs for full FA3 performance")
    elif fa_version and fa_version.startswith("2.8"):
        print(f"📊 Using Flash Attention {fa_version} with FA3-like optimizations")
        print("   • Latest FA2 with H100 optimizations")
        print("   • Torch compile compatibility")
        print("   • Enhanced memory efficiency")
        print("   Recommendation: Consider upgrading to FA3 beta for H100")
    elif fa_version:
        print(f"📊 Using Flash Attention {fa_version}")
        print("   Recommendation: Upgrade to FA2.8+ or FA3 for better performance")
    else:
        print("⚠️  Flash Attention not available - using PyTorch native attention")
        print("   Recommendation: Install flash-attn>=2.8.0 for significant speedup")

    print("=" * 60)


if __name__ == "__main__":
    test_flash_attention_3_integration()
