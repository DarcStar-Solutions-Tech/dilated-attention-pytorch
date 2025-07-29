#!/usr/bin/env python3
"""Debug why 4K Triton performance is poor."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def test_triton_configurations():
    """Test different Triton configurations for 4K."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 4096

    print("Debugging 4K Triton Performance")
    print("=" * 60)

    # Create test data
    B, H, M, D = 1, 12, seq_len, 64
    q = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    scale = D**-0.5

    # Test baseline PyTorch
    print("\n1. PyTorch Baseline:")
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 10 * 1000
        print(f"   Time: {pytorch_time:.2f}ms")

    # Test our implementation
    print("\n2. Our HilbertAttention Module:")
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=seq_len + 1,  # Disable Hilbert for now
    ).to(device)

    x = torch.randn(1, seq_len, 768, device=device)

    # Test with different settings
    configs = [
        ("PyTorch path", False, False),
        ("Triton (no fused)", True, False),
        ("Fused kernels", True, True),
    ]

    for name, use_triton, use_fused in configs:
        module._triton_available = use_triton
        module._fused_kernels_available = use_fused

        with torch.no_grad():
            # Warmup
            for _ in range(2):
                _ = module(x, use_hilbert=False)

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                _ = module(x, use_hilbert=False)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 5 * 1000

            print(f"   {name}: {elapsed:.2f}ms")

    # Check what's happening with the fused kernel
    print("\n3. Fused Kernel Analysis:")

    # Import the fused kernel directly
    try:
        from dilated_attention_pytorch.kernels.hilbert_attention_fused_v2 import (
            launch_fused_kernel,
        )

        # Test direct kernel call
        with torch.no_grad():
            _ = launch_fused_kernel(q, k, v, scale)

            # Warmup
            for _ in range(3):
                _ = launch_fused_kernel(q, k, v, scale)

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                _ = launch_fused_kernel(q, k, v, scale)
            torch.cuda.synchronize()
            kernel_time = (time.perf_counter() - start) / 10 * 1000

            print(f"   Direct kernel call: {kernel_time:.2f}ms")

            speedup = pytorch_time / kernel_time
            print(f"   Speedup vs PyTorch: {speedup:.2f}x")

    except Exception as e:
        print(f"   Error: {e}")

    # Check configuration
    print("\n4. Configuration Analysis:")
    compute_capability = torch.cuda.get_device_capability()[0]
    print(f"   Compute capability: {compute_capability}")

    if compute_capability < 7:
        print("   Using Pascal configuration:")
        print("   - BLOCK_M = 64, BLOCK_N = 64")
        print("   - BLOCK_D = 32 (reduced for memory)")
        print("   - Shared memory: ~36.5KB")
    else:
        print("   Using Volta+ configuration:")
        print("   - BLOCK_M = 128, BLOCK_N = 128")
        print("   - BLOCK_D = 64")

    # Test overhead
    print("\n5. Module Overhead Analysis:")

    # Time just the QKV projection
    module._triton_available = False
    module._fused_kernels_available = False

    with torch.no_grad():
        # Time QKV projection
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module.qkv_proj(x)
        torch.cuda.synchronize()
        qkv_time = (time.perf_counter() - start) / 10 * 1000

        print(f"   QKV projection: {qkv_time:.2f}ms")

        # Time output projection
        dummy_out = torch.randn_like(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module.out_proj(dummy_out)
        torch.cuda.synchronize()
        out_proj_time = (time.perf_counter() - start) / 10 * 1000

        print(f"   Output projection: {out_proj_time:.2f}ms")
        print(f"   Total projection overhead: {qkv_time + out_proj_time:.2f}ms")


def main():
    test_triton_configurations()

    print("\n\nCONCLUSIONS")
    print("=" * 60)
    print("""
The issue appears to be:
1. Module overhead (QKV and output projections) adds significant time
2. The fused kernel may not be optimized for Pascal's constraints
3. PyTorch's SDPA is highly optimized for 4K sequences

Potential fixes:
1. Optimize the fused kernel specifically for 4K
2. Reduce module overhead by caching projections
3. Use different block configurations for 4K
""")


if __name__ == "__main__":
    main()
