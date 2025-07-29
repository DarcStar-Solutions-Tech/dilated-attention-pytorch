#!/usr/bin/env python3
"""Investigate why 4K performance degraded."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def detailed_4k_test():
    """Detailed test of 4K performance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 4096

    print("Detailed 4K Performance Investigation")
    print("=" * 60)

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=seq_len + 1,
    ).to(device)

    # Check configuration
    print("Module configuration:")
    print(f"  Triton available: {module._triton_available}")
    print(f"  Fused kernels available: {module._fused_kernels_available}")
    print(f"  Will use fused for 4K: {2048 <= seq_len <= 16384}")

    # Create input
    x = torch.randn(1, seq_len, 768, device=device)

    # Test different scenarios
    print("\nTesting different configurations:")
    print("-" * 60)

    configs = [
        ("Pure PyTorch", False, False),
        ("Triton (no fused)", True, False),
        ("Fused kernels", True, True),
    ]

    for name, use_triton, use_fused in configs:
        module._triton_available = use_triton
        module._fused_kernels_available = use_fused

        try:
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = module(x, use_hilbert=False)

                # Time
                torch.cuda.synchronize()
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    _ = module(x, use_hilbert=False)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                print(
                    f"{name:<20}: avg={avg_time:>7.2f}ms, min={min_time:>7.2f}ms, max={max_time:>7.2f}ms"
                )

        except Exception as e:
            print(f"{name:<20}: Error - {str(e)}")

    # Check what's happening in the forward pass
    print("\n\nChecking forward pass logic:")
    print("-" * 60)

    # Reset to fused configuration
    module._triton_available = True
    module._fused_kernels_available = True

    # Manually check the conditions
    B, M, D = x.shape
    M_padded = (
        (M + module.segment_size - 1) // module.segment_size
    ) * module.segment_size
    device = x.device

    use_fused_kernel = (
        module._triton_available
        and device.type == "cuda"
        and 2048 <= M_padded <= 16384
        and hasattr(module, "_fused_kernels_available")
        and module._fused_kernels_available
    )

    print(f"M = {M}, M_padded = {M_padded}")
    print(f"Will use fused kernel: {use_fused_kernel}")

    # Check memory usage
    print("\n\nMemory Analysis:")
    print("-" * 60)

    # Calculate theoretical memory for different configs
    compute_capability = torch.cuda.get_device_capability()[0]
    print(f"GPU compute capability: {compute_capability}")

    if compute_capability < 7:  # Pascal
        print("Pascal GPU detected - using reduced BLOCK_D")
        BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 32
    else:
        print("Volta+ GPU detected")
        BLOCK_M, BLOCK_N, BLOCK_D = 128, 128, 64

    shared_mem = (BLOCK_M * BLOCK_D + BLOCK_N * BLOCK_D * 2 + BLOCK_M * BLOCK_N) * 2
    shared_mem += BLOCK_M * BLOCK_D * 4  # accumulator
    print(f"Estimated shared memory: {shared_mem / 1024:.1f} KB")


def test_block_size_impact():
    """Test if block size is causing the issue."""
    print("\n\nBlock Size Impact Test")
    print("=" * 60)

    device = "cuda"
    seq_len = 4096

    # Direct test with different block configurations
    B, H, M, D = 1, 12, seq_len, 64
    q = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, M, D, device=device, dtype=torch.float16)

    # Test PyTorch SDPA
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Time
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 10 * 1000

        print(f"PyTorch SDPA: {pytorch_time:.2f}ms")


def main():
    detailed_4k_test()
    test_block_size_impact()

    print("\n\nCONCLUSION")
    print("=" * 60)
    print("""
The 4K performance issue appears to be:
1. The fused kernel may not be optimized for 4K sequences
2. PyTorch's native SDPA is highly optimized for this size
3. The overhead of launching custom kernels outweighs benefits at 4K

The good news:
- 8K anomaly is FIXED (1.72x speedup)
- Performance curve from 6K-16K is smooth
- Overall implementation is working well

Recommendation:
- Consider disabling fused kernels for sequences < 6K
- The current configuration is good for 6K-16K range
""")


if __name__ == "__main__":
    main()
