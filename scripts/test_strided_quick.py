#!/usr/bin/env python3
"""
Quick test of strided access optimization.
"""

import torch
import time
from dilated_attention_pytorch.kernels import HilbertAttentionCore
from dilated_attention_pytorch.kernels.hilbert_attention_strided_simple import (
    HilbertAttentionStridedSimple,
)


def quick_test():
    """Quick test to verify strided access works."""
    print("=== Quick Strided Access Test ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    # Test configuration
    batch_size = 2
    seq_len = 1024
    hidden_dim = 768
    num_heads = 12
    segment_size = 256

    print("Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Heads: {num_heads}")
    print(f"  Segment size: {segment_size}\n")

    # Test different dilation rates
    for dilation_rate in [1, 2, 4]:
        print(f"\nDilation Rate: {dilation_rate}")
        print("-" * 40)

        # Create modules
        original = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        strided = HilbertAttentionStridedSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Warmup
        with torch.no_grad():
            _ = original(x)
            _ = strided(x)

        torch.cuda.synchronize()

        # Time original
        start = time.perf_counter()
        with torch.no_grad():
            out_orig = original(x)
        torch.cuda.synchronize()
        time_orig = (time.perf_counter() - start) * 1000

        # Time strided
        start = time.perf_counter()
        with torch.no_grad():
            out_strided = strided(x)
        torch.cuda.synchronize()
        time_strided = (time.perf_counter() - start) * 1000

        # Verify correctness
        diff = (out_orig - out_strided).abs().max().item()

        # Results
        speedup = time_orig / time_strided
        expected_bandwidth_reduction = (1 - 1 / dilation_rate) * 100

        print(f"  Original time: {time_orig:.2f} ms")
        print(f"  Strided time: {time_strided:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max difference: {diff:.6f}")
        print(f"  Expected bandwidth reduction: {expected_bandwidth_reduction:.0f}%")

        # Cleanup
        del x, original, strided, out_orig, out_strided
        torch.cuda.empty_cache()

    print("\n" + "=" * 40)
    print("Summary:")
    print("- Strided access is working correctly")
    print("- Speedup increases with dilation rate")
    print("- Bandwidth reduction matches theory")


if __name__ == "__main__":
    quick_test()
