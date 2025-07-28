#!/usr/bin/env python3
"""
Compare original and optimized Triton kernels.
"""

import torch
import time
import numpy as np


def benchmark_kernel(module, x, name, warmup=10, iters=50):
    """Benchmark a kernel with proper timing."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x)

    # Time
    torch.cuda.synchronize()
    times = []

    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            out = module(x)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = sorted(times)[5:-5]  # Remove outliers
    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, out


def main():
    """Compare kernel versions."""
    print("=== Kernel Version Comparison ===\n")

    # Import both versions
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )
    from dilated_attention_pytorch.kernels.dilated_attention_triton_v2 import (
        DilatedAttentionV2,
    )

    configs = [
        # (batch, seq_len, hidden_dim, segment_size, dilation_rate, desc)
        (1, 512, 256, 64, 1, "No dilation"),
        (1, 512, 256, 64, 2, "Dilation 2"),
        (1, 512, 256, 64, 4, "Dilation 4"),
        (1, 1024, 256, 128, 4, "Large seq"),
        (4, 512, 256, 64, 4, "Batched"),
        (1, 2048, 512, 256, 8, "Very sparse"),
    ]

    for batch, seq_len, hidden_dim, seg_size, dil_rate, desc in configs:
        print(
            f"\n{desc}: B={batch}, Seq={seq_len}, Hidden={hidden_dim}, Seg={seg_size}, Dil={dil_rate}"
        )

        # Create modules
        v1 = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=8,
                segment_size=seg_size,
                dilation_rate=dil_rate,
                use_custom_backward=False,
            )
            .cuda()
            .eval()
        )

        v2 = (
            DilatedAttentionV2(
                hidden_dim=hidden_dim,
                num_heads=8,
                segment_size=seg_size,
                dilation_rate=dil_rate,
            )
            .cuda()
            .eval()
        )

        # Copy weights for fair comparison
        with torch.no_grad():
            v2.qkv_proj.weight.copy_(v1.qkv_proj.weight)
            v2.out_proj.weight.copy_(v1.out_proj.weight)

        x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

        # Benchmark
        time_v1, std_v1, out_v1 = benchmark_kernel(v1, x, "V1")
        time_v2, std_v2, out_v2 = benchmark_kernel(v2, x, "V2")

        # Verify outputs match
        with torch.no_grad():
            diff = (out_v1 - out_v2).abs().max().item()
            rel_diff = diff / out_v1.abs().max().item()

        print(f"  Original V1: {time_v1:.2f} ± {std_v1:.2f} ms")
        print(f"  Optimized V2: {time_v2:.2f} ± {std_v2:.2f} ms")
        print(f"  Speedup: {time_v1 / time_v2:.2f}x")
        print(f"  Output diff: {diff:.6f} (relative: {rel_diff:.6f})")

        # Calculate theoretical speedup
        total_attn = seq_len * seq_len
        actual_attn = seq_len * (seg_size // dil_rate)
        theoretical_speedup = total_attn / actual_attn

        print(f"  Theoretical speedup: {theoretical_speedup:.1f}x")
        print(f"  Efficiency: {(time_v1 / time_v2) / theoretical_speedup * 100:.1f}%")

    print("\n=== Summary ===")
    print("The V2 kernel shows improvements by:")
    print("1. Processing dilated positions in groups")
    print("2. Better memory access patterns")
    print("3. Reduced redundant computations")
    print("\nFurther optimizations possible with:")
    print("- Custom CUDA kernels for specific patterns")
    print("- Hardware-specific tuning")
    print("- Fused QKV projections")


if __name__ == "__main__":
    main()
