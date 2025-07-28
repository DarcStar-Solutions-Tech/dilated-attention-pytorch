#!/usr/bin/env python3
"""
Detailed Triton kernel benchmark focusing on specific scenarios.
"""

import torch
import time
import numpy as np


def benchmark_with_stable_timing(func, warmup=20, iters=100):
    """More stable timing with outlier removal."""
    # Warmup
    for _ in range(warmup):
        func()

    torch.cuda.synchronize()

    # Collect timings
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Remove outliers (top and bottom 10%)
    times = sorted(times)
    trim = int(len(times) * 0.1)
    if trim > 0:
        times = times[trim:-trim]

    return np.mean(times), np.std(times)


def benchmark_triton_vs_pytorch_attention():
    """Compare our Triton implementation with PyTorch's native attention."""
    print("=== Triton vs PyTorch Native Attention ===\n")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    configs = [
        # (batch, seq_len, hidden_dim, num_heads)
        (1, 128, 256, 8),
        (1, 256, 256, 8),
        (1, 512, 256, 8),
        (4, 256, 256, 8),
        (4, 512, 256, 8),
    ]

    for batch, seq_len, hidden_dim, num_heads in configs:
        print(
            f"\nConfig: batch={batch}, seq_len={seq_len}, hidden={hidden_dim}, heads={num_heads}"
        )

        # Our implementation
        triton_model = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=min(128, seq_len),
                use_custom_backward=False,
            )
            .cuda()
            .eval()
        )

        # Input
        x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

        # Native PyTorch scaled_dot_product_attention comparison
        head_dim = hidden_dim // num_heads

        def pytorch_attention():
            with torch.no_grad():
                # Reshape for attention
                x_reshaped = x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
                # Simple self-attention
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    x_reshaped, x_reshaped, x_reshaped, dropout_p=0.0, is_causal=False
                )
                return (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(batch, seq_len, hidden_dim)
                )

        def triton_attention():
            with torch.no_grad():
                return triton_model(
                    x, use_hilbert=False
                )  # No Hilbert for fair comparison

        def triton_hilbert_attention():
            with torch.no_grad():
                return triton_model(x, use_hilbert=True)

        # Benchmark
        time_pytorch, std_pytorch = benchmark_with_stable_timing(pytorch_attention)
        time_triton, std_triton = benchmark_with_stable_timing(triton_attention)
        time_hilbert, std_hilbert = benchmark_with_stable_timing(
            triton_hilbert_attention
        )

        print(f"  PyTorch SDPA:     {time_pytorch:.3f} ± {std_pytorch:.3f} ms")
        print(f"  Triton Standard:  {time_triton:.3f} ± {std_triton:.3f} ms")
        print(f"  Triton Hilbert:   {time_hilbert:.3f} ± {std_hilbert:.3f} ms")
        print(f"  Triton/PyTorch:   {time_triton / time_pytorch:.2f}x")
        print(f"  Hilbert/Standard: {time_hilbert / time_triton:.2f}x")


def benchmark_different_segment_sizes():
    """Benchmark impact of segment size on performance."""
    print("\n=== Segment Size Impact ===\n")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    seq_len = 512
    hidden_dim = 256
    num_heads = 8
    batch_size = 4

    segment_sizes = [16, 32, 64, 128, 256]

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    for seg_size in segment_sizes:
        if seq_len % seg_size != 0:
            continue

        model = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=seg_size,
                use_custom_backward=False,
            )
            .cuda()
            .eval()
        )

        def forward():
            with torch.no_grad():
                return model(x, use_hilbert=True)

        time_ms, std_ms = benchmark_with_stable_timing(forward)
        throughput = (batch_size * seq_len) / (time_ms / 1000)

        print(
            f"Segment size {seg_size:3d}: {time_ms:.3f} ± {std_ms:.3f} ms ({throughput:.0f} tokens/sec)"
        )


def benchmark_head_dimension_impact():
    """Benchmark impact of head dimension on Triton performance."""
    print("\n=== Head Dimension Impact ===\n")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    seq_len = 256
    batch_size = 4

    # Test different head configurations
    configs = [
        (128, 8),  # hidden=128, heads=8, head_dim=16
        (256, 8),  # hidden=256, heads=8, head_dim=32
        (512, 8),  # hidden=512, heads=8, head_dim=64
        (256, 16),  # hidden=256, heads=16, head_dim=16
        (512, 16),  # hidden=512, heads=16, head_dim=32
        (768, 12),  # hidden=768, heads=12, head_dim=64
    ]

    for hidden_dim, num_heads in configs:
        head_dim = hidden_dim // num_heads

        model = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=64,
                use_custom_backward=False,
            )
            .cuda()
            .eval()
        )

        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        def forward():
            with torch.no_grad():
                return model(x, use_hilbert=True)

        time_ms, std_ms = benchmark_with_stable_timing(forward)

        print(
            f"Hidden={hidden_dim:3d}, Heads={num_heads:2d}, HeadDim={head_dim:2d}: "
            f"{time_ms:.3f} ± {std_ms:.3f} ms"
        )


def benchmark_memory_bandwidth():
    """Estimate memory bandwidth utilization."""
    print("\n=== Memory Bandwidth Analysis ===\n")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    # Test configuration
    batch_size = 4
    seq_len = 512
    hidden_dim = 256
    num_heads = 8

    model = (
        HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            use_custom_backward=False,
        )
        .cuda()
        .eval()
    )

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    def forward():
        with torch.no_grad():
            return model(x, use_hilbert=True)

    # Measure time
    time_ms, _ = benchmark_with_stable_timing(forward, warmup=50, iters=200)

    # Calculate memory operations
    # Attention requires: Q, K, V reads + attention weights + output write
    # Approximate memory: 3 * input + attention_matrix + output
    input_bytes = batch_size * seq_len * hidden_dim * 4  # float32
    attention_bytes = batch_size * num_heads * seq_len * seq_len * 4
    output_bytes = batch_size * seq_len * hidden_dim * 4

    total_bytes = 3 * input_bytes + attention_bytes + output_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

    print(f"Configuration: {batch_size}x{seq_len}x{hidden_dim}")
    print(f"Time: {time_ms:.3f} ms")
    print(f"Total memory: {total_bytes / 1e6:.1f} MB")
    print(f"Effective bandwidth: {bandwidth_gb_s:.1f} GB/s")

    # GTX 1080 theoretical bandwidth: ~320 GB/s
    print(f"Bandwidth utilization: {(bandwidth_gb_s / 320) * 100:.1f}%")


def main():
    """Run detailed benchmarks."""
    print("=== Detailed Triton Kernel Benchmarks ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Set to highest performance
    torch.backends.cudnn.benchmark = True

    # Run benchmarks
    benchmark_triton_vs_pytorch_attention()
    benchmark_different_segment_sizes()
    benchmark_head_dimension_impact()
    benchmark_memory_bandwidth()

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
