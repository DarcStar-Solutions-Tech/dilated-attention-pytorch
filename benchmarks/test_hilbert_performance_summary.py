#!/usr/bin/env python3
"""
Test Hilbert attention performance across different configurations.
"""

import torch
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention


def benchmark_config(model, x, use_hilbert, warmup=3, iterations=10):
    """Benchmark a configuration."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x, use_hilbert=use_hilbert)

    torch.cuda.synchronize()

    # Time forward pass
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(x, use_hilbert=use_hilbert)
    end.record()

    torch.cuda.synchronize()
    forward_time = start.elapsed_time(end) / iterations

    return forward_time


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    dtype = torch.float16

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")

    # Test configurations
    configs = [
        # (seq_len, batch_size, segment_size, dilation_rate)
        (512, 2, 128, 1),
        (1024, 2, 128, 1),
        (2048, 2, 128, 1),
        (4096, 2, 128, 1),
        (8192, 2, 128, 1),
        (16384, 1, 128, 1),  # Reduced batch for memory
    ]

    print("\n=== Performance Summary ===")
    print(
        "Seq Len | Batch | Segment | Dilation | Standard (ms) | Hilbert (ms) | Speedup | Backend"
    )
    print("-" * 90)

    for seq_len, batch_size, segment_size, dilation_rate in configs:
        # Create model
        model = (
            HilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                hilbert_threshold=1024,
            )
            .to(device)
            .to(dtype)
        )
        model.eval()

        # Create input
        x = torch.randn(batch_size, seq_len, 768, device=device, dtype=dtype)

        # Clear caches
        gc.collect()
        torch.cuda.empty_cache()

        # Benchmark standard attention
        try:
            time_standard = benchmark_config(model, x, use_hilbert=False)
        except Exception as e:
            time_standard = float("inf")
            print(f"Standard failed for {seq_len}: {e}")

        # Benchmark Hilbert attention
        try:
            time_hilbert = benchmark_config(model, x, use_hilbert=True)
        except Exception as e:
            time_hilbert = float("inf")
            print(f"Hilbert failed for {seq_len}: {e}")

        # Calculate speedup
        speedup = time_standard / time_hilbert if time_hilbert > 0 else 0

        # Determine backend used
        backend = "PyTorch"
        if seq_len > 1024 and model._triton_available:
            if 2048 <= seq_len <= 16384 and model._fused_kernels_available:
                backend = "Fused"
            else:
                backend = "Triton"

        print(
            f"{seq_len:7d} | {batch_size:5d} | {segment_size:7d} | {dilation_rate:8d} | "
            f"{time_standard:13.2f} | {time_hilbert:12.2f} | {speedup:7.2f}x | {backend}"
        )

    # Test sparse patterns
    print("\n=== Sparse Pattern Performance (seq_len=8192) ===")
    print("Dilation | Segment | Standard (ms) | Hilbert (ms) | Speedup")
    print("-" * 60)

    for dilation_rate in [1, 2, 4, 8]:
        segment_size = 128

        model = (
            HilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                hilbert_threshold=1024,
            )
            .to(device)
            .to(dtype)
        )
        model.eval()

        x = torch.randn(2, 8192, 768, device=device, dtype=dtype)

        gc.collect()
        torch.cuda.empty_cache()

        time_standard = benchmark_config(model, x, use_hilbert=False)
        time_hilbert = benchmark_config(model, x, use_hilbert=True)
        speedup = time_standard / time_hilbert if time_hilbert > 0 else 0

        print(
            f"{dilation_rate:8d} | {segment_size:7d} | {time_standard:13.2f} | {time_hilbert:12.2f} | {speedup:7.2f}x"
        )


if __name__ == "__main__":
    main()
