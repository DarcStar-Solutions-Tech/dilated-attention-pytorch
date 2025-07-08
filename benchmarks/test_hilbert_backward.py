#!/usr/bin/env python3
"""
Test and compare Hilbert attention backward pass performance.
"""

import torch
import torch.nn as nn
import time
import gc


def benchmark_model(model, x, name, num_iterations=10):
    """Benchmark forward and backward passes."""
    device = x.device

    # Warmup
    print(f"\nWarming up {name}...")
    for _ in range(3):
        x.grad = None
        out = model(x)
        loss = out.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Measure forward pass only
    forward_times = []
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        forward_times.append((end - start) * 1000)

    avg_forward = sum(forward_times) / len(forward_times)

    # Measure forward + backward
    total_times = []
    for _ in range(num_iterations):
        x.grad = None

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        out = model(x)
        loss = out.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_times.append((end - start) * 1000)

    avg_total = sum(total_times) / len(total_times)
    avg_backward = avg_total - avg_forward

    return avg_forward, avg_backward, avg_backward / avg_forward


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test configuration
    batch_size = 2
    seq_len = 2048
    hidden_dim = 768
    num_heads = 12

    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

    print("\nComparing Hilbert Attention Backward Pass Performance")
    print("=" * 60)
    print(
        f"Configuration: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}, heads={num_heads}"
    )

    # Test original implementation
    try:
        from dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton_fixed import (
            HilbertAttentionTritonFixed,
        )

        model_original = HilbertAttentionTritonFixed(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
        ).to(device)

        fwd_orig, bwd_orig, ratio_orig = benchmark_model(
            model_original, x, "Original", num_iterations=5
        )
        print("\nOriginal Implementation:")
        print(f"  Forward:  {fwd_orig:8.2f} ms")
        print(f"  Backward: {bwd_orig:8.2f} ms")
        print(f"  Ratio:    {ratio_orig:8.2f}x")

    except Exception as e:
        print(f"\nOriginal implementation error: {e}")
        fwd_orig, bwd_orig = 0, 0

    # Test unified implementation with custom backward
    try:
        from dilated_attention_pytorch.kernels.hilbert_attention_core import (
            HilbertAttentionCore,
        )

        model_optimized = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
            use_custom_backward=True,
        ).to(device)

        fwd_opt, bwd_opt, ratio_opt = benchmark_model(
            model_optimized, x, "Unified (Custom Backward)", num_iterations=10
        )
        print("\nOptimized Implementation:")
        print(f"  Forward:  {fwd_opt:8.2f} ms")
        print(f"  Backward: {bwd_opt:8.2f} ms")
        print(f"  Ratio:    {ratio_opt:8.2f}x")

    except Exception as e:
        print(f"\nOptimized implementation error: {e}")
        fwd_opt, bwd_opt = 0, 0

    # Test standard dilated attention for comparison
    try:
        from dilated_attention_pytorch import ImprovedDilatedAttention

        # Need to wrap it to match interface
        class WrappedDilated(nn.Module):
            def __init__(self, hidden_dim, num_heads):
                super().__init__()
                self.attention = ImprovedDilatedAttention(
                    segment_lengths=[512, 1024, 2048], dilation_rates=[1, 2, 4]
                )
                self.num_heads = num_heads
                self.head_dim = hidden_dim // num_heads

            def forward(self, x):
                B, L, D = x.shape
                x = x.view(B, L, self.num_heads, self.head_dim)
                out = self.attention(x, x, x)
                return out.view(B, L, D)

        model_standard = WrappedDilated(hidden_dim, num_heads).to(device)

        fwd_std, bwd_std, ratio_std = benchmark_model(
            model_standard, x, "Standard Dilated", num_iterations=10
        )
        print("\nStandard Dilated Attention (for reference):")
        print(f"  Forward:  {fwd_std:8.2f} ms")
        print(f"  Backward: {bwd_std:8.2f} ms")
        print(f"  Ratio:    {ratio_std:8.2f}x")

    except Exception as e:
        print(f"\nStandard implementation error: {e}")
        fwd_std, bwd_std = 0, 0

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")

    if bwd_orig > 0 and bwd_opt > 0:
        speedup = bwd_orig / bwd_opt
        print(f"\nBackward Pass Speedup: {speedup:.2f}x")
        if speedup > 1:
            print(f"✅ Custom backward is {speedup:.2f}x faster!")
        else:
            print(f"❌ Custom backward is {1 / speedup:.2f}x slower")

    if bwd_std > 0 and bwd_opt > 0:
        print("\nVs Standard Dilated Attention:")
        print(f"  Optimized Hilbert backward: {bwd_opt:.2f}ms")
        print(f"  Standard Dilated backward:  {bwd_std:.2f}ms")
        print(f"  Difference: {abs(bwd_opt - bwd_std):.2f}ms")


if __name__ == "__main__":
    main()
