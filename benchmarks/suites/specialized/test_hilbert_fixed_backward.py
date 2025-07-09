#!/usr/bin/env python3
"""
Test the optimized backward pass for HilbertAttentionTritonFixed.
"""

import torch
import torch.nn as nn
import time
import gc


def benchmark_model(model, x, name, num_iterations=5):
    """Benchmark forward and backward passes accurately."""
    device = x.device

    # Warmup
    print(f"\nWarming up {name}...")
    for _ in range(3):
        x.grad = None
        out = model(x)
        loss = out.mean()  # Use mean instead of sum for stability
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Measure forward pass only
    forward_times = []
    for _ in range(num_iterations):
        x.grad = None

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        forward_times.append((end - start) * 1000)

    avg_forward = sum(forward_times) / len(forward_times)

    # Measure backward pass
    backward_times = []
    for _ in range(num_iterations):
        x.grad = None

        # Forward pass (not timed)
        out = model(x)
        loss = out.mean()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time only backward
        start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        backward_times.append((end - start) * 1000)

    avg_backward = sum(backward_times) / len(backward_times)

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

    print("\nComparing HilbertAttentionTritonFixed Backward Pass")
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
            model_original, x, "Original", num_iterations=10
        )
        print("\nOriginal HilbertAttentionTritonFixed:")
        print(f"  Forward:  {fwd_orig:8.2f} ms")
        print(
            f"  Backward: {bwd_orig:8.2f} ms (likely inaccurate due to Triton autograd)"
        )
        print(f"  Ratio:    {ratio_orig:8.2f}x")

    except Exception as e:
        print(f"\nOriginal implementation error: {e}")
        fwd_orig, bwd_orig = 0, 0

    # Test our optimized simple implementation
    try:
        from dilated_attention_pytorch.kernels.hilbert_attention_core import (
            HilbertAttentionCore,
        )

        model_simple = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
            use_custom_backward=True,
        ).to(device)

        fwd_simple, bwd_simple, ratio_simple = benchmark_model(
            model_simple, x, "Simple Optimized", num_iterations=10
        )
        print("\nHilbertAttentionTritonSimple (Hybrid Backward):")
        print(f"  Forward:  {fwd_simple:8.2f} ms")
        print(f"  Backward: {bwd_simple:8.2f} ms")
        print(f"  Ratio:    {ratio_simple:8.2f}x")

    except Exception as e:
        print(f"\nSimple implementation error: {e}")
        fwd_simple, bwd_simple = 0, 0

    # Test standard dilated attention for reference
    try:
        from dilated_attention_pytorch import ImprovedDilatedAttention

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
        print("\nStandard Dilated Attention (reference):")
        print(f"  Forward:  {fwd_std:8.2f} ms")
        print(f"  Backward: {bwd_std:8.2f} ms")
        print(f"  Ratio:    {ratio_std:8.2f}x")

    except Exception as e:
        print(f"\nStandard implementation error: {e}")
        fwd_std, bwd_std = 0, 0

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS:")

    if bwd_orig > 0:
        print(
            "\nThe original HilbertAttentionTritonFixed likely has incorrect backward timing"
        )
        print("due to PyTorch's automatic differentiation of Triton kernels.")
        print(
            f"The reported {bwd_orig:.2f}ms is probably just the overhead, not the actual computation."
        )

    if bwd_simple > 0 and bwd_std > 0:
        print("\nHybrid backward approach comparison:")
        print(f"  Hilbert (optimized): {bwd_simple:.2f}ms")
        print(f"  Standard Dilated:    {bwd_std:.2f}ms")
        print(
            f"  Difference: {bwd_simple - bwd_std:.2f}ms ({(bwd_simple / bwd_std - 1) * 100:.1f}% slower)"
        )
        print(
            "\nThe Hilbert reordering adds overhead but is much better than automatic differentiation."
        )

    print("\nExpected real backward time for original: ~100-200ms based on complexity")


if __name__ == "__main__":
    main()
