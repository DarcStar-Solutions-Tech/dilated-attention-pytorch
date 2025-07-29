#!/usr/bin/env python3
"""
Investigate the suspiciously fast 4K d=4 performance (0.09ms).
This seems unrealistically fast - need to verify computation is actually happening.
"""

import torch
import time
import sys
import numpy as np

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def detailed_benchmark(model, x, name, warmup=5, runs=20):
    """Detailed benchmarking with multiple metrics."""
    _ = x.device

    print(f"\n=== {name} ===")

    # Warmup
    outputs = []
    for _ in range(warmup):
        torch.cuda.synchronize()
        out = model(x)
        outputs.append(out)
        torch.cuda.synchronize()

    # Verify outputs are changing (not cached)
    if len(outputs) > 1:
        diff = (outputs[0] - outputs[1]).abs().max().item()
        print(f"Output variance between runs: {diff:.6f}")
        if diff < 1e-7:
            print("WARNING: Outputs identical between runs!")

    # Detailed timing
    times = []
    for i in range(runs):
        torch.cuda.synchronize()

        # Use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            out = model(x)
        end_event.record()

        torch.cuda.synchronize()

        # Get time in milliseconds
        cuda_time = start_event.elapsed_time(end_event)
        times.append(cuda_time)

        if i == 0:
            first_out = out.clone()

    times = np.array(times)

    print("CUDA Event Timing:")
    print(f"  Mean: {times.mean():.3f}ms")
    print(f"  Std: {times.std():.3f}ms")
    print(f"  Min: {times.min():.3f}ms")
    print(f"  Max: {times.max():.3f}ms")

    # Alternative timing method
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            out = model(x)
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000

    print(f"Wall clock timing: {total_time / runs:.3f}ms per iteration")

    return first_out, times.mean()


def compute_theoretical_flops(seq_len, hidden_dim, num_heads, dilation_rate):
    """Compute theoretical FLOPS for attention."""
    head_dim = hidden_dim // num_heads
    batch_size = 2

    # For dilated attention with d=4, only 1/4 of positions are active
    effective_seq_len = seq_len // dilation_rate

    # QKV projection: 3 * (B * seq_len * hidden_dim * hidden_dim)
    qkv_flops = 3 * batch_size * seq_len * hidden_dim * hidden_dim * 2

    # Attention computation per head: (B * H * seq_len * effective_seq_len * head_dim * 2)
    # This is Q @ K^T
    attention_flops = (
        batch_size * num_heads * seq_len * effective_seq_len * head_dim * 2
    )

    # Softmax: ~5 ops per element
    softmax_flops = batch_size * num_heads * seq_len * effective_seq_len * 5

    # Attention @ V: same as attention computation
    av_flops = attention_flops

    # Output projection
    out_flops = batch_size * seq_len * hidden_dim * hidden_dim * 2

    total_flops = qkv_flops + attention_flops + softmax_flops + av_flops + out_flops

    print("\nTheoretical FLOPS breakdown:")
    print(f"  QKV projection: {qkv_flops / 1e9:.2f} GFLOPS")
    print(f"  Attention (Q@K^T): {attention_flops / 1e9:.2f} GFLOPS")
    print(f"  Softmax: {softmax_flops / 1e9:.2f} GFLOPS")
    print(f"  Attention @ V: {av_flops / 1e9:.2f} GFLOPS")
    print(f"  Output projection: {out_flops / 1e9:.2f} GFLOPS")
    print(f"  Total: {total_flops / 1e9:.2f} GFLOPS")

    return total_flops


def check_computation_correctness():
    """Verify that computation is actually happening correctly."""

    print("=== Checking 4K d=4 Computation Correctness ===")

    # Parameters
    seq_len = 4096
    dilation_rate = 4
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    # Create models
    unified = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    enhanced = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            enable_sparse_optimization=True,
        )
        .cuda()
        .eval()
    )

    # Test with different inputs
    print("\n1. Testing with random input:")
    x_random = torch.randn(
        batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
    )

    with torch.no_grad():
        out_unified = unified(x_random)
        out_enhanced = enhanced(x_random)

    diff = (out_unified - out_enhanced).abs()
    print(f"   Max difference: {diff.max().item():.6f}")
    print(f"   Mean difference: {diff.mean().item():.6f}")
    print(f"   Output norm unified: {out_unified.norm().item():.2f}")
    print(f"   Output norm enhanced: {out_enhanced.norm().item():.2f}")

    # Test with structured input
    print("\n2. Testing with structured input (ones):")
    x_ones = torch.ones(
        batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
    )

    with torch.no_grad():
        out_unified_ones = unified(x_ones)
        out_enhanced_ones = enhanced(x_ones)

    print(f"   Output norm unified: {out_unified_ones.norm().item():.2f}")
    print(f"   Output norm enhanced: {out_enhanced_ones.norm().item():.2f}")

    # Check if output changes with input
    print("\n3. Testing input sensitivity:")
    x_perturbed = x_random + 0.01 * torch.randn_like(x_random)

    with torch.no_grad():
        out_perturbed = enhanced(x_perturbed)

    sensitivity = (out_perturbed - out_enhanced).abs().mean().item()
    print(f"   Output change from 1% input change: {sensitivity:.6f}")

    if sensitivity < 1e-6:
        print("   WARNING: Output insensitive to input changes!")

    # Check actual operations performed
    print("\n4. Checking actual operations:")

    # Get config used
    config = enhanced._get_optimal_config(seq_len)
    print(f"   Block config: {config['block_m']}x{config['block_n']}")
    print(f"   Use fused softmax: {config.get('use_fused_softmax', True)}")

    # Check which path is used
    M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
    use_pytorch = (
        M_padded <= 512
        or not enhanced._triton_available
        or (
            enhanced.enable_sparse_optimization
            and dilation_rate >= 4
            and M_padded <= 2048
        )
    )

    print(f"   Uses PyTorch path: {use_pytorch}")
    print(f"   M_padded: {M_padded}")

    # Measure FLOPS
    print("\n5. Performance Analysis:")

    # Theoretical FLOPS
    total_flops = compute_theoretical_flops(
        seq_len, hidden_dim, num_heads, dilation_rate
    )

    # Measure actual time
    _, enhanced_time = detailed_benchmark(enhanced, x_random, "Enhanced 4K d=4")

    # Calculate TFLOPS
    tflops = (total_flops / 1e12) / (enhanced_time / 1000)
    print(f"\nAchieved performance: {tflops:.2f} TFLOPS")

    # GTX 1080 theoretical peak: ~9 TFLOPS FP32
    print(f"Percentage of theoretical peak: {(tflops / 9) * 100:.1f}%")

    if tflops > 9:
        print("WARNING: Exceeding theoretical peak performance!")


def test_different_configs():
    """Test various configurations to find anomalies."""

    print("\n\n=== Testing Different Configurations ===")

    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    configs = [
        (2048, 4, "2K d=4"),
        (4096, 1, "4K d=1"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (4096, 8, "4K d=8"),
        (8192, 4, "8K d=4"),
    ]

    for seq_len, dilation_rate, desc in configs:
        print(f"\n{desc}:")

        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                enable_sparse_optimization=True,
            )
            .cuda()
            .eval()
        )

        x = torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
        )

        # Quick timing
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = enhanced(x)
        end.record()

        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end)

        # Calculate effective work
        effective_seq = seq_len // dilation_rate
        print(f"  Time: {time_ms:.3f}ms")
        print(f"  Effective sequence length: {effective_seq}")
        print(
            f"  Time per effective position: {(time_ms * 1000) / effective_seq:.1f} μs"
        )


def main():
    print("=== Investigating 4K d=4 Performance Anomaly ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Run investigations
    check_computation_correctness()
    test_different_configs()

    print("\n\n=== Summary ===")
    print("If the 0.09ms timing is real, it would mean:")
    print("- 90 microseconds for full attention computation")
    print("- Over 100% of theoretical GPU peak performance")
    print("- Faster than just memory transfer time")
    print("\nThis suggests either:")
    print("1. The measurement is incorrect")
    print("2. A shortcut/optimization is being taken")
    print("3. The computation is incomplete")


if __name__ == "__main__":
    main()
