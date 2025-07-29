#!/usr/bin/env python3
"""
Debug why some sparse configurations regressed.
"""

import torch
import time
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def detailed_benchmark(seq_len, dilation_rate):
    """Detailed benchmark of a specific configuration."""

    print(f"\n=== Debugging {seq_len} tokens, dilation={dilation_rate} ===")

    # Parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

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
        )
        .cuda()
        .eval()
    )

    # Get configuration
    config = enhanced._get_optimal_config(seq_len)
    effective_len = seq_len // dilation_rate

    print(f"Effective length: {effective_len}")
    print(f"Enhanced config: {config}")

    # Check if Hilbert is being used
    print(f"Hilbert threshold: {enhanced.hilbert_threshold}")
    print(f"Will use Hilbert: {seq_len > enhanced.hilbert_threshold}")

    # Create different input sizes to test
    test_sizes = [(1, seq_len), (batch_size, seq_len)]

    for bs, sl in test_sizes:
        print(f"\nTesting batch_size={bs}, seq_len={sl}:")
        x = torch.randn(bs, sl, hidden_dim, device="cuda", dtype=torch.float32)

        # Test both models
        for name, model in [("Unified", unified), ("Enhanced", enhanced)]:
            try:
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(x)
                torch.cuda.synchronize()

                # Time
                times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = model(x)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)

                avg_time = sum(times) / len(times)
                print(
                    f"  {name}: {avg_time:.2f}ms (min: {min(times):.2f}, max: {max(times):.2f})"
                )

            except Exception as e:
                print(f"  {name}: ERROR - {str(e)}")

    # Test with different dtypes
    print("\nTesting with float16:")
    x_fp16 = torch.randn(
        batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16
    )

    for name, model in [("Unified", unified), ("Enhanced", enhanced)]:
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    start = time.perf_counter()
                    _ = model(x_fp16)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
            print(f"  {name}: {elapsed:.2f}ms")
        except Exception as e:
            print(f"  {name}: ERROR - {str(e)}")


def check_kernel_paths():
    """Check which code paths are being used."""

    print("\n=== Checking Code Paths ===")

    # Test configurations that performed poorly
    poor_configs = [
        (4096, 2, "4K d=2"),
        (8192, 2, "8K d=2"),
    ]

    for seq_len, dilation_rate, desc in poor_configs:
        print(f"\n{desc}:")

        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=dilation_rate,
        ).cuda()

        # Check conditions
        M_padded = ((seq_len + 128 - 1) // 128) * 128
        use_hilbert = M_padded > enhanced.hilbert_threshold
        use_pytorch = M_padded <= 512 or not enhanced._triton_available

        print(f"  Padded length: {M_padded}")
        print(f"  Hilbert threshold: {enhanced.hilbert_threshold}")
        print(f"  Will use Hilbert: {use_hilbert}")
        print(f"  Will use PyTorch path: {use_pytorch}")
        print(f"  Triton available: {enhanced._triton_available}")

        # Check segment calculations
        num_segments = (M_padded + 128 - 1) // 128
        print(f"  Number of segments: {num_segments}")

        for seg_idx in range(min(2, num_segments)):
            seg_start = seg_idx * 128
            seg_end = min(seg_start + 128, M_padded)
            num_active = (seg_end - seg_start + dilation_rate - 1) // dilation_rate
            print(
                f"  Segment {seg_idx}: positions {seg_start}-{seg_end}, active positions: {num_active}"
            )


def main():
    print("=== Debugging Sparse Pattern Performance Regression ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Debug the problematic configurations
    problematic = [
        (4096, 2),  # 13.5x slower!
        (8192, 2),  # 6.16x slower!
    ]

    for seq_len, dilation_rate in problematic:
        detailed_benchmark(seq_len, dilation_rate)

    check_kernel_paths()

    # Compare with good performers
    print("\n\n=== Comparing with Good Performers ===")
    good = [
        (4096, 4),  # 0.88x (faster than Unified!)
        (8192, 4),  # 0.63x (faster than Unified!)
    ]

    for seq_len, dilation_rate in good:
        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=dilation_rate,
        ).cuda()

        config = enhanced._get_optimal_config(seq_len)
        effective_len = seq_len // dilation_rate

        print(f"\n{seq_len} d={dilation_rate} (GOOD):")
        print(f"  Effective length: {effective_len}")
        print(
            f"  Config: block={config['block_m']}x{config['block_n']}, fused={config['use_fused_softmax']}"
        )


if __name__ == "__main__":
    main()
