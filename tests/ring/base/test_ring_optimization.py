"""
Test the optimization of RingDilatedAttention
"""

import time

import torch

from dilated_attention_pytorch import DilatedAttention
from dilated_attention_pytorch import (
    RingDilatedAttentionHilbertGPUOptimized as RingDilatedAttention,
)


def test_optimization():
    print("Testing RingDilatedAttention Optimization")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test parameters
    batch_size = 1
    num_heads = 8
    head_dim = 64

    test_configs = [
        (2048, "Short sequence"),
        (8192, "Medium sequence"),
        (32768, "Long sequence"),
    ]

    for seq_len, desc in test_configs:
        print(f"\n{desc} ({seq_len:,} tokens):")

        segments = [
            min(1024, seq_len // 4),
            min(2048, seq_len // 2),
            min(4096, seq_len),
        ]
        dilation_rates = [1, 2, 4]

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        implementations = [
            ("DilatedAttention", DilatedAttention(segments, dilation_rates, 0.0)),
            (
                "RingDilated (optimized)",
                RingDilatedAttention(segments, dilation_rates, 0.0, ring_size=1),
            ),
        ]

        results = []

        for name, module in implementations:
            module = module.to(device, dtype)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = module(q, k, v)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            iterations = 10
            for _ in range(iterations):
                with torch.no_grad():
                    output = module(q, k, v)

            torch.cuda.synchronize()
            elapsed = (time.time() - start) / iterations * 1000

            results.append((name, elapsed, output))
            print(f"  {name:25} {elapsed:8.2f}ms")

            del module

        # Compare results
        if len(results) == 2:
            _, time1, output1 = results[0]
            _, time2, output2 = results[1]

            speedup = time1 / time2
            print(f"  Speedup: {speedup:.2f}x")

            if torch.allclose(output1, output2, rtol=1e-3):
                print("  ✓ Results match")
            else:
                print("  ✗ Results differ!")

        # Cleanup
        del q, k, v
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("The optimization should show improvement, especially for sequences")
    print("where offset=0 is common (which happens for 1/3 of the groups)")


def analyze_offset_distribution():
    """Analyze how often offset=0 occurs"""
    print("\n\nOFFSET DISTRIBUTION ANALYSIS")
    print("=" * 80)

    dilation_rates = [1, 2, 4]
    num_groups = len(dilation_rates)

    print("With dilation_rates = [1, 2, 4]:")
    for i, r in enumerate(dilation_rates):
        offset = i % r
        print(
            f"  Group {i}: dilation={r}, offset={offset} {'<-- Direct slicing!' if offset == 0 else ''}"
        )

    # Count zero offsets
    zero_offsets = sum(1 for i, r in enumerate(dilation_rates) if i % r == 0)
    percentage = zero_offsets / num_groups * 100

    print(
        f"\nDirect slicing applicable: {zero_offsets}/{num_groups} groups ({percentage:.0f}%)"
    )
    print(
        "This means the optimization applies to a significant portion of computations!"
    )


if __name__ == "__main__":
    test_optimization()
    analyze_offset_distribution()
