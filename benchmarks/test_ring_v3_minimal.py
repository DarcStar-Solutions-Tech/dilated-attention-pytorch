#!/usr/bin/env python3
"""
Minimal test to debug Ring V3 performance.
Run with: python benchmarks/test_ring_v3_minimal.py
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_minimal():
    """Test basic functionality on single GPU."""

    print("Testing Ring V3 on Single GPU")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test different configurations
    configs = [
        (256, False, "Small, no bucketing"),
        (256, True, "Small, with bucketing"),
        (512, False, "Medium, no bucketing"),
        (512, True, "Medium, with bucketing"),
        (1024, False, "Large, no bucketing"),
    ]

    for seq_len, use_bucketed, desc in configs:
        print(f"\n{desc} (seq_len={seq_len}):")

        try:
            # Create model
            model = RingDilatedAttentionV3(
                segment_lengths=[seq_len // 2],
                dilation_rates=[1],
                bucket_size=128 if use_bucketed else None,
                use_bucketed=use_bucketed,
                device=device,
                dtype=torch.float32,
                ring_size=1,  # Single GPU
            )

            # Create inputs
            q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

            # Time forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()

            import time

            start = time.time()

            output = model(q, k, v, is_causal=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start

            # Check output
            has_nan = torch.isnan(output).any().item()
            output_mean = output.mean().item()

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Output: mean={output_mean:.6f}, NaN={has_nan}")

            if device.type == "cuda":
                mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
                print(f"  Memory: {mem_mb:.1f} MB")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n✅ Single GPU test completed")


if __name__ == "__main__":
    test_minimal()
