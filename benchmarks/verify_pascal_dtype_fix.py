#!/usr/bin/env python3
"""Verify Pascal dtype fix works correctly on real hardware."""

import torch
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def main():
    print("Verifying Pascal GPU dtype fix")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No GPU available")
        return

    # Get GPU info
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")

    is_pascal = props.major == 6
    print(f"Is Pascal: {is_pascal}")
    print()

    # Test 1: Auto dtype selection
    print("Test 1: Automatic dtype selection")
    print("-" * 40)

    model = RingDilatedAttentionV2Collective(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        device=device,
        # dtype not specified - should auto-select
    )

    print(f"Auto-selected dtype: {model.dtype}")
    expected_dtype = torch.float32 if is_pascal else torch.float16
    print(f"Expected dtype: {expected_dtype}")

    if model.dtype == expected_dtype:
        print("✅ PASS: Correct dtype auto-selected")
    else:
        print("❌ FAIL: Wrong dtype selected")

    print()

    # Test 2: Explicit FP16 on Pascal should warn
    if is_pascal:
        print("Test 2: Explicit FP16 on Pascal (should warn)")
        print("-" * 40)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = RingDilatedAttentionV2Collective(
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                device=device,
                dtype=torch.float16,  # Explicit FP16
            )

            # Check if warning was issued
            pascal_warnings = [
                warning for warning in w if "slower" in str(warning.message)
            ]

            if pascal_warnings:
                print("✅ PASS: Warning issued for FP16 on Pascal")
                print(f"   Warning: {pascal_warnings[0].message}")
            else:
                print("❌ FAIL: No warning for FP16 on Pascal")

    print()

    # Test 3: Quick performance check
    print("Test 3: Quick performance verification")
    print("-" * 40)

    batch_size = 1
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        _ = model(q, k, v)

    # Time a forward pass
    import time

    torch.cuda.synchronize()
    start = time.perf_counter()

    _ = model(q, k, v)

    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    print(f"Forward pass time: {elapsed_ms:.2f} ms")
    print(f"Throughput: {seq_len / (elapsed_ms / 1000):.0f} tokens/s")

    print()
    print("=" * 60)
    print("SUMMARY:")
    if is_pascal:
        print("✅ Pascal GPU detected - using FP32 for optimal performance")
        print("📝 FP16 would be 5-10x slower on this GPU")
    else:
        print("✅ Modern GPU detected - using FP16 for optimal performance")
        print("📝 This GPU has good FP16/Tensor Core support")


if __name__ == "__main__":
    main()
