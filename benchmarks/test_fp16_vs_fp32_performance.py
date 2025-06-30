#!/usr/bin/env python3
"""
Test FP16 vs FP32 performance on GTX 1080 to understand the slowdown.

The GTX 1080 (Pascal architecture) has limited FP16 support compared to newer GPUs.
"""

import torch
import time


def test_compute_performance():
    """Test raw compute performance for FP16 vs FP32."""
    device = torch.device("cuda")

    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(
        f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # Check FP16 support
    print("\nFP16 Support:")
    print(
        f"  Has Tensor Cores: {'Yes' if torch.cuda.get_device_capability(0)[0] >= 7 else 'No'}"
    )
    print(
        f"  Native FP16: {'Yes' if torch.cuda.get_device_capability(0)[0] >= 5 and torch.cuda.get_device_capability(0)[1] >= 3 else 'Limited'}"
    )

    # Test matrix multiplication performance
    print("\n" + "=" * 60)
    print("Matrix Multiplication Performance (GEMM)")
    print("=" * 60)

    sizes = [1024, 2048, 4096]

    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")

        # FP32 test
        a_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
        b_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = torch.matmul(a_fp32, b_fp32)

        torch.cuda.synchronize()
        start = time.perf_counter()

        num_iters = 100
        for _ in range(num_iters):
            _ = torch.matmul(a_fp32, b_fp32)

        torch.cuda.synchronize()
        fp32_time = (time.perf_counter() - start) / num_iters * 1000

        # Calculate TFLOPS
        flops = 2 * size * size * size  # For matrix multiplication
        fp32_tflops = flops / (fp32_time / 1000) / 1e12

        # FP16 test
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()

        # Warmup
        for _ in range(10):
            _ = torch.matmul(a_fp16, b_fp16)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(num_iters):
            _ = torch.matmul(a_fp16, b_fp16)

        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / num_iters * 1000

        fp16_tflops = flops / (fp16_time / 1000) / 1e12

        print(f"  FP32: {fp32_time:.2f} ms ({fp32_tflops:.2f} TFLOPS)")
        print(f"  FP16: {fp16_time:.2f} ms ({fp16_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {fp32_time / fp16_time:.2f}x")

        # Memory usage
        fp32_memory = (3 * size * size * 4) / (1024**2)  # 3 matrices, 4 bytes per float
        fp16_memory = (3 * size * size * 2) / (1024**2)  # 3 matrices, 2 bytes per half
        print(f"  Memory - FP32: {fp32_memory:.1f} MB, FP16: {fp16_memory:.1f} MB")

    # Test specific operations that might be slow
    print("\n" + "=" * 60)
    print("Operation-Specific Performance")
    print("=" * 60)

    size = 2048
    a_fp32 = torch.randn(1, size, 8, 64, device=device, dtype=torch.float32)
    a_fp16 = a_fp32.half()

    operations = [
        ("Softmax", lambda x: torch.softmax(x, dim=-1)),
        ("Exponential", lambda x: torch.exp(x)),
        ("Division", lambda x: x / 8.0),
        ("Transpose", lambda x: x.transpose(-2, -1)),
    ]

    for op_name, op_func in operations:
        print(f"\n{op_name}:")

        # FP32
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = op_func(a_fp32)
        torch.cuda.synchronize()
        fp32_time = (time.perf_counter() - start) / 100 * 1000

        # FP16
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = op_func(a_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / 100 * 1000

        print(f"  FP32: {fp32_time:.2f} ms")
        print(f"  FP16: {fp16_time:.2f} ms")
        print(f"  Speedup: {fp32_time / fp16_time:.2f}x")

    # Test mixed precision conversions
    print("\n" + "=" * 60)
    print("Type Conversion Overhead")
    print("=" * 60)

    x_fp32 = torch.randn(1024, 1024, device=device, dtype=torch.float32)

    # Test conversion cost
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        x_fp16 = x_fp32.half()
        _ = x_fp16.float()
    torch.cuda.synchronize()
    conversion_time = (time.perf_counter() - start) / 1000 * 1000

    print(f"FP32 -> FP16 -> FP32 conversion: {conversion_time:.2f} ms")

    # GTX 1080 specific information
    print("\n" + "=" * 60)
    print("GTX 1080 (Pascal) Architecture Limitations")
    print("=" * 60)
    print("1. No Tensor Cores (introduced in Volta/Turing)")
    print("2. FP16 compute rate: 1/32 of FP32 (vs 2x on modern GPUs)")
    print("3. Limited FP16 ALU support")
    print("4. FP16 storage saves memory bandwidth but not compute")
    print("5. Many operations fall back to FP32 internally")

    # Theoretical vs actual performance
    fp32_peak = 8.9  # TFLOPS for GTX 1080
    fp16_peak = fp32_peak / 32  # Pascal limitation

    print("\nTheoretical Peak Performance:")
    print(f"  FP32: {fp32_peak:.1f} TFLOPS")
    print(f"  FP16: {fp16_peak:.1f} TFLOPS (1/32 of FP32)")
    print("\nThis explains why FP16 is slower on GTX 1080!")


if __name__ == "__main__":
    test_compute_performance()
