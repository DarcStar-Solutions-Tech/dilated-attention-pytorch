#!/usr/bin/env python3
"""
Check dtypes used in benchmarks and verify fp32 usage.
"""

import torch


def check_module_dtypes():
    """Check default dtypes of our modules."""
    print("=== Checking Module Default Dtypes ===\n")

    # Import modules
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    # Create module
    module = HilbertAttentionCore(hidden_dim=256, num_heads=8, segment_size=128).cuda()

    print("Module weight dtypes:")
    print(f"  qkv_proj.weight: {module.qkv_proj.weight.dtype}")
    print(f"  out_proj.weight: {module.out_proj.weight.dtype}")

    # Create input and check what happens
    x_fp32 = torch.randn(1, 128, 256, device="cuda", dtype=torch.float32)
    x_fp16 = torch.randn(1, 128, 256, device="cuda", dtype=torch.float16)

    print("\nInput dtypes:")
    print(f"  x_fp32: {x_fp32.dtype}")
    print(f"  x_fp16: {x_fp16.dtype}")

    # Check outputs
    with torch.no_grad():
        out_fp32 = module(x_fp32)
        print(f"\nOutput dtype with fp32 input: {out_fp32.dtype}")

        try:
            out_fp16 = module(x_fp16)
            print(f"Output dtype with fp16 input: {out_fp16.dtype}")
        except Exception as e:
            print(f"Error with fp16 input: {e}")

    # Check PyTorch SDPA default
    print("\n=== PyTorch SDPA Default Dtypes ===")

    q = torch.randn(1, 8, 128, 32, device="cuda")  # Default dtype
    print(f"Default tensor dtype: {q.dtype}")

    with torch.no_grad():
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        print(f"SDPA output dtype: {out_sdpa.dtype}")

    # Check if benchmarks might be using mixed precision
    print("\n=== Checking for Mixed Precision ===")

    # Check if autocast affects our module
    with torch.cuda.amp.autocast():
        x_auto = torch.randn(1, 128, 256, device="cuda")
        print(f"Tensor dtype inside autocast: {x_auto.dtype}")

        # Our module inside autocast
        out_auto = module(x_auto)
        print(f"Module output dtype inside autocast: {out_auto.dtype}")


def benchmark_dtype_impact():
    """Benchmark the actual impact of dtype on performance."""
    print("\n=== Benchmarking Dtype Impact ===\n")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )
    import time

    # Create modules
    module_fp32 = HilbertAttentionCore(
        hidden_dim=256, num_heads=8, segment_size=128
    ).cuda()

    module_fp16 = (
        HilbertAttentionCore(hidden_dim=256, num_heads=8, segment_size=128)
        .cuda()
        .half()
    )

    # Test different sizes
    configs = [
        (1, 256, 256),
        (4, 512, 256),
    ]

    for batch, seq_len, hidden_dim in configs:
        print(f"\nConfig: batch={batch}, seq_len={seq_len}, hidden={hidden_dim}")

        # Create inputs
        x_fp32 = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.float32
        )
        x_fp16 = x_fp32.half()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = module_fp32(x_fp32)
                _ = module_fp16(x_fp16)

        # Time FP32
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = module_fp32(x_fp32)
        torch.cuda.synchronize()
        time_fp32 = (time.time() - start) / 50 * 1000

        # Time FP16
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = module_fp16(x_fp16)
        torch.cuda.synchronize()
        time_fp16 = (time.time() - start) / 50 * 1000

        print(f"  FP32: {time_fp32:.3f} ms")
        print(f"  FP16: {time_fp16:.3f} ms")
        print(f"  Speedup: {time_fp32 / time_fp16:.2f}x")

        # Also check PyTorch SDPA with different dtypes
        q_fp32 = x_fp32.view(batch, seq_len, 8, 32).transpose(1, 2)
        q_fp16 = x_fp16.view(batch, seq_len, 8, 32).transpose(1, 2)

        # Time PyTorch FP32
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q_fp32, q_fp32, q_fp32
                )
        torch.cuda.synchronize()
        time_pytorch_fp32 = (time.time() - start) / 50 * 1000

        # Time PyTorch FP16
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q_fp16, q_fp16, q_fp16
                )
        torch.cuda.synchronize()
        time_pytorch_fp16 = (time.time() - start) / 50 * 1000

        print(f"  PyTorch SDPA FP32: {time_pytorch_fp32:.3f} ms")
        print(f"  PyTorch SDPA FP16: {time_pytorch_fp16:.3f} ms")
        print(f"  PyTorch Speedup: {time_pytorch_fp32 / time_pytorch_fp16:.2f}x")


def check_triton_kernel_dtypes():
    """Check what dtypes Triton kernels are using internally."""
    print("\n=== Checking Triton Kernel Internal Dtypes ===\n")

    # Check the Triton kernel by looking at what it does
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    module = HilbertAttentionCore(hidden_dim=64, num_heads=4, segment_size=16).cuda()

    # Hook to check intermediate dtypes
    intermediate_dtypes = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            if isinstance(output, torch.Tensor):
                intermediate_dtypes[name] = output.dtype

        return hook

    # Register hooks
    module.qkv_proj.register_forward_hook(hook_fn("qkv_proj"))
    module.out_proj.register_forward_hook(hook_fn("out_proj"))

    # Test with fp32
    x = torch.randn(1, 32, 64, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out = module(x)

    print("With FP32 input:")
    for name, dtype in intermediate_dtypes.items():
        print(f"  {name} output: {dtype}")
    print(f"  Final output: {out.dtype}")


def main():
    """Run all dtype checks."""
    print("=== Dtype Analysis for Triton Benchmarks ===\n")

    check_module_dtypes()
    check_triton_kernel_dtypes()
    benchmark_dtype_impact()

    print("\n=== Summary ===")
    print("1. Default PyTorch tensors use float32")
    print("2. Our modules default to float32 weights")
    print("3. Mixed precision requires explicit .half() conversion")
    print("4. Benchmarks should be using float32 unless explicitly converted")


if __name__ == "__main__":
    main()
