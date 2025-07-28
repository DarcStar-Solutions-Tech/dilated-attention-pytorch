#!/usr/bin/env python3
"""
Test script for GPU utilities and backend benchmarking.

This script demonstrates:
1. GPU detection and architecture identification
2. Optimal dtype selection based on GPU
3. Backend availability checking
4. Benchmark different attention backends
5. Integration with Hilbert attention
"""

import torch
import logging
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_gpu_detection():
    """Test GPU detection capabilities."""
    print("\n" + "=" * 60)
    print("GPU DETECTION TEST")
    print("=" * 60)

    from dilated_attention_pytorch.utils import get_gpu_info

    # Test on default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_info(device)

    print(f"\nDevice: {device}")
    print(f"GPU Name: {gpu_info.name}")
    print(f"Architecture: {gpu_info.architecture}")
    print(f"Compute Capability: {gpu_info.compute_capability}")
    print(f"Total Memory: {gpu_info.total_memory_gb:.2f} GB")
    print(f"Available Memory: {gpu_info.available_memory_gb:.2f} GB")
    print(f"Memory Bandwidth: {gpu_info.memory_bandwidth_gbps:.0f} GB/s")
    print(f"CUDA Cores: {gpu_info.cuda_cores}")
    print(f"Tensor Cores: {gpu_info.tensor_cores}")
    print(f"FP32 Performance: {gpu_info.fp32_performance_tflops:.1f} TFLOPS")
    print(f"FP16 Performance: {gpu_info.fp16_performance_tflops:.1f} TFLOPS")
    print(f"Supports FP8: {gpu_info.supports_fp8}")
    print(f"Supports BF16: {gpu_info.supports_bf16}")
    print(f"Optimal dtype: {gpu_info.optimal_dtype}")
    print(f"Optimal block size: {gpu_info.optimal_block_size}")

    print("\nBackend Support:")
    print(f"  Has Flash Attention: {gpu_info.has_flash_attn}")
    print(f"  Has Flash Attention 2: {gpu_info.has_flash_attn_2}")
    print(f"  Has Flash Attention 3: {gpu_info.has_flash_attn_3}")
    print(f"  Has xformers: {gpu_info.has_xformers}")
    print(f"  Has SDPA: {gpu_info.has_sdpa}")
    print(f"  Recommended backend: {gpu_info.recommended_backend}")


def test_backend_selection():
    """Test backend selection for different configurations."""
    print("\n" + "=" * 60)
    print("BACKEND SELECTION TEST")
    print("=" * 60)

    from dilated_attention_pytorch.utils import select_gpu_attention_backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different configurations
    configs = [
        {"seq_len": 128, "desc": "Short sequence"},
        {"seq_len": 1024, "desc": "Medium sequence"},
        {"seq_len": 16384, "desc": "Long sequence"},
        {"seq_len": 1024, "has_custom_mask": True, "desc": "With custom mask"},
        {"seq_len": 1024, "is_causal": True, "desc": "Causal attention"},
        {"seq_len": 1024, "use_dilation": True, "desc": "With dilation"},
    ]

    print("\nBackend selection for different configurations:")
    for config in configs:
        desc = config.pop("desc")
        backend = select_gpu_attention_backend(device=device, **config)
        print(f"  {desc}: {backend}")


def benchmark_attention_backends(
    batch_size: int = 2,
    seq_len: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
):
    """Benchmark different attention backends."""
    print("\n" + "=" * 60)
    print("ATTENTION BACKEND BENCHMARKS")
    print("=" * 60)

    from dilated_attention_pytorch.utils import (
        benchmark_attention_backends as benchmark_backends,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nBenchmarking with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")

    results = benchmark_backends(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    print("\nResults (ms per forward pass):")
    for backend, time_ms in sorted(results.items(), key=lambda x: x[1]):
        if time_ms < float("inf"):
            print(f"  {backend:20s}: {time_ms:8.2f} ms")
        else:
            print(f"  {backend:20s}: FAILED")

    # Find speedup vs manual
    if "standard" in results and results["standard"] < float("inf"):
        manual_time = results["standard"]
        print("\nSpeedup vs manual implementation:")
        for backend, time_ms in sorted(results.items(), key=lambda x: x[1]):
            if time_ms < float("inf") and backend != "standard":
                speedup = manual_time / time_ms
                print(f"  {backend:20s}: {speedup:5.1f}x")


def test_hilbert_gpu_attention():
    """Test Hilbert attention with GPU optimization."""
    print("\n" + "=" * 60)
    print("HILBERT GPU-OPTIMIZED ATTENTION TEST")
    print("=" * 60)

    try:
        from dilated_attention_pytorch.ring_dilated_attention_hilbert_gpu_optimized import (
            RingDilatedAttentionHilbertGPUOptimized,
        )
    except ImportError:
        print("Could not import RingDilatedAttentionHilbertGPUOptimized")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    batch_size = 2
    seq_len = 4096
    embed_dim = 768
    num_heads = 12
    segment_lengths = [1024, 2048, 1024]
    dilation_rates = [1, 2, 4]

    print("\nTesting with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Segments: {segment_lengths}")
    print(f"  Dilation rates: {dilation_rates}")

    # Create model
    model = RingDilatedAttentionHilbertGPUOptimized(
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        benchmark_backends=True,  # This will benchmark on init
    ).to(device)

    print(f"\nModel created with backend: {model.attention_backend}")
    print(f"Using dtype: {model.dtype}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=model.dtype)

    # Warmup
    for _ in range(3):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time forward pass
    num_iterations = 10
    start = time.perf_counter()

    for _ in range(num_iterations):
        output = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    avg_time = (end - start) / num_iterations * 1000  # ms

    print(f"\nForward pass time: {avg_time:.2f} ms")
    print(f"Output shape: {output.shape}")

    # Test with causal masking
    output_causal = model(x, is_causal=True)
    print(f"\nCausal output shape: {output_causal.shape}")


def test_dtype_performance():
    """Test performance with different dtypes."""
    print("\n" + "=" * 60)
    print("DTYPE PERFORMANCE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping dtype test")
        return

    from dilated_attention_pytorch.utils import get_gpu_info

    gpu_info = get_gpu_info(device)

    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    # Dtypes to test
    dtypes_to_test = [torch.float32, torch.float16]
    if gpu_info.supports_bf16:
        dtypes_to_test.append(torch.bfloat16)

    print(f"\nTesting dtypes: {[str(dt) for dt in dtypes_to_test]}")

    results = {}

    for dtype in dtypes_to_test:
        # Create test tensors
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        num_iters = 100

        for _ in range(num_iters):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / num_iters * 1000  # ms
        results[dtype] = avg_time

    print("\nResults (ms per forward pass):")
    for dtype, time_ms in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {str(dtype):20s}: {time_ms:8.2f} ms")

    # Calculate speedup
    if torch.float32 in results:
        fp32_time = results[torch.float32]
        print("\nSpeedup vs FP32:")
        for dtype, time_ms in results.items():
            if dtype != torch.float32:
                speedup = fp32_time / time_ms
                print(f"  {str(dtype):20s}: {speedup:5.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Test GPU utilities and benchmarking")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--skip-benchmarks", action="store_true", help="Skip benchmarks"
    )

    args = parser.parse_args()

    # Run tests
    test_gpu_detection()
    test_backend_selection()

    if not args.skip_benchmarks:
        benchmark_attention_backends(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )
        test_dtype_performance()

    test_hilbert_gpu_attention()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
