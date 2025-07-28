#!/usr/bin/env python3
"""
Demonstrate Flash Attention integration with GPU architecture awareness.

This script shows:
1. Automatic GPU detection and backend selection
2. Performance comparison across different GPUs
3. Memory efficiency improvements
4. Automatic fallback for unsupported hardware
"""

import torch
import time
import numpy as np
from typing import Dict

# Import utilities
from dilated_attention_pytorch.utils import (
    get_flash_attention_support,
    get_gpu_compute_capability,
)

# Import attention implementations
from dilated_attention_pytorch import (
    RingDilatedAttentionV2Collective,
)


def print_gpu_info():
    """Print GPU information and Flash Attention support."""
    print("GPU Information")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No GPU available")
        return

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    capability = get_gpu_compute_capability(device)

    print(f"GPU: {props.name}")
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")

    # Check Flash Attention support
    flash_support = get_flash_attention_support(device)

    print("\nFlash Attention Support:")
    print(f"  Architecture: {flash_support['gpu_architecture']}")
    print(f"  Flash Attention Available: {flash_support['has_flash_attn']}")
    print(f"  Flash Attention 2: {flash_support['has_flash_attn_2']}")
    print(f"  Flash Attention 3: {flash_support['has_flash_attn_3']}")
    print(f"  Recommended Backend: {flash_support['recommended_backend']}")
    print(f"  Supports FP8: {flash_support['supports_fp8']}")


def benchmark_attention(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark attention model."""
    device = model.device
    dtype = model.dtype

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Memory stats
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0

    return {
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "peak_memory_mb": peak_memory,
        "throughput": seq_len / (np.mean(times) / 1000),
    }


def compare_implementations():
    """Compare standard vs Flash Attention implementations."""
    print("\n\nPerformance Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("GPU required for comparison")
        return

    device = torch.device("cuda:0")

    # Configuration
    batch_size = 2
    seq_len = 4096
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048]
    dilation_rates = [1, 2]

    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")

    # Standard implementation
    print("\n1. Standard RingDilatedAttentionV2Collective:")
    model_standard = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        ring_size=1,
    )

    results_standard = benchmark_attention(
        model_standard, batch_size, seq_len, num_heads, head_dim
    )

    print(
        f"  Time: {results_standard['mean_time_ms']:.2f} ± {results_standard['std_time_ms']:.2f} ms"
    )
    print(f"  Memory: {results_standard['peak_memory_mb']:.1f} MB")
    print(f"  Throughput: {results_standard['throughput']:.0f} tokens/s")

    # Flash Attention implementation
    print("\n2. Flash-optimized RingDilatedAttentionV2Collective:")
    try:
        model_flash = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            ring_size=1,
            use_flash_attention=True,
        )

        print(f"  Using backend: {model_flash.flash_backend}")

        results_flash = benchmark_attention(
            model_flash, batch_size, seq_len, num_heads, head_dim
        )

        print(
            f"  Time: {results_flash['mean_time_ms']:.2f} ± {results_flash['std_time_ms']:.2f} ms"
        )
        print(f"  Memory: {results_flash['peak_memory_mb']:.1f} MB")
        print(f"  Throughput: {results_flash['throughput']:.0f} tokens/s")

        # Calculate improvements
        speedup = results_standard["mean_time_ms"] / results_flash["mean_time_ms"]
        memory_reduction = 1 - (
            results_flash["peak_memory_mb"] / results_standard["peak_memory_mb"]
        )

        print(f"\n  Speedup: {speedup:.2f}x")
        print(f"  Memory reduction: {memory_reduction * 100:.1f}%")

    except Exception as e:
        print(f"  Error: {e}")
        print("  Flash Attention not available on this system")


def test_long_sequences():
    """Test with very long sequences to show memory efficiency."""
    print("\n\nLong Sequence Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("GPU required for long sequence test")
        return

    device = torch.device("cuda:0")

    # Try increasingly long sequences
    seq_lengths = [8192, 16384, 32768]

    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        try:
            model = RingDilatedAttentionV2Collective(
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                device=device,
                ring_size=1,
                use_flash_attention=True,
                flash_chunk_size=2048,  # Process in chunks to avoid OOM
            )

            # Small batch to manage memory
            batch_size = 1
            num_heads = 8
            head_dim = 64

            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output = model(q, k, v, is_causal=False)

            torch.cuda.synchronize()
            end = time.perf_counter()

            time_ms = (end - start) * 1000
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2

            print("  ✓ Success!")
            print(f"  Time: {time_ms:.2f} ms")
            print(f"  Memory: {memory_mb:.1f} MB")
            print(f"  Throughput: {seq_len / (time_ms / 1000):.0f} tokens/s")

        except torch.cuda.OutOfMemoryError:
            print("  ✗ OOM - sequence too long for available memory")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Clear memory
        if "model" in locals():
            del model
        if "q" in locals():
            del q, k, v
        torch.cuda.empty_cache()


def main():
    """Run Flash Attention demonstration."""
    print("Flash Attention Integration Demo")
    print("=" * 80)

    # Print GPU info
    print_gpu_info()

    # Compare implementations
    compare_implementations()

    # Test long sequences
    test_long_sequences()

    print("\n\nKey Benefits:")
    print("=" * 60)
    print("1. Automatic GPU detection and optimal backend selection")
    print("2. Significant speedup on supported GPUs (Turing+)")
    print("3. Memory efficiency for long sequences")
    print("4. Graceful fallback on older GPUs (Pascal)")
    print("5. Support for Flash Attention 3 on H100 GPUs")


if __name__ == "__main__":
    main()
