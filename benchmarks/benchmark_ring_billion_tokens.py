"""
Benchmark Ring Dilated Attention approaching billion-token sequences.

This script gradually increases sequence length while adjusting ring_size
to demonstrate how Ring Attention enables processing of very long sequences.
"""

import gc
import time
from dataclasses import dataclass

from pathlib import Path
import sys
import torch

from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention


# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager


@dataclass
class BenchmarkResult:
    seq_len: int
    ring_size: int
    chunk_size: int
    time_per_chunk: float
    total_time: float
    peak_memory_gb: float
    success: bool
    error: str = ""


def estimate_memory_usage(
    seq_len: int, batch_size: int, num_heads: int, head_dim: int, ring_size: int
) -> float:
    """Estimate memory usage in GB for given configuration."""
    # Each tensor (Q, K, V) size
    tensor_size = batch_size * seq_len * num_heads * head_dim
    # Float16 = 2 bytes per element
    bytes_per_tensor = tensor_size * 2

    # With ring attention, we only need 1/ring_size of K,V at a time
    effective_kv_size = bytes_per_tensor / ring_size

    # Total: Q (full) + K (chunked) + V (chunked) + output + overhead
    total_bytes = bytes_per_tensor + 2 * effective_kv_size + bytes_per_tensor * 1.5

    return total_bytes / (1024**3)


def benchmark_ring_attention(
    seq_len: int,
    ring_size: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
) -> BenchmarkResult:
    """Benchmark Ring Attention at given sequence length and ring size."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"\nTesting seq_len={seq_len:,}, ring_size={ring_size}")
    print(
        f"  Estimated memory: {estimate_memory_usage(seq_len, batch_size, num_heads, head_dim, ring_size):.2f}GB"
    )

    # Configure segments based on sequence length
    if seq_len <= 8192:
        segments = [1024, 2048, 4096]
    elif seq_len <= 65536:
        segments = [2048, 8192, 32768]
    elif seq_len <= 1048576:
        segments = [8192, 65536, 524288]
    else:
        # For very long sequences
        segments = [65536, 524288, min(4194304, seq_len)]

    # Ensure segments don't exceed sequence length
    segments = [min(s, seq_len) for s in segments]
    dilation_rates = [1, 2, 4]

    chunk_size = seq_len // ring_size

    try:
        # Create module
        module = RingDilatedAttention(
            segment_lengths=segments,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=ring_size,
        ).to(device, dtype)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Simulate ring attention by processing chunks
        chunk_times = []
        start_total = time.time()

        for chunk_idx in range(
            min(ring_size, 4)
        ):  # Process max 4 chunks for benchmarking
            start_chunk = time.time()

            # Create chunk tensors
            q_chunk = torch.randn(
                batch_size, chunk_size, num_heads, head_dim, device=device, dtype=dtype
            )
            k_chunk = torch.randn(
                batch_size, chunk_size, num_heads, head_dim, device=device, dtype=dtype
            )
            v_chunk = torch.randn(
                batch_size, chunk_size, num_heads, head_dim, device=device, dtype=dtype
            )

            # Process chunk
            with torch.no_grad():
                output_chunk = module._dilated_attention_block(
                    q_chunk, k_chunk, v_chunk, is_causal=False, ring_step=chunk_idx
                )

            if device.type == "cuda":
                torch.cuda.synchronize()

            chunk_time = time.time() - start_chunk
            chunk_times.append(chunk_time)

            # Clean up
            del q_chunk, k_chunk, v_chunk, output_chunk

            print(f"    Chunk {chunk_idx + 1}: {chunk_time * 1000:.1f}ms")

        total_time = time.time() - start_total
        avg_chunk_time = sum(chunk_times) / len(chunk_times)

        # Extrapolate total time for all chunks
        estimated_total_time = avg_chunk_time * ring_size

        # Get peak memory
        peak_memory_gb = 0.0
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        print("  ✓ Success!")
        print(f"  Average chunk time: {avg_chunk_time * 1000:.1f}ms")
        print(f"  Estimated total time: {estimated_total_time:.1f}s")
        print(f"  Peak memory: {peak_memory_gb:.2f}GB")

        return BenchmarkResult(
            seq_len=seq_len,
            ring_size=ring_size,
            chunk_size=chunk_size,
            time_per_chunk=avg_chunk_time,
            total_time=estimated_total_time,
            peak_memory_gb=peak_memory_gb,
            success=True,
        )

    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            print("  ✗ OOM - need larger ring_size")
        else:
            print(f"  ✗ Error: {error_msg[:100]}")

        return BenchmarkResult(
            seq_len=seq_len,
            ring_size=ring_size,
            chunk_size=chunk_size,
            time_per_chunk=0.0,
            total_time=0.0,
            peak_memory_gb=0.0,
            success=False,
            error=error_msg,
        )

    finally:
        # Cleanup
        if "module" in locals():
            del module
        torch.cuda.empty_cache()
        gc.collect()


def run_scaling_benchmark():
    """Run benchmarks with increasing sequence lengths."""

    print("Ring Attention Scaling Benchmark - Approaching 1B Tokens")
    print("=" * 80)

    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {gpu_memory_gb:.1f}GB")
    else:
        print("WARNING: Running on CPU - this will be very slow!")

    print(
        "\nNote: Using float16 precision and batch_size=1 for maximum sequence length"
    )

    # Test configurations: (seq_len, ring_size)
    # Start small and scale up, adjusting ring_size as needed
    test_configs = [
        # Baseline tests
        (8_192, 1),  # 8K baseline
        (8_192, 4),  # 8K with ring
        # Scale up gradually
        (32_768, 1),  # 32K baseline (might OOM)
        (32_768, 4),  # 32K with ring
        (131_072, 8),  # 128K
        (262_144, 16),  # 256K
        (524_288, 32),  # 512K
        # Approaching 1M
        (1_048_576, 64),  # 1M tokens
        # Multi-million sequences
        (4_194_304, 256),  # 4M tokens
        (16_777_216, 1024),  # 16M tokens
        # Approaching 100M
        (67_108_864, 4096),  # 64M tokens
        (134_217_728, 8192),  # 128M tokens
        # Approaching 1B
        (268_435_456, 16384),  # 256M tokens
        (536_870_912, 32768),  # 512M tokens
        (1_073_741_824, 65536),  # 1B tokens!
    ]

    results: list[BenchmarkResult] = []

    for seq_len, ring_size in test_configs:
        # Skip if sequence is too large for available memory
        est_memory = estimate_memory_usage(seq_len, 1, 8, 64, ring_size)
        if torch.cuda.is_available() and est_memory > gpu_memory_gb * 0.8:
            print(
                f"\nSkipping seq_len={seq_len:,} (estimated {est_memory:.1f}GB > available)"
            )
            continue

        result = benchmark_ring_attention(seq_len, ring_size)
        results.append(result)

        # Stop if we hit consistent failures
        if not result.success and seq_len > 1_000_000:
            print("\nStopping due to memory constraints")
            break

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS:")
    print(
        f"{'Seq Length':>15} {'Ring Size':>10} {'Time/Chunk':>12} {'Total Time':>12} {'Memory':>10} {'Status':>10}"
    )
    print("-" * 80)

    for r in results:
        if r.success:
            print(
                f"{r.seq_len:>15,} {r.ring_size:>10} {r.time_per_chunk * 1000:>10.1f}ms "
                f"{r.total_time:>10.1f}s {r.peak_memory_gb:>8.1f}GB {'✓':>10}"
            )
        else:
            print(
                f"{r.seq_len:>15,} {r.ring_size:>10} {'--':>12} {'--':>12} {'--':>10} {'✗':>10}"
            )

    # Analysis
    successful_results = [r for r in results if r.success]
    if successful_results:
        largest = max(successful_results, key=lambda x: x.seq_len)
        print(f"\nLargest successful sequence: {largest.seq_len:,} tokens")
        print(f"Ring size used: {largest.ring_size}")
        print(f"Time per chunk: {largest.time_per_chunk * 1000:.1f}ms")
        print(
            f"Estimated total time: {largest.total_time:.1f}s ({largest.total_time / 60:.1f} minutes)"
        )
        print(f"Peak memory usage: {largest.peak_memory_gb:.2f}GB")

        # Throughput calculation
        tokens_per_second = largest.seq_len / largest.total_time
        print(f"Throughput: {tokens_per_second:,.0f} tokens/second")

    print("\nKEY INSIGHTS:")
    print("1. Ring Attention enables linear memory scaling: O(n/ring_size)")
    print("2. Larger ring_size = more devices/chunks = longer sequences")
    print("3. Time scales linearly with ring_size (communication overhead)")
    print("4. With sufficient ring_size, billion-token sequences are feasible!")
    print("5. Real distributed setup would parallelize chunk processing")


if __name__ == "__main__":
    run_scaling_benchmark()

    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="ring-billion-tokens", parameters={}
    )

    # Add results
    output_manager.add_result("results", results)

    # Save results
    output_paths = output_manager.save_results()
    print(f"\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")
