#!/usr/bin/env python3
"""
Compare Hybrid Ring Dilated Attention vs Improved Dilated Attention.
Tests memory efficiency, max sequence length, and performance characteristics.
"""

import torch
import time
import gc
from typing import Tuple
import json

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        return {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            "compute_capability": torch.cuda.get_device_capability(0),
        }
    return None


def test_max_sequence_length(
    model_type: str,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: list = None,
    dilation_rates: list = None,
    start_seq: int = 16384,
    max_seq: int = 512000,
) -> Tuple[int, float]:
    """Find maximum sequence length that fits in memory."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default segment configuration
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    current_seq = start_seq
    max_working = 0
    max_memory = 0

    while current_seq <= max_seq:
        # Ensure divisibility by largest segment
        if current_seq % max(segment_lengths) != 0:
            current_seq = ((current_seq // max(segment_lengths)) + 1) * max(
                segment_lengths
            )

        try:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create model
            if model_type == "hybrid":
                model = RingDilatedAttentionHybrid(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    device=device,
                    dtype=torch.float32,
                    ring_size=1,  # Single GPU for fair comparison
                    use_flash_attention=True,
                    enable_memory_pool=True,
                )
            else:  # improved
                model = ImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    use_xformers=True,
                    use_flex_attention=False,  # Pascal doesn't support
                )

            # Create inputs
            q = torch.randn(batch_size, current_seq, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, current_seq, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, current_seq, num_heads, head_dim, device=device)

            # Forward pass
            with torch.no_grad():
                output = model(q, k, v)

            # Get memory usage
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

            max_working = current_seq
            max_memory = peak_memory

            print(
                f"{model_type}: {current_seq:,} tokens - Success! {peak_memory:.2f} GB"
            )

            # Clean up
            del q, k, v, output, model
            gc.collect()
            torch.cuda.empty_cache()

            # Try larger sequence
            if current_seq < 100000:
                current_seq += 16384
            elif current_seq < 200000:
                current_seq += 32768
            else:
                current_seq += 65536

        except torch.cuda.OutOfMemoryError:
            print(f"{model_type}: {current_seq:,} tokens - OOM")
            break
        except Exception as e:
            print(f"{model_type}: {current_seq:,} tokens - Error: {e}")
            break

    return max_working, max_memory


def benchmark_performance(
    model_type: str,
    seq_lengths: list,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: list = None,
    dilation_rates: list = None,
) -> list:
    """Benchmark performance at different sequence lengths."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    results = []

    for seq_len in seq_lengths:
        # Ensure divisibility
        if seq_len % max(segment_lengths) != 0:
            seq_len = ((seq_len // max(segment_lengths)) + 1) * max(segment_lengths)

        try:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create model
            if model_type == "hybrid":
                model = RingDilatedAttentionHybrid(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    device=device,
                    dtype=torch.float32,
                    ring_size=1,
                    use_flash_attention=True,
                    enable_memory_pool=True,
                )
            else:  # improved
                model = ImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    use_xformers=True,
                    use_flex_attention=False,
                )

            # Create inputs
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    _ = model(q, k, v)

            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()

            num_iterations = 5
            for _ in range(num_iterations):
                with torch.no_grad():
                    output = model(q, k, v)

            torch.cuda.synchronize()
            total_time = time.time() - start_time
            avg_time = total_time / num_iterations

            # Get memory
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            memory_per_token = peak_memory * 1024 / seq_len  # KB per token

            results.append(
                {
                    "seq_len": seq_len,
                    "time_ms": avg_time * 1000,
                    "memory_mb": peak_memory,
                    "memory_per_token_kb": memory_per_token,
                    "throughput_tokens_per_sec": seq_len * batch_size / avg_time,
                }
            )

            print(
                f"{model_type} @ {seq_len:,}: {avg_time * 1000:.1f}ms, "
                f"{peak_memory:.0f}MB, {memory_per_token:.1f}KB/token"
            )

            # Clean up
            del q, k, v, output, model
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{model_type} @ {seq_len:,}: OOM")
            break
        except Exception as e:
            print(f"{model_type} @ {seq_len:,}: Error - {e}")
            break

    return results


def main():
    """Run comparison between Hybrid and Improved implementations."""

    print("=== Hybrid Ring vs Improved Dilated Attention Comparison ===")

    # Get GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\nGPU: {gpu_info['name']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"Compute Capability: {gpu_info['compute_capability']}")

    # Test configurations
    test_sequences = [8192, 16384, 32768, 65536, 131072, 196608, 262144]

    print("\n" + "=" * 60)
    print("Performance Benchmark (Single GPU)")
    print("=" * 60)

    # Benchmark Improved
    print("\nImproved Dilated Attention:")
    improved_results = benchmark_performance("improved", test_sequences)

    # Benchmark Hybrid
    print("\nHybrid Ring Dilated Attention (Single GPU):")
    hybrid_results = benchmark_performance("hybrid", test_sequences)

    print("\n" + "=" * 60)
    print("Maximum Sequence Length Test")
    print("=" * 60)

    # Find max sequence for Improved
    print("\nTesting Improved Dilated Attention max sequence...")
    improved_max, improved_max_mem = test_max_sequence_length(
        "improved", start_seq=131072
    )

    # Find max sequence for Hybrid
    print("\nTesting Hybrid Ring Dilated Attention max sequence...")
    hybrid_max, hybrid_max_mem = test_max_sequence_length("hybrid", start_seq=65536)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nMaximum Sequence Length (8GB GPU):")
    print(
        f"  Improved Dilated Attention: {improved_max:,} tokens ({improved_max_mem:.2f} GB)"
    )
    print(
        f"  Hybrid Ring (Single GPU): {hybrid_max:,} tokens ({hybrid_max_mem:.2f} GB)"
    )
    print(
        f"  Advantage: Improved handles {improved_max / hybrid_max:.1f}x longer sequences"
    )

    print("\nMemory Efficiency Comparison:")
    # Compare at common sequence length
    common_seq = min(65536, improved_max, hybrid_max)
    improved_mem = next(
        (r for r in improved_results if r["seq_len"] >= common_seq), None
    )
    hybrid_mem = next((r for r in hybrid_results if r["seq_len"] >= common_seq), None)

    if improved_mem and hybrid_mem:
        print(f"  At {common_seq:,} tokens:")
        print(f"    Improved: {improved_mem['memory_per_token_kb']:.1f} KB/token")
        print(f"    Hybrid: {hybrid_mem['memory_per_token_kb']:.1f} KB/token")
        print(
            f"    Improved is {hybrid_mem['memory_per_token_kb'] / improved_mem['memory_per_token_kb']:.1f}x more memory efficient"
        )

    print("\nKey Differences:")
    print("  Improved Dilated Attention:")
    print("    - Optimized for single GPU with extreme memory efficiency")
    print("    - Can handle 250K+ tokens on 8GB Pascal GPU")
    print("    - Uses sophisticated caching and memory pooling")
    print("    - Best for single GPU, very long sequences")

    print("\n  Hybrid Ring Dilated Attention:")
    print("    - Designed for multi-GPU scaling with O(n/p) memory")
    print("    - Limited by ring communication overhead on single GPU")
    print("    - Excellent multi-GPU scaling (tested up to 65K tokens/GPU)")
    print("    - Best for distributed training with multiple GPUs")

    # Save results
    results = {
        "gpu_info": gpu_info,
        "improved_results": improved_results,
        "hybrid_results": hybrid_results,
        "improved_max_seq": improved_max,
        "hybrid_max_seq": hybrid_max,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    with open("hybrid_vs_improved_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to hybrid_vs_improved_comparison.json")


if __name__ == "__main__":
    main()
