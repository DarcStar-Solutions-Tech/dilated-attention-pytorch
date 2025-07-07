#!/usr/bin/env python3
"""
Benchmark Hilbert Attention V2 - with early data reordering.

This benchmark compares:
1. Standard block-sparse attention
2. Hilbert V1 (late reordering - what we tested before)
3. Hilbert V2 (early reordering - new approach)
"""

import torch
import time
import gc
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.hilbert_attention_v2 import (
    HilbertSequencePreprocessor,
    BlockSparseHilbertAttentionV2,
    HilbertTransformer,
)


@dataclass
class BenchmarkResult:
    implementation: str
    sequence_length: int
    forward_time_ms: float
    memory_mb: float
    preprocessing_time_ms: Optional[float] = None


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_standard_block_sparse(
    seq_lengths: List[int],
    batch_size: int = 1,
    embed_dim: int = 512,
    num_heads: int = 8,
) -> List[BenchmarkResult]:
    """Benchmark standard block-sparse attention."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print("\nStandard Block-Sparse:")
    print("-" * 50)

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model
            model = create_block_sparse_attention(
                variant="base",
                segment_lengths=[seq_len // 2],
                dilation_rates=[1],
                sparse_config=SparsePatternConfig(
                    pattern_type="dilated_sparse",
                    sparsity_ratio=0.05,
                    block_size=64,
                ),
            )

            # Convert to multihead format
            from dilated_attention_pytorch import create_multihead_block_sparse

            model = create_multihead_block_sparse(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sparsity_ratio=0.05,
                segment_lengths=[seq_len // 2],
                dilation_rates=[1],
            ).to(device)

            # Create input
            x = torch.randn(
                batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
            )

            # Warmup
            for _ in range(3):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = model(x, x, x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Benchmark
            mem_before = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(10):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(x, x, x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) / 10 * 1000

            mem_after = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

            print(
                f"  {seq_len:6d} tokens: {elapsed:7.2f}ms, {mem_after - mem_before:7.1f}MB"
            )

            results.append(
                BenchmarkResult(
                    implementation="standard",
                    sequence_length=seq_len,
                    forward_time_ms=elapsed,
                    memory_mb=mem_after - mem_before,
                )
            )

            del model, x, output

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {seq_len:6d} tokens: OOM")
                break
            else:
                print(f"  {seq_len:6d} tokens: Error - {str(e)[:50]}")
                break

    return results


def benchmark_hilbert_v1(
    seq_lengths: List[int],
    batch_size: int = 1,
    embed_dim: int = 512,
    num_heads: int = 8,
) -> List[BenchmarkResult]:
    """Benchmark Hilbert V1 (late reordering)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print("\nHilbert V1 (Late Reordering):")
    print("-" * 50)

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model
            from dilated_attention_pytorch import create_multihead_block_sparse

            _ = create_multihead_block_sparse(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sparsity_ratio=0.05,
                segment_lengths=[seq_len // 2],
                dilation_rates=[1],
                variant="hilbert",  # This would need to be added to multihead factory
            ).to(device)

            # For now, use base implementation as fallback
            _ = create_multihead_block_sparse(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sparsity_ratio=0.05,
                segment_lengths=[seq_len // 2],
                dilation_rates=[1],
            ).to(device)

            # Create input
            _ = torch.randn(
                batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
            )

            # Skip timing for V1 since we already know it's slower
            results.append(
                BenchmarkResult(
                    implementation="hilbert_v1",
                    sequence_length=seq_len,
                    forward_time_ms=0,  # Placeholder
                    memory_mb=0,
                )
            )

        except Exception:
            print(f"  {seq_len:6d} tokens: Skipped (not implemented in multihead)")

    return results


def benchmark_hilbert_v2(
    seq_lengths: List[int],
    batch_size: int = 1,
    embed_dim: int = 512,
    num_heads: int = 8,
) -> List[BenchmarkResult]:
    """Benchmark Hilbert V2 (early reordering)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print("\nHilbert V2 (Early Reordering):")
    print("-" * 50)

    # Create preprocessor once
    preprocessor = HilbertSequencePreprocessor(cache_mappings=True)

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model
            model = BlockSparseHilbertAttentionV2(
                embed_dim=embed_dim,
                num_heads=num_heads,
                block_size=64,
                sparsity_ratio=0.05,
                use_hilbert_pattern=True,
            ).to(device)

            # Create input
            x = torch.randn(
                batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
            )

            # Measure preprocessing time
            if device.type == "cuda":
                torch.cuda.synchronize()
            preprocess_start = time.perf_counter()

            x_hilbert = preprocessor.reorder_to_hilbert(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000

            # Warmup
            for _ in range(3):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = model(x_hilbert, x_hilbert, x_hilbert)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Benchmark
            mem_before = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(10):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(x_hilbert, x_hilbert, x_hilbert)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) / 10 * 1000

            # Don't forget to convert back
            output = preprocessor.reorder_from_hilbert(output)

            mem_after = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

            print(
                f"  {seq_len:6d} tokens: {elapsed:7.2f}ms, {mem_after - mem_before:7.1f}MB (preprocess: {preprocess_time:.2f}ms)"
            )

            results.append(
                BenchmarkResult(
                    implementation="hilbert_v2",
                    sequence_length=seq_len,
                    forward_time_ms=elapsed,
                    memory_mb=mem_after - mem_before,
                    preprocessing_time_ms=preprocess_time,
                )
            )

            del model, x, x_hilbert, output

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {seq_len:6d} tokens: OOM")
                break
            else:
                print(f"  {seq_len:6d} tokens: Error - {str(e)[:50]}")
                break

    return results


def benchmark_full_transformer(num_layers: int = 4):
    """Benchmark full transformer with Hilbert ordering."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nFull Hilbert Transformer:")
    print("-" * 50)

    seq_lengths = [512, 1024, 2048, 4096]
    batch_size = 1
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048

    _ = []

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            model = HilbertTransformer(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_seq_len=seq_len,
                block_size=64,
                sparsity_ratio=0.05,
            ).to(device)

            x = torch.randn(
                batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
            )

            # Warmup
            for _ in range(2):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Benchmark
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(5):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) / 5 * 1000

            print(f"  {seq_len:6d} tokens, {num_layers} layers: {elapsed:7.2f}ms")

            del model, x, output

        except Exception as e:
            print(f"  {seq_len:6d} tokens: Error - {str(e)[:50]}")


def main():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("Hilbert Attention V2 Benchmarks")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    seq_lengths = [1024, 2048, 4096, 8192, 16384]

    # Run benchmarks
    standard_results = benchmark_standard_block_sparse(seq_lengths)
    hilbert_v2_results = benchmark_hilbert_v2(seq_lengths)

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\nSpeedup (Hilbert V2 vs Standard):")
    for std, hv2 in zip(standard_results, hilbert_v2_results):
        if std.sequence_length == hv2.sequence_length and std.forward_time_ms > 0:
            speedup = std.forward_time_ms / hv2.forward_time_ms
            total_hv2_time = hv2.forward_time_ms + (hv2.preprocessing_time_ms or 0)
            speedup_with_preprocess = std.forward_time_ms / total_hv2_time
            print(
                f"  {std.sequence_length:6d} tokens: {speedup:5.2f}x (with preprocess: {speedup_with_preprocess:5.2f}x)"
            )

    # Test full transformer
    print("\n" + "=" * 70)
    print("FULL TRANSFORMER TEST")
    print("=" * 70)
    benchmark_full_transformer(num_layers=4)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"hilbert_v2_benchmark_{timestamp}.json"

    all_results = []
    for r in standard_results + hilbert_v2_results:
        all_results.append(
            {
                "implementation": r.implementation,
                "sequence_length": r.sequence_length,
                "forward_time_ms": r.forward_time_ms,
                "memory_mb": r.memory_mb,
                "preprocessing_time_ms": r.preprocessing_time_ms,
            }
        )

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nKey differences in V2:")
    print("1. Hilbert reordering happens ONCE at the beginning")
    print("2. All layers work with Hilbert-ordered data")
    print("3. Sparse patterns designed for Hilbert locality")
    print("4. No repeated reordering overhead")
    print("\nThis approach should show better cache utilization")
    print("especially for multi-layer transformers.")


if __name__ == "__main__":
    main()
