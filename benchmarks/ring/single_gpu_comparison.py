#!/usr/bin/env python3
"""
Single GPU comparison of ring attention implementations.

Quick benchmark to compare all 4 standardized implementations on a single GPU.
"""

import torch
import time
import gc
import pandas as pd

from dilated_attention_pytorch import (
    StandardRingAttention,
    DistributedRingAttention,
    HilbertRingAttention,
    RingBlockSparseAttention,
    RingAttentionConfig,
)
from dilated_attention_pytorch.utils import get_optimal_dtype


def benchmark_single_gpu():
    """Run single GPU benchmark of all implementations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_optimal_dtype(device)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Test parameters
    batch_size = 2
    num_heads = 8
    head_dim = 64
    seq_lengths = [1024, 2048, 4096, 8192]

    # Common config
    segment_lengths = [512, 1024, 2048]
    dilation_rates = [1, 2, 4]

    results = []

    implementations = {
        "Standard": StandardRingAttention,
        "Distributed": DistributedRingAttention,
        "Hilbert": HilbertRingAttention,
        "BlockSparse": RingBlockSparseAttention,
    }

    for seq_len in seq_lengths:
        print(f"\n{'=' * 60}")
        print(f"Testing sequence length: {seq_len}")
        print(f"{'=' * 60}")

        # Adjust segment lengths for sequence
        adjusted_segments = [min(s, seq_len) for s in segment_lengths]

        for name, cls in implementations.items():
            print(f"\n{name} Ring Attention:")

            try:
                # Create config
                config = RingAttentionConfig(
                    segment_lengths=adjusted_segments,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                )

                # Create model
                if name == "BlockSparse":
                    model = cls(
                        config=config,
                        sparsity_ratio=0.1,
                        pattern_type="dilated_sparse",
                        device=device,
                        dtype=dtype,
                    )
                else:
                    model = cls(config=config, device=device, dtype=dtype)

                model.eval()

                # Create inputs
                q = torch.randn(
                    batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(q, k, v)

                # Clear cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Measure memory
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated()

                # Time measurement
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    torch.cuda.synchronize()
                    start_event.record()
                else:
                    start_time = time.perf_counter()

                # Run forward pass
                num_iters = 10
                with torch.no_grad():
                    for _ in range(num_iters):
                        output = model(q, k, v)

                # Record time
                if device.type == "cuda":
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
                    peak_mem = (torch.cuda.max_memory_allocated() - start_mem) / (
                        1024**2
                    )
                else:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000 / num_iters
                    peak_mem = 0

                # Calculate metrics
                total_tokens = batch_size * seq_len
                throughput = (total_tokens / elapsed_ms) * 1000  # tokens/sec

                print(f"  ✓ Time: {elapsed_ms:.2f} ms")
                print(f"  ✓ Throughput: {throughput:,.0f} tokens/sec")
                print(f"  ✓ Peak Memory: {peak_mem:.2f} MB")
                print(f"  ✓ Output shape: {output.shape}")

                results.append(
                    {
                        "implementation": name,
                        "seq_length": seq_len,
                        "time_ms": elapsed_ms,
                        "throughput": throughput,
                        "memory_mb": peak_mem,
                    }
                )

                # Cleanup
                del model, output
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results.append(
                    {
                        "implementation": name,
                        "seq_length": seq_len,
                        "time_ms": None,
                        "throughput": None,
                        "memory_mb": None,
                    }
                )

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    df = pd.DataFrame(results)

    # Pivot table for better readability
    for metric in ["time_ms", "throughput", "memory_mb"]:
        print(f"\n{metric.upper()}:")
        pivot = df.pivot(index="seq_length", columns="implementation", values=metric)
        print(pivot.round(2))

    # Find best implementation for each sequence length
    print("\n\nBEST IMPLEMENTATION BY SEQUENCE LENGTH:")
    print("(Based on throughput)")
    for seq_len in seq_lengths:
        seq_df = df[df["seq_length"] == seq_len].dropna()
        if len(seq_df) > 0:
            best = seq_df.loc[seq_df["throughput"].idxmax()]
            print(
                f"  {seq_len}: {best['implementation']} ({best['throughput']:,.0f} tokens/sec)"
            )


if __name__ == "__main__":
    benchmark_single_gpu()
