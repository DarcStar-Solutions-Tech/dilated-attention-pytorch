#!/usr/bin/env python3
"""Quick test of dilation impact on Hilbert performance."""

import torch
import time
import numpy as np

# Import the model
from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
    RingDilatedAttentionHybridHilbert,
)


def test_configuration(seq_len, segments, dilations, batch_size=1, iterations=5):
    """Test a single configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model parameters
    num_heads = 12
    hidden_dim = 768
    head_dim = hidden_dim // num_heads

    print(f"\nTesting seq_len={seq_len:,}, segments={segments}, dilation={dilations}")

    try:
        # Create model with Hilbert
        model_hilbert = RingDilatedAttentionHybridHilbert(
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
            ring_size=1,
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=4096,
            enable_memory_pool=True,
            use_xformers=True,
        ).eval()

        # Create model without Hilbert
        model_no_hilbert = RingDilatedAttentionHybridHilbert(
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
            ring_size=1,
            device=device,
            dtype=dtype,
            use_hilbert=False,
            enable_memory_pool=True,
            use_xformers=True,
        ).eval()

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model_hilbert(q, k, v, is_causal=False)
                _ = model_no_hilbert(q, k, v, is_causal=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark with Hilbert
        hilbert_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_hilbert(q, k, v, is_causal=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            hilbert_times.append(time.perf_counter() - start)

        # Benchmark without Hilbert
        no_hilbert_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_no_hilbert(q, k, v, is_causal=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            no_hilbert_times.append(time.perf_counter() - start)

        # Calculate results
        hilbert_ms = np.mean(hilbert_times) * 1000
        no_hilbert_ms = np.mean(no_hilbert_times) * 1000
        speedup = no_hilbert_ms / hilbert_ms

        # Memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            memory_gb = 0

        print(
            f"  With Hilbert:    {hilbert_ms:6.2f} ms ({seq_len / np.mean(hilbert_times):,.0f} tokens/sec)"
        )
        print(
            f"  Without Hilbert: {no_hilbert_ms:6.2f} ms ({seq_len / np.mean(no_hilbert_times):,.0f} tokens/sec)"
        )
        print(f"  Hilbert speedup: {speedup:.2f}x")
        print(f"  Memory usage:    {memory_gb:.2f} GB")

        return {
            "success": True,
            "hilbert_ms": hilbert_ms,
            "no_hilbert_ms": no_hilbert_ms,
            "speedup": speedup,
            "memory_gb": memory_gb,
        }

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("QUICK DILATION IMPACT TEST")
    print("=" * 60)

    # Test configurations
    configs = [
        # (seq_len, segments, dilations, description)
        (16384, [4096], [1], "16K, no dilation"),
        (16384, [4096], [2], "16K, dilation=2"),
        (16384, [2048, 4096, 8192], [1, 2, 4], "16K, standard dilation"),
        (16384, [2048, 4096], [1, 4], "16K, high dilation"),
        (32768, [8192], [1], "32K, no dilation"),
        (32768, [8192], [2], "32K, dilation=2"),
        (32768, [4096, 8192, 16384], [1, 2, 4], "32K, standard dilation"),
        (32768, [4096, 8192], [1, 4], "32K, high dilation"),
        (65536, [16384], [1], "64K, no dilation"),
        (65536, [16384], [2], "64K, dilation=2"),
        (65536, [8192, 16384, 32768], [1, 2, 4], "64K, standard dilation"),
    ]

    results = []

    for seq_len, segments, dilations, desc in configs:
        # Check validity
        if seq_len % max(segments) != 0:
            print(f"\nSkipping {desc}: sequence not divisible by largest segment")
            continue

        print(f"\n{desc}:")
        result = test_configuration(seq_len, segments, dilations)
        result["description"] = desc
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["success"]]
    if successful:
        # Best speedup
        best_speedup = max(successful, key=lambda x: x["speedup"])
        print(f"\nBest Hilbert speedup: {best_speedup['speedup']:.2f}x")
        print(f"  Configuration: {best_speedup['description']}")

        # Group by sequence length
        for seq_len in [16384, 32768, 65536]:
            seq_results = [r for r in successful if seq_len in r["description"]]
            if seq_results:
                print(f"\n{seq_len // 1024}K sequence results:")
                for r in seq_results:
                    print(f"  {r['description']:<30} Speedup: {r['speedup']:.2f}x")


if __name__ == "__main__":
    main()
