#!/usr/bin/env python3
"""Quick test of Hilbert V2 implementation."""

import torch
import time
import gc


def test_hilbert_v2():
    """Quick test of the Hilbert V2 improvements."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Test parameters
    seq_len = 16384
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print("=" * 60)
    print("HILBERT V2 QUICK TEST")
    print("=" * 60)
    print(f"Testing {seq_len:,} tokens with dilation=4")

    # Import implementations
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )
    from dilated_attention_pytorch.ring_dilated_attention_hilbert_v2 import (
        RingDilatedAttentionHilbertV2,
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test configurations
    configs = [
        ("Baseline", RingDilatedAttentionHybridHilbert, {"use_hilbert": False}),
        ("Original Hilbert", RingDilatedAttentionHybridHilbert, {"use_hilbert": True}),
        (
            "Hilbert V2 (dilated)",
            RingDilatedAttentionHilbertV2,
            {"use_hilbert": True, "hilbert_mode": "dilated"},
        ),
        (
            "Hilbert V2 (segment)",
            RingDilatedAttentionHilbertV2,
            {"use_hilbert": True, "hilbert_mode": "segment"},
        ),
    ]

    results = {}

    for name, model_class, kwargs in configs:
        print(f"\n{name}:")

        try:
            # Create model
            model = model_class(
                segment_lengths=[4096],
                dilation_rates=[4],
                dropout=0.0,
                ring_size=1,
                device=device,
                dtype=dtype,
                enable_memory_pool=False,
                use_xformers=False,
                **kwargs,
            ).eval()

            # Warmup
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)

            torch.cuda.synchronize()

            # Time
            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            throughput = seq_len / elapsed
            print(f"  Time: {elapsed * 1000:.2f} ms")
            print(f"  Throughput: {throughput:,.0f} tokens/sec")

            results[name] = {
                "time_ms": elapsed * 1000,
                "throughput": throughput,
            }

            # Cleanup
            del model, output
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results[name] = {"error": str(e)}

    # Analysis
    if "Baseline" in results and "throughput" in results["Baseline"]:
        baseline_throughput = results["Baseline"]["throughput"]

        print("\n" + "=" * 60)
        print("SPEEDUP ANALYSIS")
        print("=" * 60)

        for name, res in results.items():
            if "throughput" in res:
                speedup = res["throughput"] / baseline_throughput
                print(f"{name}: {speedup:.2f}x")


if __name__ == "__main__":
    test_hilbert_v2()
