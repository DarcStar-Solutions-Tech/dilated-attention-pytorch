#!/usr/bin/env python3
"""Simple test of dilation impact on performance."""

import torch
import time


def main():
    # Check if CUDA available
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    device = torch.device("cuda")
    dtype = torch.float32  # Use float32 for GTX 1080

    # Test parameters
    batch_size = 1
    num_heads = 8
    head_dim = 64
    _ = num_heads * head_dim

    print("DILATION IMPACT ON HILBERT PERFORMANCE")
    print("=" * 50)

    # Test configurations
    test_cases = [
        (16384, [4096], [1]),  # 16K, no dilation
        (16384, [4096], [2]),  # 16K, dilation=2
        (32768, [8192], [1]),  # 32K, no dilation
        (32768, [8192], [2]),  # 32K, dilation=2
    ]

    for seq_len, segments, dilations in test_cases:
        print(f"\nSeq: {seq_len}, Segments: {segments}, Dilation: {dilations}")

        try:
            # Create models
            model_hilbert = RingDilatedAttentionHybridHilbert(
                segment_lengths=segments,
                dilation_rates=dilations,
                dropout=0.0,
                ring_size=1,
                device=device,
                dtype=dtype,
                use_hilbert=True,
                enable_memory_pool=False,  # Disable for simple test
                use_xformers=False,  # Disable for simple test
            ).eval()

            model_no_hilbert = RingDilatedAttentionHybridHilbert(
                segment_lengths=segments,
                dilation_rates=dilations,
                dropout=0.0,
                ring_size=1,
                device=device,
                dtype=dtype,
                use_hilbert=False,
                enable_memory_pool=False,
                use_xformers=False,
            ).eval()

            # Create inputs
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup
            with torch.no_grad():
                _ = model_hilbert(q, k, v, is_causal=False)
                _ = model_no_hilbert(q, k, v, is_causal=False)

            torch.cuda.synchronize()

            # Time Hilbert
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_hilbert(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            hilbert_time = time.perf_counter() - start

            # Time no Hilbert
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_no_hilbert(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            no_hilbert_time = time.perf_counter() - start

            # Results
            speedup = no_hilbert_time / hilbert_time
            print(f"  Hilbert: {hilbert_time * 1000:.1f}ms")
            print(f"  No Hilbert: {no_hilbert_time * 1000:.1f}ms")
            print(f"  Speedup: {speedup:.2f}x")

            # Memory
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Memory: {mem_gb:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            print(f"  ERROR: {str(e)}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
