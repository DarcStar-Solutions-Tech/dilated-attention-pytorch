#!/usr/bin/env python3
"""Single GPU scaling test with fp32 for Pascal architecture."""

import torch
import time
import gc
from datetime import datetime


def test_sequence_length(seq_len, segments, dilations, batch_size=1):
    """Test if a sequence length works on single GPU."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Model parameters
    num_heads = 8
    head_dim = 64
    _ = num_heads * head_dim
    dtype = torch.float32  # fp32 for Pascal

    try:
        # Create model
        model = RingDilatedAttentionHybridHilbert(
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
            ring_size=1,  # Single GPU
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=4096,
            enable_memory_pool=False,  # Disable for memory efficiency
            use_xformers=False,
            enable_profiling=False,
        ).eval()

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start

        # Get memory stats
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        throughput = (batch_size * seq_len) / forward_time

        # Cleanup
        del q, k, v, output, model
        torch.cuda.empty_cache()
        gc.collect()

        return True, throughput, memory_gb, forward_time * 1000

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False, None, None, None
    except Exception as e:
        print(f"Error: {str(e)[:50]}...")
        torch.cuda.empty_cache()
        gc.collect()
        return False, None, None, None


def find_max_sequence(min_seq, max_seq, segments, dilations):
    """Binary search for maximum sequence length."""
    print(f"\nFinding max sequence for segments={segments}, dilation={dilations}")

    max_segment = max(segments)
    best_seq = 0
    best_stats = None

    while min_seq <= max_seq:
        # Find midpoint that's divisible by max_segment
        mid_seq = (min_seq + max_seq) // 2
        mid_seq = ((mid_seq + max_segment - 1) // max_segment) * max_segment

        print(f"  Testing {mid_seq:,} tokens... ", end="", flush=True)

        success, throughput, memory, time_ms = test_sequence_length(
            mid_seq, segments, dilations
        )

        if success:
            print(f"✓ ({throughput:,.0f} tok/s, {memory:.1f}GB)")
            best_seq = mid_seq
            best_stats = (throughput, memory, time_ms)
            min_seq = mid_seq + max_segment
        else:
            print("✗ (OOM)")
            max_seq = mid_seq - max_segment

    return best_seq, best_stats


def main():
    print("=" * 60)
    print("SINGLE GPU FP32 SCALING TEST (Pascal Architecture)")
    print("=" * 60)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    print()

    # Test configurations
    configs = [
        # (segments, dilations, description)
        ([8192], [1], "Single segment, no dilation"),
        ([8192], [2], "Single segment, dilation=2"),
        ([4096, 8192], [1, 2], "Multi-segment, standard dilation"),
        ([8192, 16384], [1, 2], "Large segments"),
        ([4096, 8192, 16384], [1, 2, 4], "Three segments"),
    ]

    results = []

    for segments, dilations, desc in configs:
        print(f"\n{desc}:")

        # Start from 16K, try up to 1M
        max_seq, stats = find_max_sequence(16384, 1048576, segments, dilations)

        if max_seq > 0 and stats:
            throughput, memory, time_ms = stats
            print(f"  Maximum: {max_seq:,} tokens")
            print(f"  Performance: {throughput:,.0f} tokens/sec")
            print(f"  Memory: {memory:.2f} GB")

            results.append(
                {
                    "segments": segments,
                    "dilations": dilations,
                    "description": desc,
                    "max_seq_len": max_seq,
                    "throughput": throughput,
                    "memory_gb": memory,
                    "time_ms": time_ms,
                }
            )

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Sort by max sequence length
        results.sort(key=lambda x: x["max_seq_len"], reverse=True)

        print(f"\n{'Configuration':<30} | {'Max Sequence':<12} | {'Memory':<8}")
        print("-" * 55)

        for r in results:
            print(
                f"{r['description']:<30} | {r['max_seq_len']:>11,} | {r['memory_gb']:>7.2f} GB"
            )

        # Best configuration
        best = results[0]
        print(f"\nBest configuration: {best['max_seq_len']:,} tokens")
        print(f"  Segments: {best['segments']}")
        print(f"  Dilations: {best['dilations']}")

        # Compare with baseline
        baseline = 425984  # From previous test
        improvement = best["max_seq_len"] / baseline
        print(f"\nImprovement over baseline: {improvement:.2f}x")

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"single_gpu_fp32_scaling_{timestamp}.json"

        import json

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "gpu": gpu_name,
                    "gpu_memory_gb": gpu_memory,
                    "dtype": "float32",
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
