#!/usr/bin/env python3
"""Benchmark using the actual Triton Hilbert kernel."""

import torch
import torch.distributed as dist
import time
import os
import gc
from datetime import datetime


def init_distributed():
    """Initialize distributed if not already done."""
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank
    return 0, 0


def benchmark_triton_hilbert():
    """Benchmark the Triton Hilbert implementation."""

    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32  # fp32 for Pascal

    print(f"{'=' * 60}")
    print("TRITON HILBERT KERNEL BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"World size: {world_size}")
    print("Using fp32 for Pascal architecture")

    # Import implementations
    try:
        from dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton import (
            HilbertDilatedAttentionTriton,
        )

        print("✓ Triton Hilbert kernel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Triton kernel: {e}")
        return

    # Also import baseline for comparison
    from dilated_attention_pytorch.dilated_attention import DilatedAttention
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Test configurations
    test_cases = [
        (8192, [2048], [1], 8, 64, "8K, no dilation"),
        (8192, [2048], [4], 8, 64, "8K, dilation=4"),
        (16384, [4096], [1], 8, 64, "16K, no dilation"),
        (16384, [4096], [4], 8, 64, "16K, dilation=4"),
        (32768, [8192], [1], 8, 64, "32K, no dilation"),
        (32768, [8192], [4], 8, 64, "32K, dilation=4"),
    ]

    results = []

    for seq_len, segments, dilations, num_heads, head_dim, desc in test_cases:
        if seq_len % world_size != 0:
            continue

        print(f"\n{desc}:")
        print(f"  Segments: {segments}, Dilation: {dilations}")

        # Create inputs
        batch_size = 1
        hidden_dim = num_heads * head_dim

        # For Triton kernel - expects (batch, heads, seq, dim)
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # For other implementations - expects (batch, seq, heads, dim)
        q_alt = q.transpose(1, 2).contiguous()
        k_alt = k.transpose(1, 2).contiguous()
        v_alt = v.transpose(1, 2).contiguous()

        result = {"description": desc, "seq_len": seq_len}

        # Test 1: Baseline DilatedAttention
        try:
            model_baseline = DilatedAttention(
                segment_lengths=segments,
                dilation_rates=dilations,
                attention_dropout=0.0,
            )

            # Warmup
            with torch.no_grad():
                _ = model_baseline(q_alt, k_alt, v_alt, is_causal=False)

            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_baseline(q_alt, k_alt, v_alt, is_causal=False)
            torch.cuda.synchronize()
            baseline_time = time.perf_counter() - start

            print(f"  Baseline DilatedAttention: {baseline_time * 1000:.2f} ms")
            result["baseline_ms"] = baseline_time * 1000

        except Exception as e:
            print(f"  Baseline failed: {str(e)[:50]}...")
            result["baseline_error"] = str(e)

        # Test 2: Python Hilbert (current implementation)
        try:
            model_python = RingDilatedAttentionHybridHilbert(
                segment_lengths=segments,
                dilation_rates=dilations,
                dropout=0.0,
                ring_size=1,
                device=device,
                dtype=dtype,
                use_hilbert=True,
                enable_memory_pool=False,
                use_xformers=False,
            )

            with torch.no_grad():
                _ = model_python(q_alt, k_alt, v_alt, is_causal=False)

            torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = model_python(q_alt, k_alt, v_alt, is_causal=False)
            torch.cuda.synchronize()
            python_time = time.perf_counter() - start

            print(f"  Python Hilbert: {python_time * 1000:.2f} ms")
            result["python_hilbert_ms"] = python_time * 1000

        except Exception as e:
            print(f"  Python Hilbert failed: {str(e)[:50]}...")
            result["python_hilbert_error"] = str(e)

        # Test 3: Triton Hilbert kernel
        try:
            model_triton = HilbertDilatedAttentionTriton(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segments[0],
                dilation_rate=dilations[0],
                dropout=0.0,
            ).to(device)

            # Warmup
            with torch.no_grad():
                _ = model_triton(q, k, v)

            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_triton(q, k, v)
            torch.cuda.synchronize()
            triton_time = time.perf_counter() - start

            print(f"  Triton Hilbert: {triton_time * 1000:.2f} ms")
            result["triton_hilbert_ms"] = triton_time * 1000

            # Calculate speedups
            if "baseline_ms" in result:
                speedup = result["baseline_ms"] / (triton_time * 1000)
                print(f"  Triton speedup vs baseline: {speedup:.2f}x")
                result["triton_speedup"] = speedup

        except Exception as e:
            print(f"  Triton Hilbert failed: {str(e)[:50]}...")
            result["triton_hilbert_error"] = str(e)

        results.append(result)

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    print(
        f"\n{'Config':<20} | {'Baseline':<10} | {'Python H':<10} | {'Triton H':<10} | {'Speedup':<8}"
    )
    print("-" * 70)

    for r in results:
        config = f"{r['seq_len'] // 1024}K, d={r['description'].split('=')[-1] if '=' in r['description'] else '1'}"
        baseline = f"{r.get('baseline_ms', 0):.1f}" if "baseline_ms" in r else "error"
        python = (
            f"{r.get('python_hilbert_ms', 0):.1f}"
            if "python_hilbert_ms" in r
            else "error"
        )
        triton = (
            f"{r.get('triton_hilbert_ms', 0):.1f}"
            if "triton_hilbert_ms" in r
            else "error"
        )
        speedup = (
            f"{r.get('triton_speedup', 0):.2f}x" if "triton_speedup" in r else "n/a"
        )

        print(
            f"{config:<20} | {baseline:<10} | {python:<10} | {triton:<10} | {speedup:<8}"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"triton_hilbert_benchmark_{world_size}gpu_{timestamp}.json"

    import json

    with open(filename, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "world_size": world_size,
                "dtype": "float32",
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")

    # Multi-GPU test if available
    if world_size > 1:
        print(f"\n{'=' * 60}")
        print("MULTI-GPU RING ATTENTION TEST")
        print(f"{'=' * 60}")

        # Test ring attention across GPUs
        # ... (add multi-GPU specific tests here)


if __name__ == "__main__":
    benchmark_triton_hilbert()
