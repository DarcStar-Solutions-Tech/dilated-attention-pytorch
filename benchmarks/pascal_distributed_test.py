#!/usr/bin/env python3
"""Distributed test optimized for Pascal GPUs (GTX 1080) with fp32."""

import os
import torch
import torch.distributed as dist
import time
from datetime import datetime
import gc


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


def test_configuration(seq_len, segments, dilations, batch_size=1):
    """Test a specific configuration."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    # Clear memory before test
    torch.cuda.empty_cache()
    gc.collect()

    # Import after device setup
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Reduced model parameters for Pascal GPUs
    num_heads = 4  # Reduced from 8
    head_dim = 32  # Reduced from 64
    _ = num_heads * head_dim
    dtype = torch.float32  # fp32 for Pascal

    try:
        # Create model with minimal settings
        model = RingDilatedAttentionHybridHilbert(
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=1024,  # Small chunk size
            enable_memory_pool=False,  # Disable memory pool
            use_xformers=False,  # Disable xformers
            enable_profiling=False,
        ).eval()

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Single forward pass for memory test
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start

        throughput = (batch_size * seq_len) / forward_time
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Clean up immediately
        del q, k, v, output, model
        torch.cuda.empty_cache()
        gc.collect()

        return True, throughput, memory_gb, forward_time * 1000

    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(f"    OOM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB used")
        torch.cuda.empty_cache()
        gc.collect()
        return False, None, None, None
    except Exception as e:
        if rank == 0:
            print(f"    Error: {str(e)[:50]}...")
        torch.cuda.empty_cache()
        gc.collect()
        return False, None, None, None


def main():
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("=" * 60)
        print("PASCAL GPU DISTRIBUTED TEST (fp32)")
        print("=" * 60)
        print(f"World size: {world_size} GPU(s)")
        print("Model: 4 heads, 32 dim/head (reduced for memory)")
        print()

    # Conservative test configurations for Pascal GPUs
    test_cases = [
        # Start very small to find working range
        (8192, [2048], [1], "8K, no dilation"),
        (8192, [2048], [2], "8K, dilation=2"),
        (16384, [4096], [1], "16K, no dilation"),
        (16384, [4096], [2], "16K, dilation=2"),
        (32768, [8192], [1], "32K, no dilation"),
        (32768, [8192], [2], "32K, dilation=2"),
        (65536, [16384], [1], "64K, no dilation"),
        (131072, [32768], [1], "128K, no dilation"),
        (262144, [65536], [1], "256K, no dilation"),
        (524288, [65536], [1], "512K, no dilation"),
    ]

    results = []
    max_working_seq = 0

    for seq_len, segments, dilations, desc in test_cases:
        # Check if valid for world size
        if seq_len % world_size != 0:
            continue

        if seq_len % max(segments) != 0:
            continue

        if rank == 0:
            print(f"\nTesting {desc}...")

        success, throughput, memory, time_ms = test_configuration(
            seq_len, segments, dilations
        )

        if rank == 0:
            if success:
                print("  ✓ Success!")
                print(f"    Time: {time_ms:.1f} ms")
                print(f"    Throughput: {throughput:,.0f} tokens/sec")
                print(f"    Memory per GPU: {memory:.2f} GB")

                max_working_seq = max(max_working_seq, seq_len)

                results.append(
                    {
                        "description": desc,
                        "seq_len": seq_len,
                        "segments": segments,
                        "dilations": dilations,
                        "throughput": throughput,
                        "memory_gb": memory,
                        "time_ms": time_ms,
                        "world_size": world_size,
                    }
                )
            else:
                print("  ✗ Failed")
                # Stop testing larger sizes after first failure
                if len(results) > 0:
                    break

    # Test with Hilbert disabled for comparison
    if rank == 0 and max_working_seq > 0:
        print("\n" + "-" * 60)
        print("TESTING WITHOUT HILBERT FOR COMPARISON")
        print("-" * 60)

    if max_working_seq > 0:
        # Test largest working sequence without Hilbert
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
            RingDilatedAttentionHybridHilbert,
        )

        torch.cuda.empty_cache()
        gc.collect()

        try:
            model_no_hilbert = RingDilatedAttentionHybridHilbert(
                segment_lengths=[max_working_seq // 4],
                dilation_rates=[1],
                dropout=0.0,
                ring_size=world_size,
                device=torch.device(f"cuda:{local_rank}"),
                dtype=torch.float32,
                use_hilbert=False,  # Disable Hilbert
                enable_memory_pool=False,
                use_xformers=False,
            ).eval()

            q = torch.randn(
                1,
                max_working_seq,
                4,
                32,
                device=torch.device(f"cuda:{local_rank}"),
                dtype=torch.float32,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Benchmark
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_no_hilbert(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            no_hilbert_time = time.perf_counter() - start

            if rank == 0:
                print(
                    f"\n{max_working_seq:,} tokens without Hilbert: {no_hilbert_time * 1000:.1f} ms"
                )

                # Find corresponding Hilbert result
                hilbert_result = next(
                    (r for r in results if r["seq_len"] == max_working_seq), None
                )
                if hilbert_result:
                    speedup = no_hilbert_time / (hilbert_result["time_ms"] / 1000)
                    print(f"Hilbert speedup: {speedup:.2f}x")

        except Exception as e:
            if rank == 0:
                print(f"No-Hilbert test failed: {str(e)[:50]}...")

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(
            f"\nMaximum sequence length on {world_size} GPUs: {max_working_seq:,} tokens"
        )
        print(f"Scaling from 1 GPU baseline (425K): {max_working_seq / 425984:.1f}x\n")

        print(
            f"{'Sequence':<15} | {'Throughput':<15} | {'Memory/GPU':<12} | {'Time':<10}"
        )
        print("-" * 60)

        for r in results[-3:]:  # Show last 3 results
            print(
                f"{r['description']:<15} | {r['throughput']:>14,.0f} | "
                f"{r['memory_gb']:>11.2f} | {r['time_ms']:>9.1f} ms"
            )

        # Dilation analysis
        dilation_results = [
            (r["seq_len"], r["dilations"][0], r["throughput"])
            for r in results
            if len(r["dilations"]) == 1
        ]

        if len(set(r[0] for r in dilation_results)) > 1:
            print("\nDilation Impact:")
            for seq_len in sorted(set(r[0] for r in dilation_results)):
                seq_results = [r for r in dilation_results if r[0] == seq_len]
                if len(seq_results) > 1:
                    no_dil = next((r[2] for r in seq_results if r[1] == 1), None)
                    with_dil = next((r[2] for r in seq_results if r[1] > 1), None)
                    if no_dil and with_dil:
                        print(
                            f"  {seq_len:,} tokens: {with_dil / no_dil:.2f}x speedup with dilation"
                        )

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"pascal_distributed_{world_size}gpu_{timestamp}.json"

        import json

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "gpu_type": "GTX 1080 (Pascal)",
                    "dtype": "float32",
                    "results": results,
                    "max_sequence_length": max_working_seq,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
