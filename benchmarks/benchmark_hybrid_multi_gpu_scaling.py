#!/usr/bin/env python3
"""
Benchmark Hybrid Ring Attention on multiple GPUs focusing on:
- Maximum sequence length
- Memory per token
- Performance scaling

Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_multi_gpu_scaling.py
"""

import os
import gc
import time
import json
import torch
import torch.distributed as dist
from datetime import datetime
from typing import Dict, List, Tuple

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def measure_sequence_performance(
    seq_len: int,
    world_size: int,
    device: torch.device,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    num_iterations: int = 3,
) -> Tuple[bool, Dict]:
    """Measure performance for a specific sequence length."""

    # Clear memory before test
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Create model - let it auto-select dtype
        model = RingDilatedAttentionHybrid(
            segment_lengths=[seq_len // 2],  # Single segment for simplicity
            dilation_rates=[1],
            dropout=0.0,
            ring_size=world_size,
            device=device,
            # Auto dtype selection based on GPU
            enable_memory_pool=True,
            use_pattern_cache=True,
            use_flash_attention=False,  # Disable for consistent measurement
        )

        # Get model dtype
        dtype = model.dtype
        bytes_per_element = 4 if dtype == torch.float32 else 2

        # Memory after model creation
        mem_model = torch.cuda.memory_allocated(device) / (1024**3)  # GB

        # Create inputs
        torch.manual_seed(42)
        scale = 0.1 / (seq_len**0.25)

        q = (
            torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            * scale
        )
        k = (
            torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            * scale
        )
        v = (
            torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            * scale
        )

        # Memory after inputs
        mem_with_inputs = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        input_memory = mem_with_inputs - mem_model

        # Theoretical memory for inputs
        total_tokens = batch_size * seq_len
        elements_per_token = num_heads * head_dim * 3  # Q, K, V
        theoretical_input_gb = (
            total_tokens * elements_per_token * bytes_per_element
        ) / (1024**3)

        # Warmup
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        # Timed iterations
        torch.cuda.reset_peak_memory_stats()
        mem_before_forward = torch.cuda.memory_allocated(device) / (1024**3)

        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                output = model(q, k, v, is_causal=False)

            torch.cuda.synchronize()
            times.append(time.time() - start)

        # Memory stats
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        forward_memory = peak_memory_gb - mem_before_forward

        # Calculate memory per token
        memory_per_token_bytes = (peak_memory_gb * 1024**3) / total_tokens

        # Validate output
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        # Clean up
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        # Success - return metrics
        avg_time = sum(times) / len(times)

        return True, {
            "seq_len": seq_len,
            "total_tokens": total_tokens,
            "avg_time_ms": avg_time * 1000,
            "throughput_tokens_per_sec": total_tokens / avg_time,
            "model_memory_gb": mem_model,
            "input_memory_gb": input_memory,
            "theoretical_input_gb": theoretical_input_gb,
            "peak_memory_gb": peak_memory_gb,
            "forward_memory_gb": forward_memory,
            "memory_per_token_bytes": memory_per_token_bytes,
            "memory_per_token_mb": memory_per_token_bytes / (1024**2),
            "dtype": str(dtype),
            "bytes_per_element": bytes_per_element,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

    except torch.cuda.OutOfMemoryError as e:
        return False, {"seq_len": seq_len, "error": "OOM", "error_detail": str(e)}
    except Exception as e:
        return False, {
            "seq_len": seq_len,
            "error": str(type(e).__name__),
            "error_detail": str(e),
        }


def find_max_sequence_length(
    world_size: int, device: torch.device
) -> Tuple[int, List[Dict]]:
    """Binary search to find maximum sequence length."""

    # Start with reasonable bounds
    min_seq = 1024
    max_seq = 128 * 1024  # 128K tokens

    # First, find upper bound that fails
    test_seq = min_seq
    while test_seq <= max_seq:
        success, result = measure_sequence_performance(test_seq, world_size, device)
        if not success:
            max_seq = test_seq
            break
        test_seq *= 2

    # Binary search for exact maximum
    results = []
    while min_seq < max_seq - 256:  # 256 token precision
        mid_seq = (min_seq + max_seq) // 2
        mid_seq = (mid_seq // 256) * 256  # Round to nearest 256

        success, result = measure_sequence_performance(mid_seq, world_size, device)
        results.append(result)

        if success:
            min_seq = mid_seq
        else:
            max_seq = mid_seq

    return min_seq, results


def benchmark_scaling():
    """Main benchmark function."""

    # Initialize distributed
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_multi_gpu_scaling.py"
        )
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Hybrid Ring Attention - Multi-GPU Scaling Benchmark")
        print("=" * 70)
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"World size: {world_size} GPUs")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
        print()

    # Test sequence lengths - exponential growth
    test_sequences = []
    seq = 1024
    while seq <= 64 * 1024:  # Up to 64K
        test_sequences.append(seq)
        seq = int(seq * 1.5)  # 50% growth

    # Add some key points
    test_sequences.extend([2048, 4096, 8192, 16384, 32768])
    test_sequences = sorted(list(set(test_sequences)))

    all_results = []
    max_successful_seq = 0

    for seq_len in test_sequences:
        if rank == 0:
            print(
                f"\nTesting sequence length: {seq_len:,} tokens...", end="", flush=True
            )

        # Synchronize all ranks
        dist.barrier()

        # Test this sequence length
        success, result = measure_sequence_performance(seq_len, world_size, device)

        # Gather results from all ranks
        all_rank_results = [None] * world_size
        dist.all_gather_object(all_rank_results, (success, result))

        # Check if all ranks succeeded
        all_success = all(r[0] for r in all_rank_results)

        if all_success:
            max_successful_seq = seq_len

            if rank == 0:
                # Aggregate metrics
                results = [r[1] for r in all_rank_results]

                avg_time = sum(r["avg_time_ms"] for r in results) / len(results)
                max_peak = max(r["peak_memory_gb"] for r in results)
                avg_peak = sum(r["peak_memory_gb"] for r in results) / len(results)
                avg_mem_per_token = sum(
                    r["memory_per_token_mb"] for r in results
                ) / len(results)

                aggregated = {
                    "seq_len": seq_len,
                    "success": True,
                    "avg_time_ms": avg_time,
                    "throughput_tokens_per_sec": seq_len / (avg_time / 1000),
                    "avg_peak_memory_gb": avg_peak,
                    "max_peak_memory_gb": max_peak,
                    "avg_memory_per_token_mb": avg_mem_per_token,
                    "all_results": results,
                }
                all_results.append(aggregated)

                print(" ✓ Success!")
                print(
                    f"  Time: {avg_time:.1f}ms, Memory: {avg_peak:.2f}GB, Mem/token: {avg_mem_per_token:.2f}MB"
                )
        else:
            if rank == 0:
                print(" ✗ Failed")
                failed_results = [r[1] for r in all_rank_results if not r[0]]
                error_types = set(r.get("error", "Unknown") for r in failed_results)
                print(f"  Errors: {', '.join(error_types)}")

                all_results.append(
                    {
                        "seq_len": seq_len,
                        "success": False,
                        "errors": list(error_types),
                    }
                )
            break  # Stop testing larger sequences

    # Find true maximum through binary search
    if rank == 0 and max_successful_seq > 0:
        print(
            f"\nRefining maximum sequence length (last success: {max_successful_seq:,})..."
        )

    dist.barrier()

    # Save results
    if rank == 0:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"benchmarks/hybrid_scaling_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "gpu_name": torch.cuda.get_device_name(device),
                    "compute_capability": str(torch.cuda.get_device_capability(device)),
                    "max_successful_seq": max_successful_seq,
                    "results": all_results,
                },
                f,
                indent=2,
            )

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"\nMaximum successful sequence length: {max_successful_seq:,} tokens")

        # Performance scaling analysis
        print("\nPerformance Scaling:")
        print(
            f"{'Seq Length':<12} {'Time (ms)':<12} {'Throughput':<15} {'Memory (GB)':<12} {'MB/token':<10}"
        )
        print("-" * 70)

        for r in all_results:
            if r["success"]:
                print(
                    f"{r['seq_len']:<12,} {r['avg_time_ms']:<12.1f} {r['throughput_tokens_per_sec']:<15.0f} "
                    f"{r['avg_peak_memory_gb']:<12.2f} {r['avg_memory_per_token_mb']:<10.2f}"
                )

        # Memory efficiency analysis
        print(f"\nMemory Efficiency (World size = {world_size}):")
        successful = [r for r in all_results if r["success"]]
        if successful:
            # Memory per token should remain relatively constant
            mem_per_token_values = [r["avg_memory_per_token_mb"] for r in successful]
            avg_mem_per_token = sum(mem_per_token_values) / len(mem_per_token_values)

            print(f"Average memory per token: {avg_mem_per_token:.2f} MB")
            print(
                f"Expected O(n/p) scaling: Each GPU handles ~{100 / world_size:.0f}% of K,V"
            )

            # Show how memory scales with sequence length
            if len(successful) > 1:
                first = successful[0]
                last = successful[-1]
                memory_growth = last["avg_peak_memory_gb"] / first["avg_peak_memory_gb"]
                seq_growth = last["seq_len"] / first["seq_len"]

                print(
                    f"\nScaling from {first['seq_len']:,} to {last['seq_len']:,} tokens:"
                )
                print(f"  Sequence growth: {seq_growth:.1f}x")
                print(f"  Memory growth: {memory_growth:.1f}x")
                print(f"  Efficiency: {seq_growth / memory_growth:.2f}x (ideal: 1.0x)")

        print(f"\nResults saved to: {filename}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_scaling()
