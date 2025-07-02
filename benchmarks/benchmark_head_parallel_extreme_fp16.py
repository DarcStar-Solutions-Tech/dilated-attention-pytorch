#!/usr/bin/env python3
"""
Benchmark head-parallel dilated attention with FP16 for extreme sequences.
Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_head_parallel_extreme_fp16.py
"""

import os
import time
import json
from datetime import datetime
import torch
import torch.distributed as dist
import gc


def main():
    if "RANK" not in os.environ:
        print("This benchmark requires multiple GPUs.")
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_head_parallel_extreme_fp16.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("=== Head-Parallel Extreme Sequence Benchmark (FP16) ===")
        print(f"GPUs: {world_size}")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print("Using FP16 throughout for maximum capacity\n")

    # Import implementations
    from dilated_attention_pytorch.head_parallel_dilated_attention import (
        HeadParallelDilatedAttention,
    )

    # Test configurations - start from where FP32 failed
    configs = [
        {"seq": 131072, "batch": 1, "heads": 8},  # 128K - baseline
        {"seq": 262144, "batch": 1, "heads": 8},  # 256K - FP32 OOM here
        {"seq": 393216, "batch": 1, "heads": 8},  # 384K
        {"seq": 524288, "batch": 1, "heads": 8},  # 512K
        {"seq": 786432, "batch": 1, "heads": 8},  # 768K
        {"seq": 1048576, "batch": 1, "heads": 8},  # 1M
    ]

    # Segment configurations for different sizes
    segment_configs = {
        131072: ([8192, 16384, 32768], [1, 2, 4]),
        262144: ([16384, 32768, 65536], [1, 2, 4]),
        393216: ([16384, 32768, 65536], [1, 2, 4]),
        524288: ([32768, 65536, 131072], [1, 2, 4]),
        786432: ([32768, 65536, 131072], [1, 2, 4]),
        1048576: ([65536, 131072, 262144], [1, 2, 4]),
    }

    results = []
    max_seq_achieved = 0

    # Force FP16 throughout
    dtype = torch.float16

    for config in configs:
        seq_len = config["seq"]
        batch_size = config["batch"]
        num_heads = config["heads"]

        # Get segment config
        segment_lengths, dilation_rates = segment_configs.get(
            seq_len, ([32768, 65536, 131072], [1, 2, 4])
        )

        # Ensure divisibility
        max_seg = max(segment_lengths)
        if seq_len % max_seg != 0:
            seq_len = ((seq_len // max_seg) + 1) * max_seg
            config["seq"] = seq_len

        if rank == 0:
            print(
                f"\nTesting seq_len={seq_len:,} ({seq_len / 1024 / 1024:.1f}M tokens) with FP16..."
            )
            print(f"  Segments: {segment_lengths}")
            print(
                f"  Memory estimate: {seq_len * num_heads * 64 * 2 * 3 / 1024**3:.2f} GB for QKV (FP16)"
            )

        try:
            # Aggressive memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create model with FP16
            model = HeadParallelDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                use_xformers=True,
                device=device,
                dtype=dtype,  # FP16
            )

            # Create FP16 inputs
            head_dim = 64
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )

            # Single warmup
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Benchmark
            num_iters = 3
            torch.cuda.reset_peak_memory_stats()
            times = []

            for i in range(num_iters):
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    output = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()
                times.append(time.time() - start)

                if rank == 0:
                    print(f"    Iteration {i + 1}: {times[-1] * 1000:.1f}ms")

            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3

            # Calculate metrics
            tokens = batch_size * seq_len
            throughput = tokens / avg_time
            memory_per_token = peak_memory * 1024 * 1024 / tokens
            tflops = (4 * seq_len * num_heads * head_dim * seq_len) / (avg_time * 1e12)

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "num_heads": num_heads,
                "time_ms": avg_time * 1000,
                "memory_gb": peak_memory,
                "throughput": throughput,
                "memory_per_token_kb": memory_per_token,
                "tflops": tflops,
                "dtype": "float16",
            }

            results.append(result)
            max_seq_achieved = seq_len

            if rank == 0:
                print("  ✓ SUCCESS with FP16!")
                print(f"    Time: {avg_time * 1000:.1f}ms")
                print(f"    Memory: {peak_memory:.2f} GB")
                print(f"    Throughput: {throughput:,.0f} tokens/sec")
                print(f"    Memory/token: {memory_per_token:.2f} KB")
                print(f"    TFLOPS: {tflops:.2f}")

            # Cleanup
            del q, k, v, output, model
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print("  ✗ OOM with FP16 - Maximum sequence reached")
            break
        except Exception as e:
            if rank == 0:
                print(f"  ✗ Error: {type(e).__name__}: {str(e)}")
            break

    # Synchronize before final summary
    dist.barrier()

    # Save and print results
    if rank == 0 and results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"benchmarks/head_parallel_extreme_fp16_{world_size}gpu_{timestamp}.json"
        )

        with open(filename, "w") as f:
            json.dump(
                {
                    "world_size": world_size,
                    "gpu_model": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
                    / 1024**3,
                    "dtype": "float16",
                    "timestamp": timestamp,
                    "max_sequence_achieved": max_seq_achieved,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {filename}")

        # Print summary
        print("\n" + "=" * 70)
        print("FP16 EXTREME SEQUENCE BENCHMARK SUMMARY")
        print("=" * 70)
        print(
            f"Maximum sequence achieved: {max_seq_achieved:,} tokens ({max_seq_achieved / 1024 / 1024:.1f}M)"
        )

        print("\nPerformance Summary:")
        print(
            f"{'Seq Length':>12} | {'Time (ms)':>10} | {'Memory (GB)':>12} | {'Throughput':>15} | {'KB/token':>10}"
        )
        print("-" * 75)

        for r in results:
            print(
                f"{r['seq_len']:>12,} | {r['time_ms']:>10.1f} | {r['memory_gb']:>12.2f} | "
                f"{r['throughput']:>15,.0f} | {r['memory_per_token_kb']:>10.2f}"
            )

        # Compare to FP32 results
        print("\nFP16 vs FP32 Comparison:")
        print("- FP32 max: 131,072 tokens")
        print(f"- FP16 max: {max_seq_achieved:,} tokens")
        print(f"- Improvement: {max_seq_achieved / 131072:.1f}x capacity")

        print("\nKey Benefits of FP16:")
        print("- 2x memory reduction for tensors")
        print("- Faster computation on modern GPUs")
        print("- Enables processing of much longer sequences")

        if max_seq_achieved >= 262144:
            print("\n✅ Successfully exceeded FP32 limits!")
            print(
                f"   Processed {max_seq_achieved:,} tokens with head-parallel on {world_size} GPUs"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
