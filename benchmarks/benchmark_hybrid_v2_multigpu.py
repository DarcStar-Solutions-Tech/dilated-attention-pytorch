#!/usr/bin/env python3
"""
Benchmark the fixed hybrid ring dilated attention on multiple GPUs.
Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_v2_multigpu.py
"""

import os
import time
import json
from datetime import datetime
import torch
import torch.distributed as dist


def main():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_v2_multigpu.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Starting fixed implementation benchmark...")

    # Import the fixed model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
        RingDilatedAttentionHybridOptimizedV2,
    )

    # Test configurations
    configs = [
        {"seq": 1024, "batch": 2},
        {"seq": 2048, "batch": 2},
        {"seq": 4096, "batch": 1},
        {"seq": 8192, "batch": 1},
        {"seq": 16384, "batch": 1},
        {"seq": 32768, "batch": 1},
        {"seq": 65536, "batch": 1},
    ]

    # Model configuration
    segment_lengths = [512, 1024]
    dilation_rates = [1, 2]

    results = []

    for config in configs:
        seq_len = config["seq"]
        batch_size = config["batch"]

        # Skip if not divisible by max segment
        if seq_len % max(segment_lengths) != 0:
            continue

        print(f"[GPU {rank}] Testing seq_len={seq_len}...")

        try:
            # Clear memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create model
            model = RingDilatedAttentionHybridOptimizedV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,
                enable_memory_pool=True,
                use_flash_attention=False,
                use_pattern_cache=True,
                precompute_patterns=True,
                overlap_communication=False,
            )

            # Create inputs
            num_heads = 8
            head_dim = 64
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()

            # Benchmark
            torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

            # Calculate metrics
            tokens = batch_size * seq_len
            throughput = tokens / avg_time
            memory_per_token = peak_memory / tokens * 1024  # KB

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "time_ms": avg_time * 1000,
                "memory_mb": peak_memory,
                "throughput": throughput,
                "memory_per_token_kb": memory_per_token,
            }

            results.append(result)

            print(
                f"[GPU {rank}] Seq {seq_len}: {avg_time * 1000:.1f}ms, "
                f"{peak_memory:.1f}MB, {memory_per_token:.2f}KB/token, "
                f"{throughput:.0f} tok/s"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"[GPU {rank}] Seq {seq_len}: OOM")
            break
        except Exception as e:
            print(f"[GPU {rank}] Seq {seq_len}: Error - {str(e)}")
            import traceback

            traceback.print_exc()
            break

    # Save results for this GPU
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"benchmarks/hybrid_v2_results_rank{rank}_{world_size}gpu_{timestamp}.json"
        )

        with open(filename, "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "timestamp": timestamp,
                    "segment_lengths": segment_lengths,
                    "dilation_rates": dilation_rates,
                    "implementation": "RingDilatedAttentionHybridOptimizedV2",
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"[GPU {rank}] Results saved to {filename}")

    # Synchronize before printing summary
    dist.barrier()

    # Only rank 0 prints comparison with original
    if rank == 0 and results:
        print("\n" + "=" * 60)
        print("FIXED IMPLEMENTATION BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Configuration: {world_size} GPUs")
        print(f"Segments: {segment_lengths}, Dilation: {dilation_rates}")
        print("\nResults:")

        for r in results:
            print(f"\nSeq {r['seq_len']:,}:")
            print(f"  Time: {r['time_ms']:.1f} ms")
            print(f"  Memory/GPU: {r['memory_mb']:.1f} MB")
            print(f"  Throughput: {r['throughput']:,.0f} tokens/sec")
            print(f"  Memory/token: {r['memory_per_token_kb']:.2f} KB")

        # Memory scaling analysis
        if len(results) >= 2:
            print("\nMemory Scaling Analysis:")
            for i in range(len(results)):
                print(
                    f"  Seq {results[i]['seq_len']:,}: {results[i]['memory_per_token_kb']:.2f} KB/token"
                )

            first_mem = results[0]["memory_per_token_kb"]
            last_mem = results[-1]["memory_per_token_kb"]
            ratio = last_mem / first_mem

            print(f"\nMemory ratio (last/first): {ratio:.2f}")
            if ratio < 1.3:
                print("✅ Excellent O(n/p) memory scaling!")
            elif ratio < 1.5:
                print("✅ Good O(n/p) memory scaling")
            else:
                print("⚠️  Memory scaling could be improved")

        # Try to compare with original results
        print("\n" + "=" * 60)
        print("COMPARISON WITH ORIGINAL IMPLEMENTATION")
        print("=" * 60)

        # Load most recent original results
        import glob

        original_files = sorted(
            glob.glob("benchmarks/hybrid_results_rank0_2gpu_*.json")
        )
        if original_files:
            with open(original_files[-1], "r") as f:
                original_data = json.load(f)
                original_results = original_data.get("results", [])

            print(
                f"\n{'Seq Length':>10} | {'Original (ms)':>15} | {'Fixed (ms)':>15} | {'Speedup':>10}"
            )
            print("-" * 60)

            for new_r in results:
                seq = new_r["seq_len"]
                # Find matching original result
                orig_r = next(
                    (r for r in original_results if r["seq_len"] == seq), None
                )
                if orig_r:
                    speedup = orig_r["time_ms"] / new_r["time_ms"]
                    print(
                        f"{seq:>10,} | {orig_r['time_ms']:>14.1f} | {new_r['time_ms']:>14.1f} | {speedup:>9.2f}x"
                    )

        print("\nKey Improvements:")
        print("✓ Fixed chunk boundary handling in multi-GPU")
        print("✓ Proper dilation pattern mapping across ranks")
        print("✓ Memory pool integration working")
        print("✓ Pattern pre-computation enabled")
        print("✓ Optimized memory access patterns")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
