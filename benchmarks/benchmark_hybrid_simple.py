#!/usr/bin/env python3
"""
Simple benchmark for hybrid ring dilated attention on multiple GPUs.
Run with: torchrun --nproc_per_node=<num_gpus> benchmarks/benchmark_hybrid_simple.py
"""

import os
import time
import torch
import torch.distributed as dist
from datetime import datetime


def main():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=<num_gpus> benchmarks/benchmark_hybrid_simple.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Benchmarking Hybrid Ring Dilated Attention")
        print(f"Number of GPUs: {world_size}")
        print("=" * 60)

    # Import model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    # Test configurations
    configs = [
        {"batch": 4, "seq_len": 2048, "heads": 16, "dim": 64},
        {"batch": 2, "seq_len": 4096, "heads": 16, "dim": 64},
        {"batch": 1, "seq_len": 8192, "heads": 16, "dim": 64},
        {"batch": 1, "seq_len": 16384, "heads": 16, "dim": 64},
        {"batch": 1, "seq_len": 32768, "heads": 16, "dim": 64},
    ]

    # Segment configuration
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    results = []

    for config in configs:
        batch_size = config["batch"]
        seq_len = config["seq_len"]
        num_heads = config["heads"]
        head_dim = config["dim"]

        # Skip if sequence length not divisible by largest segment
        if seq_len % max(segment_lengths) != 0:
            continue

        try:
            # Create model
            model = RingDilatedAttentionHybrid(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,  # Use float32 for Pascal GPUs
                enable_memory_pool=True,
                use_flash_attention=False,  # Disable for now due to API issue
            )

            # Create inputs
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
            k = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
            v = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()

            # Time forward pass
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

            # Calculate metrics
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / avg_time
            ms_per_batch = avg_time * 1000
            memory_per_token = peak_memory / total_tokens * 1024  # KB

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "time_ms": ms_per_batch,
                "memory_mb": peak_memory,
                "tokens_per_sec": tokens_per_sec,
                "memory_per_token_kb": memory_per_token,
            }

            # Gather results
            all_results = [None] * world_size
            dist.all_gather_object(all_results, result)

            if rank == 0:
                # Average across GPUs
                avg_time_ms = sum(r["time_ms"] for r in all_results) / world_size
                avg_memory = sum(r["memory_mb"] for r in all_results) / world_size
                total_throughput = sum(r["tokens_per_sec"] for r in all_results)

                print(f"\nSeq={seq_len}, Batch={batch_size}:")
                print(f"  Time: {avg_time_ms:.1f} ms")
                print(f"  Memory/GPU: {avg_memory:.1f} MB")
                print(f"  Throughput: {total_throughput:,.0f} tokens/sec")
                print(f"  Memory/token: {memory_per_token:.2f} KB")

                results.append(
                    {
                        "config": config,
                        "metrics": {
                            "time_ms": avg_time_ms,
                            "memory_mb": avg_memory,
                            "throughput": total_throughput,
                            "memory_per_token_kb": memory_per_token,
                        },
                    }
                )

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"\nSeq={seq_len}: Out of memory")
        except Exception as e:
            if rank == 0:
                print(f"\nSeq={seq_len}: Error - {str(e)}")

    # Save results
    if rank == 0 and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmarks/hybrid_results_{world_size}gpu_{timestamp}.json"

        import json

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "num_gpus": world_size,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {filename}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Segment lengths: {segment_lengths}")
        print(f"Dilation rates: {dilation_rates}")

        for r in results:
            cfg = r["config"]
            m = r["metrics"]
            print(f"\nSeq={cfg['seq_len']:,}:")
            print(f"  Throughput: {m['throughput']:,.0f} tokens/sec")
            print(f"  Memory/GPU: {m['memory_mb']:.1f} MB")
            print(
                f"  Memory scaling: O(n/{world_size}) = O({cfg['seq_len']}/{world_size})"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
