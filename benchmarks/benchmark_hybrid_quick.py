#!/usr/bin/env python3
"""
Quick benchmark for hybrid ring dilated attention.
Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_quick.py
"""

import os
import time
import torch
import torch.distributed as dist


def main():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_quick.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Quick Hybrid Ring Attention Benchmark")
        print(f"GPUs: {world_size}")
        print("=" * 50)

    # Import model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    # Simple test configurations
    configs = [
        {"seq": 1024, "batch": 2},
        {"seq": 2048, "batch": 2},
        {"seq": 4096, "batch": 1},
        {"seq": 8192, "batch": 1},
    ]

    for cfg in configs:
        seq_len = cfg["seq"]
        batch_size = cfg["batch"]

        try:
            # Create model with simple config
            model = RingDilatedAttentionHybrid(
                segment_lengths=[256],
                dilation_rates=[2],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,
                enable_memory_pool=False,  # Disable for simplicity
                use_flash_attention=False,
            )

            # Create inputs
            q = torch.randn(batch_size, seq_len, 8, 64, device=device)
            k = torch.randn(batch_size, seq_len, 8, 64, device=device)
            v = torch.randn(batch_size, seq_len, 8, 64, device=device)

            # Warmup
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Time it
            torch.cuda.reset_peak_memory_stats()
            start = time.time()

            with torch.no_grad():
                for _ in range(5):
                    _ = model(q, k, v, is_causal=False)
                    torch.cuda.synchronize()

            elapsed = (time.time() - start) / 5
            memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

            # Gather results
            times = torch.tensor([elapsed], device=device)
            memories = torch.tensor([memory_mb], device=device)

            dist.all_reduce(times, op=dist.ReduceOp.AVG)
            dist.all_reduce(memories, op=dist.ReduceOp.AVG)

            if rank == 0:
                tokens = batch_size * seq_len
                throughput = tokens / times.item()
                mem_per_token = memories.item() / tokens * 1024  # KB

                print(f"\nSeq={seq_len}, Batch={batch_size}:")
                print(f"  Time: {times.item() * 1000:.1f} ms")
                print(f"  Memory/GPU: {memories.item():.1f} MB")
                print(f"  Throughput: {throughput:,.0f} tokens/sec")
                print(f"  Memory/token: {mem_per_token:.2f} KB")

        except Exception as e:
            if rank == 0:
                print(f"\nSeq={seq_len}: Error - {str(e)}")

    if rank == 0:
        print("\n" + "=" * 50)
        print("Benchmark complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
