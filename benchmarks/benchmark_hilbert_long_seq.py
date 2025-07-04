#!/usr/bin/env python3
"""Test Hilbert ring attention with very long sequences."""

import os
import torch
import torch.distributed as dist
import time
import json
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


def test_long_sequences():
    """Test with progressively longer sequences."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("TESTING LONG SEQUENCES WITH HILBERT RING ATTENTION")
        print(f"{'=' * 70}")
        print(f"Running on {world_size} GPU(s)")
        print("Target: 262K tokens like the original implementation\n")

    # Import after setting device
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Test configurations - start where we left off
    configs = [
        # (seq_len, batch_size, segment_lengths, dilation_rates)
        (65536, 1, [8192], [1]),  # 64K
        (131072, 1, [16384], [1]),  # 128K
        (262144, 1, [32768], [1]),  # 262K - target
    ]

    results = []

    for seq_len, batch_size, segment_lengths, dilation_rates in configs:
        # Skip if not divisible by world size
        if seq_len % world_size != 0:
            continue

        if rank == 0:
            print(f"Testing {seq_len:,} tokens:")
            print(f"  Per GPU: {seq_len // world_size:,} tokens")
            print(f"  Segments: {segment_lengths}, Dilation: {dilation_rates}")

        try:
            # Create model
            model = RingDilatedAttentionHybridHilbert(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,
                use_hilbert=True,
                hilbert_chunk_size=min(8192, seq_len // world_size),
            )

            # Create inputs
            num_heads = 8
            head_dim = 64
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Memory before
            if rank == 0:
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                print(f"  Memory before: {allocated_before:.2f} GB")

            # Warmup
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)

            if dist.is_initialized():
                dist.barrier()

            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                for i in range(3):
                    output = model(q, k, v, is_causal=False)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 3

            # Results
            if rank == 0:
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                print(f"  Time: {elapsed * 1000:.1f} ms")
                print(f"  Throughput: {seq_len / elapsed:,.0f} tokens/sec")
                print(f"  Memory after: {allocated_after:.2f} GB")
                print(f"  Memory used: {allocated_after - allocated_before:.2f} GB")
                print("  ✓ SUCCESS\n")

                results.append(
                    {
                        "seq_len": seq_len,
                        "world_size": world_size,
                        "time_ms": elapsed * 1000,
                        "tokens_per_sec": seq_len / elapsed,
                        "memory_gb": allocated_after,
                    }
                )

            # Clean up
            del q, k, v, output, model
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"  ✗ FAILED: {e}\n")
            break

    # Summary
    if rank == 0 and results:
        print(f"{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}\n")

        max_seq = max(r["seq_len"] for r in results)
        print(f"Maximum successful sequence length: {max_seq:,} tokens")

        if max_seq >= 262144:
            print("\n🎉 ACHIEVED 262K TOKEN TARGET WITH HILBERT ORDERING! 🎉")

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"hilbert_long_seq_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "results": results,
                    "max_seq_len": max_seq,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_long_sequences()
