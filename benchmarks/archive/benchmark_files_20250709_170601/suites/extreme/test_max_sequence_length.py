#!/usr/bin/env python3
"""Test maximum sequence lengths achievable with Hilbert Ring Attention."""

import torch
import torch.distributed as dist
import os
import time
from typing import Optional, Tuple


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


def test_sequence_length(
    seq_len: int,
    segments: list,
    dilations: list,
    batch_size: int = 1,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """Test if a sequence length works and return throughput."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    # Import after device setup
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Model parameters
    num_heads = 8
    head_dim = 64
    _ = num_heads * head_dim
    dtype = torch.float16  # Use fp16 for memory efficiency

    try:
        # Create model
        model = RingDilatedAttentionHybridHilbert(
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=min(4096, seq_len // world_size),
            enable_memory_pool=True,
            use_xformers=False,  # Disable for GTX 1080
        ).eval()

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Test forward pass
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start

        # Calculate memory usage
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Calculate throughput
        throughput = (batch_size * seq_len) / forward_time

        # Cleanup
        del q, k, v, output, model
        torch.cuda.empty_cache()

        return True, throughput, memory_gb

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, None, None
    except Exception as e:
        if rank == 0:
            print(f"Error: {str(e)}")
        torch.cuda.empty_cache()
        return False, None, None


def find_max_sequence_length(
    min_seq: int,
    max_seq: int,
    segments: list,
    dilations: list,
    batch_size: int = 1,
) -> int:
    """Binary search to find maximum sequence length."""
    rank, _ = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print(f"\nFinding max sequence length for {world_size} GPU(s)")
        print(f"Segments: {segments}, Dilation: {dilations}")

    # Ensure sequence lengths are valid
    max_segment = max(segments)
    min_seq = ((min_seq + max_segment - 1) // max_segment) * max_segment
    max_seq = ((max_seq + max_segment - 1) // max_segment) * max_segment

    # Also ensure divisible by world_size
    min_seq = ((min_seq + world_size - 1) // world_size) * world_size
    max_seq = ((max_seq + world_size - 1) // world_size) * world_size

    best_seq_len = 0
    best_throughput = 0
    best_memory = 0

    while min_seq <= max_seq:
        mid_seq = (min_seq + max_seq) // 2
        # Round to nearest valid sequence length
        mid_seq = ((mid_seq + max_segment - 1) // max_segment) * max_segment
        mid_seq = ((mid_seq + world_size - 1) // world_size) * world_size

        if rank == 0:
            print(f"  Testing {mid_seq:,} tokens... ", end="", flush=True)

        success, throughput, memory = test_sequence_length(
            mid_seq, segments, dilations, batch_size
        )

        if success:
            if rank == 0:
                print(f"✓ ({throughput:,.0f} tokens/sec, {memory:.1f}GB)")
            best_seq_len = mid_seq
            best_throughput = throughput
            best_memory = memory
            min_seq = mid_seq + max_segment
        else:
            if rank == 0:
                print("✗ (OOM)")
            max_seq = mid_seq - max_segment

    return best_seq_len, best_throughput, best_memory


def main():
    rank, _ = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("=" * 60)
        print("MAXIMUM SEQUENCE LENGTH TEST")
        print("=" * 60)
        print(f"World size: {world_size} GPU(s)")
        print("Testing with batch size: 1")

    # Test configurations
    configs = [
        # Standard configuration
        ([2048, 4096, 8192], [1, 2, 4]),
        # Large segments for longer sequences
        ([8192, 16384, 32768], [1, 2, 4]),
        # High dilation for efficiency
        ([4096, 16384], [1, 4]),
        # Single large segment
        ([16384], [1]),
        ([16384], [2]),
    ]

    results = []

    for segments, dilations in configs:
        # Start from 16K * world_size
        min_seq = 16384 * world_size
        # Try up to 1M tokens
        max_seq = 1048576

        max_len, throughput, memory = find_max_sequence_length(
            min_seq, max_seq, segments, dilations
        )

        if rank == 0 and max_len > 0:
            print(f"\nMax sequence: {max_len:,} tokens")
            print(f"Throughput: {throughput:,.0f} tokens/sec")
            print(f"Memory per GPU: {memory:.2f} GB")

            results.append(
                {
                    "segments": segments,
                    "dilations": dilations,
                    "max_seq_len": max_len,
                    "throughput": throughput,
                    "memory_gb": memory,
                    "world_size": world_size,
                }
            )

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Sort by max sequence length
        results.sort(key=lambda x: x["max_seq_len"], reverse=True)

        print(f"\n{'Config':<30} | {'Max Seq':<12} | {'Throughput':<15}")
        print("-" * 60)

        for r in results:
            config = f"S={r['segments'][0]}+ D={r['dilations'][-1]}"
            print(f"{config:<30} | {r['max_seq_len']:>11,} | {r['throughput']:>14,.0f}")

        # Best config
        best = results[0]
        print(f"\nBest configuration achieved {best['max_seq_len']:,} tokens")
        print(f"Segments: {best['segments']}")
        print(f"Dilations: {best['dilations']}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
