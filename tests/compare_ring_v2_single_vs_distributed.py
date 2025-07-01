#!/usr/bin/env python3
"""
Compare RingDilatedAttentionV2Collective performance: single GPU vs distributed.

Usage:
  # Single GPU
  python compare_ring_v2_single_vs_distributed.py

  # Multi-GPU
  torchrun --nproc_per_node=2 compare_ring_v2_single_vs_distributed.py
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def measure_performance(model, q, k, v, num_iterations=10):
    """Measure forward pass performance (backward has issues in distributed mode)."""
    _ = q.device

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(q, k, v, is_causal=True)

    # Forward pass timing
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(q, k, v, is_causal=True)
            torch.cuda.synchronize()

    forward_time = (time.time() - start) / num_iterations

    # Note: Backward pass has in-place operation issues in distributed mode
    # so we'll only measure forward pass
    backward_time = 0.0

    return forward_time * 1000, backward_time  # Convert to ms


def main():
    # Check if distributed
    is_distributed = "WORLD_SIZE" in os.environ

    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    seq_lengths = [2048, 4096, 8192]
    batch_size = 1
    num_heads = 8
    head_dim = 64

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Ring Attention V2 Collective Performance Comparison")
        print(f"World Size: {world_size}")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

    results = []

    for seq_length in seq_lengths:
        if rank == 0:
            print(f"\nTesting sequence length: {seq_length}")

        try:
            # Determine optimal dtype for GPU
            # Pascal (compute 6.x) should use FP32
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability(device)
                if compute_capability[0] < 7:  # Pascal or older
                    dtype = torch.float32
                else:
                    dtype = torch.float16
            else:
                dtype = torch.float32

            if rank == 0:
                print(f"  Using dtype: {dtype}")

            # Create model with ring_size = world_size
            model = RingDilatedAttentionV2Collective(
                segment_lengths=[2048, 4096, 8192],
                dilation_rates=[1, 2, 4],
                ring_size=world_size,
                device=device,
                dtype=dtype,
                use_flash_attention=True,
            )

            # Create inputs (no grad for now due to distributed issues)
            shape = (batch_size, seq_length, num_heads, head_dim)
            q = torch.randn(shape, device=device, dtype=dtype)
            k = torch.randn(shape, device=device, dtype=dtype)
            v = torch.randn(shape, device=device, dtype=dtype)

            # Measure performance
            forward_ms, backward_ms = measure_performance(model, q, k, v)

            # Calculate metrics
            total_ms = forward_ms + backward_ms
            throughput = (batch_size * seq_length) / (total_ms / 1000)

            # Memory usage
            allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024

            result = {
                "seq_length": seq_length,
                "world_size": world_size,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "total_ms": total_ms,
                "throughput": throughput,
                "memory_mb": allocated_mb,
            }
            results.append(result)

            if rank == 0:
                print(f"  Forward: {forward_ms:.2f} ms")
                print(f"  Backward: {backward_ms:.2f} ms")
                print(f"  Total: {total_ms:.2f} ms")
                print(f"  Throughput: {throughput:.0f} tokens/sec")
                print(f"  Memory: {allocated_mb:.1f} MB")

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"  OOM for sequence length {seq_length}")
        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")

    # Summary
    if rank == 0 and results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(
            f"{'Seq Len':<10} {'GPUs':<6} {'Forward':<10} {'Backward':<10} {'Total':<10} {'Throughput':<12}"
        )
        print("-" * 60)

        for r in results:
            print(
                f"{r['seq_length']:<10} {r['world_size']:<6} "
                f"{r['forward_ms']:<10.1f} {r['backward_ms']:<10.1f} "
                f"{r['total_ms']:<10.1f} {r['throughput']:<12.0f}"
            )

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
