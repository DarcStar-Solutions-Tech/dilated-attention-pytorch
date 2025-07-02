#!/usr/bin/env python3
"""
Simple test to compare RingDilatedAttentionV2Collective on 1 vs 2 GPUs.
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def setup_distributed():
    """Setup distributed environment."""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def run_benchmark(seq_length=8192, batch_size=1, num_heads=8, head_dim=64):
    """Run a simple benchmark."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Print setup info
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Running with {world_size} GPU(s)")
        print(f"Sequence length: {seq_length}")
        print(f"Batch size: {batch_size}")
        print(f"{'=' * 60}\n")

    # Create model
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    model = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        ring_size=world_size,  # Use world_size as ring_size
        use_flash_attention=True,
        device=device,
        dtype=torch.float16,
    ).to(device)

    # Create input tensors
    shape = (batch_size, seq_length, num_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)

    # Warmup
    if rank == 0:
        print("Warming up...")
    for _ in range(3):
        output = model(q, k, v, is_causal=True)
        loss = output.sum()
        loss.backward()

    if world_size > 1:
        dist.barrier()

    # Benchmark forward pass
    torch.cuda.synchronize()
    start_time = time.time()

    num_iterations = 10
    for _ in range(num_iterations):
        output = model(q, k, v, is_causal=True)
        torch.cuda.synchronize()

    forward_time = (time.time() - start_time) / num_iterations * 1000  # ms

    # Benchmark backward pass
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        output = model(q, k, v, is_causal=True)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

    total_time = (time.time() - start_time) / num_iterations * 1000  # ms
    backward_time = total_time - forward_time

    # Calculate metrics
    total_tokens = batch_size * seq_length
    throughput = total_tokens / (total_time / 1000)  # tokens/sec

    # Get memory usage
    allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024

    # Print results (only from rank 0)
    if rank == 0:
        print("\nResults:")
        print(f"  Forward time: {forward_time:.2f} ms")
        print(f"  Backward time: {backward_time:.2f} ms")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Throughput: {throughput:.0f} tokens/sec")
        print(f"  Memory allocated: {allocated_mb:.2f} MB")
        print(f"  Memory reserved: {reserved_mb:.2f} MB")

        if world_size > 1:
            # Calculate scaling efficiency
            # Note: This is approximate since we don't have single GPU baseline here
            ideal_speedup = world_size
            actual_speedup = throughput / (total_tokens / (forward_time / 1000))
            efficiency = (actual_speedup / ideal_speedup) * 100
            print(f"  Scaling efficiency: ~{efficiency:.1f}%")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Test different sequence lengths
    for seq_len in [8192, 16384]:
        try:
            run_benchmark(seq_length=seq_len)
        except torch.cuda.OutOfMemoryError:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"OOM for sequence length {seq_len}")
        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Error for sequence length {seq_len}: {e}")
