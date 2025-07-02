#!/usr/bin/env python3
"""
Quick benchmark comparing Ring Attention implementations without external dependencies.
"""

import os
import torch
import time

from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)

# Check if we're in distributed mode
IS_DISTRIBUTED = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

if IS_DISTRIBUTED:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if rank == 0:
    print("=" * 80)
    print(f"Quick Ring Attention Comparison - {world_size} GPU(s)")
    print("=" * 80)

# Test parameters
seq_len = 4096
batch_size = 1
num_heads = 8
head_dim = 64

implementations = [
    ("Collective", RingDilatedAttentionV2Collective),
]

for impl_name, impl_class in implementations:
    try:
        # Create model
        model = impl_class(
            segment_lengths=[1024, 1024, 1024, 1024],
            dilation_rates=[1, 2, 4, 8],
            ring_size=world_size if IS_DISTRIBUTED else 1,
            device=device,
        )

        # Force distributed mode for ring implementations
        if hasattr(model, "min_seq_length_for_ring"):
            model.min_seq_length_for_ring = 1

        dtype = model.dtype

        # Create input
        x = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(x, x, x, is_causal=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Time
        if device.type == "cuda":
            torch.cuda.synchronize()
        if IS_DISTRIBUTED:
            dist.barrier()
        start = time.time()

        iterations = 5
        with torch.no_grad():
            for _ in range(iterations):
                output = model(x, x, x, is_causal=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        if IS_DISTRIBUTED:
            dist.barrier()
        avg_time = (time.time() - start) / iterations

        # Memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model(x, x, x, is_causal=False)

            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0

        if rank == 0:
            print(f"\n{impl_name}:")
            print(f"  Time: {avg_time * 1000:.1f} ms")
            print(f"  Memory: {peak_memory_mb:.1f} MB")

    except Exception as e:
        if rank == 0:
            print(f"\n{impl_name}:")
            print(f"  Error: {str(e)[:80]}...")

    if IS_DISTRIBUTED:
        dist.barrier()

if IS_DISTRIBUTED:
    dist.destroy_process_group()

if rank == 0:
    print("\n" + "=" * 80)
    print("Summary:")
    print("- Collective: Uses all-gather for ring communication")
    print("- This is now the only Ring Attention V2 implementation")
    print("=" * 80)
