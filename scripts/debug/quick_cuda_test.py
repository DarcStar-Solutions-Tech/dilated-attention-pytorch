#!/usr/bin/env python3
"""
Quick test to verify CUDA fix with small sequences.
"""

import os
import torch
import torch.distributed as dist

from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

print(f"[Rank {rank}] Testing with world_size={world_size}")

# Small test parameters
seq_len = 1024
batch_size = 1
num_heads = 8
head_dim = 64

try:
    # Create model
    model = RingDilatedAttentionV2Collective(
        segment_lengths=[256, 256, 256, 256],
        dilation_rates=[1, 2, 4, 8],
        ring_size=world_size,
        device=device,
    )

    # Force distributed mode
    model.min_seq_length_for_ring = 1

    # Create input
    x = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
    )

    # Run forward pass
    with torch.no_grad():
        output = model(x, x, x, is_causal=False)

    print(f"[Rank {rank}] ✅ SUCCESS - Output shape: {output.shape}")

except Exception as e:
    print(f"[Rank {rank}] ❌ FAILED - {type(e).__name__}: {str(e)[:100]}")

# Cleanup
dist.barrier()
dist.destroy_process_group()
