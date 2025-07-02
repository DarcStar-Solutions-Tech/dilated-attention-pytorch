#!/usr/bin/env python3
"""
Debug hybrid forward pass to find where it hangs.
Run with: torchrun --nproc_per_node=2 benchmarks/debug_hybrid_forward.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def debug_forward():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/debug_hybrid_forward.py"
        )
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Creating model...")

    # Patch the forward method to add debug prints
    original_forward = RingDilatedAttentionHybrid.forward

    def debug_forward_wrapper(self, q, k, v, is_causal=False):
        print(
            f"[Rank {self.rank}] Forward called with shapes: q={q.shape}, k={k.shape}, v={v.shape}"
        )
        print(f"[Rank {self.rank}] Ring size: {self.ring_size}")

        # Call original forward with more debug info
        try:
            result = original_forward(self, q, k, v, is_causal)
            print(f"[Rank {self.rank}] Forward completed successfully!")
            return result
        except Exception as e:
            print(f"[Rank {self.rank}] Forward failed with error: {e}")
            raise

    RingDilatedAttentionHybrid.forward = debug_forward_wrapper

    # Create model
    model = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[1],
        ring_size=world_size,
        device=device,
    )

    print(f"[Rank {rank}] Model created")

    # Create small inputs
    seq_len = 512
    q = torch.randn(1, seq_len, 4, 32, device=device, dtype=model.dtype) * 0.1
    k = torch.randn(1, seq_len, 4, 32, device=device, dtype=model.dtype) * 0.1
    v = torch.randn(1, seq_len, 4, 32, device=device, dtype=model.dtype) * 0.1

    print(f"[Rank {rank}] Starting forward pass...")

    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    print(f"[Rank {rank}] Output shape: {output.shape}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    debug_forward()
