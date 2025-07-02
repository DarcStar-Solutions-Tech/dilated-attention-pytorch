#!/usr/bin/env python3
"""
Debug Ring V3 multi-GPU deadlock with detailed logging.
Run with: torchrun --nproc_per_node=2 debug_ring_v3_deadlock.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def debug_ring_communication():
    """Debug ring communication with detailed logging."""

    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 debug_ring_v3_deadlock.py")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized, world_size={world_size}")
    dist.barrier()

    # Test case medium size
    seq_len = 1024  # Medium sequence
    batch_size = 1
    num_heads = 4
    head_dim = 32

    print(f"[Rank {rank}] Creating model...")

    model = RingDilatedAttentionV3(
        segment_lengths=[32],
        dilation_rates=[1],
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    print(f"[Rank {rank}] Model created")

    # Create inputs
    print(f"[Rank {rank}] Creating inputs...")
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    print(
        f"[Rank {rank}] Inputs created, shapes: q={q.shape}, k={k.shape}, v={v.shape}"
    )

    # Add hooks to trace execution

    # Monkey-patch all_ring_pass to add logging
    from dilated_attention_pytorch import ring_attention_utils

    original_all_ring_pass = ring_attention_utils.all_ring_pass

    def logged_all_ring_pass(*args, **kwargs):
        print(
            f"[Rank {rank}] Entering all_ring_pass with ring_size={kwargs.get('ring_size', 'default')}"
        )
        try:
            for i, result in enumerate(original_all_ring_pass(*args, **kwargs)):
                print(f"[Rank {rank}] Ring pass iteration {i}, yielding result")
                yield result
                print(f"[Rank {rank}] Ring pass iteration {i} consumed")
        except Exception as e:
            print(f"[Rank {rank}] Exception in all_ring_pass: {e}")
            raise
        print(f"[Rank {rank}] Exiting all_ring_pass")

    ring_attention_utils.all_ring_pass = logged_all_ring_pass

    # Also patch ring_pass
    original_ring_pass = ring_attention_utils.ring_pass

    def logged_ring_pass(*args, **kwargs):
        print(f"[Rank {rank}] Calling ring_pass")
        result = original_ring_pass(*args, **kwargs)
        print(f"[Rank {rank}] Ring_pass completed")
        return result

    ring_attention_utils.ring_pass = logged_ring_pass

    # Try forward pass
    try:
        print(f"[Rank {rank}] Starting forward pass...")
        output = model(q, k, v, is_causal=False)
        print(f"[Rank {rank}] ✅ Forward pass completed! Output shape: {output.shape}")

        # Sync
        dist.barrier()
        print(f"[Rank {rank}] ✅ Barrier passed!")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Clean up
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group destroyed")


if __name__ == "__main__":
    debug_ring_communication()
