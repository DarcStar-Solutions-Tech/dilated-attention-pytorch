#!/usr/bin/env python3
"""
Debug Ring V3 timeout issues with detailed logging.
Run with: torchrun --nproc_per_node=2 test_ring_v3_debug_timeout.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def debug_timeout():
    """Debug where the timeout occurs."""

    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 test_ring_v3_debug_timeout.py")
        return

    # Initialize distributed
    print(f"[PID {os.getpid()}] Initializing process group...")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized successfully")

    # Test with 1024 tokens which was timing out
    seq_len = 1024

    print(f"[Rank {rank}] Creating model...")
    model = RingDilatedAttentionV3(
        segment_lengths=[512],
        dilation_rates=[1],
        bucket_size=256,
        use_bucketed=True,
        device=device,
        dtype=torch.float32,  # Use float32 for better stability
        ring_size=world_size,
    )
    print(f"[Rank {rank}] Model created")

    # Create inputs
    print(f"[Rank {rank}] Creating inputs with seq_len={seq_len}...")
    q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)
    k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)
    v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)
    print(f"[Rank {rank}] Inputs created")

    # Add detailed logging to trace execution
    from dilated_attention_pytorch import ring_attention_utils

    # Patch all_ring_pass to add logging
    original_all_ring_pass = ring_attention_utils.all_ring_pass

    def logged_all_ring_pass(*args, **kwargs):
        ring_size = kwargs.get("ring_size", dist.get_world_size())
        print(f"[Rank {rank}] Entering all_ring_pass with ring_size={ring_size}")

        for i, result in enumerate(original_all_ring_pass(*args, **kwargs)):
            print(f"[Rank {rank}] Ring iteration {i}")
            yield result

        print(f"[Rank {rank}] Exiting all_ring_pass")

    ring_attention_utils.all_ring_pass = logged_all_ring_pass

    # Also patch the ring_pass function
    original_ring_pass = ring_attention_utils.ring_pass

    def logged_ring_pass(x, receive_buffer=None, ring_size=None):
        print(f"[Rank {rank}] ring_pass called, tensor shape: {x.shape}")
        result = original_ring_pass(x, receive_buffer, ring_size)
        print(f"[Rank {rank}] ring_pass completed")
        return result

    ring_attention_utils.ring_pass = logged_ring_pass

    # Patch send_and_receive_
    original_send_recv = ring_attention_utils.send_and_receive_

    def logged_send_recv(x, left_rank, right_rank, ring_size):
        print(
            f"[Rank {rank}] send_and_receive_: sending to {right_rank}, receiving from {left_rank}"
        )
        result = original_send_recv(x, left_rank, right_rank, ring_size)
        print(f"[Rank {rank}] send_and_receive_ completed")
        return result

    ring_attention_utils.send_and_receive_ = logged_send_recv

    try:
        print(f"[Rank {rank}] Starting forward pass...")
        output = model(q, k, v, is_causal=False)
        print(f"[Rank {rank}] Forward pass completed!")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(f"[Rank {rank}] Output mean: {output.float().mean().item():.6f}")

    except Exception as e:
        print(f"[Rank {rank}] Exception: {e}")
        import traceback

        traceback.print_exc()

    print(f"[Rank {rank}] Cleaning up...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    debug_timeout()
