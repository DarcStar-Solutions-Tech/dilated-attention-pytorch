#!/usr/bin/env python3
"""
Debug the dtype mismatch issue in distributed mode.
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def main():
    # Setup distributed
    if "WORLD_SIZE" not in os.environ:
        print("This script must be run with torchrun")
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("\nDebugging dtype issue in distributed mode")
        print(f"World size: {world_size}\n")

    # Create model with float32
    model = RingDilatedAttentionV2Collective(
        segment_lengths=[2048, 2048],
        dilation_rates=[1, 1],
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
    )

    # Create float32 input
    seq_len = 4096
    x = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32)

    if rank == 0:
        print(f"Model dtype: {model.dtype}")
        print(f"Input dtype: {x.dtype}")
        print(f"Input shape: {x.shape}")

    # Add debug hook to track dtype changes
    def check_tensors(name, tensor):
        if tensor is not None:
            print(f"[Rank {rank}] {name}: shape={tensor.shape}, dtype={tensor.dtype}")

    try:
        # Trace through forward pass
        with torch.no_grad():
            if rank == 0:
                print("\nStarting forward pass...")

            # Manually trace through _ring_attention to find dtype issue
            b, n, h, d = x.shape
            chunk_size = (n + world_size - 1) // world_size

            # Get local chunks
            local_start = rank * chunk_size
            local_end = min((rank + 1) * chunk_size, n)

            k_local = x[:, local_start:local_end].contiguous()
            if rank == 0:
                check_tensors("k_local", k_local)

            # Check what _apply_dilated_patterns_to_chunk returns
            k_dilated, v_dilated = model._apply_dilated_patterns_to_chunk(
                k_local, k_local, local_start, local_end - local_start
            )

            if rank == 0:
                check_tensors("k_dilated", k_dilated)

                # Check internal buffers
                if (
                    hasattr(model, "_k_chunks_list")
                    and model._k_chunks_list is not None
                ):
                    print(f"_k_chunks_list length: {len(model._k_chunks_list)}")
                    if len(model._k_chunks_list) > 0:
                        check_tensors("_k_chunks_list[0]", model._k_chunks_list[0])

            # Try the full forward
            output = model(x, x, x, is_causal=False)

            if rank == 0:
                print(f"\nSuccess! Output shape: {output.shape}, dtype: {output.dtype}")

    except Exception as e:
        if rank == 0:
            print(f"\nError: {e}")
            print(f"Error type: {type(e)}")

            # Try to find where the dtype mismatch happens
            import traceback

            traceback.print_exc()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
