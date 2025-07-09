#!/usr/bin/env python3
"""
Test the simple fixed ring attention with 2 GPUs.

Launch with:
torchrun --nproc_per_node=2 test_2gpu_simple_fixed.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_fixed_simple import (
    RingDilatedAttentionFixedSimple,
)


def main():
    """Test simple fixed ring attention."""
    # Setup distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Testing Simple Fixed Ring Attention with {world_size} GPUs")
        print("-" * 60)

    # Test configurations
    test_configs = [
        (8192, "8K"),
        (16384, "16K"),
        (32768, "32K"),
        (65536, "64K"),
        (100000, "100K"),
        (150000, "150K"),
        (200000, "200K"),
    ]

    max_successful = 0

    for total_seq_len, label in test_configs:
        # Make divisible by world_size
        total_seq_len = (total_seq_len // world_size) * world_size
        local_seq_len = total_seq_len // world_size

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            # Create input
            x_local = torch.randn(
                2,
                local_seq_len,
                768,  # batch_size=2
                device=device,
                dtype=torch.float32,
            ).contiguous()

            # Create model
            model = RingDilatedAttentionFixedSimple(
                embed_dim=768,
                num_heads=12,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
            )
            model.eval()

            # Synchronize
            if dist.is_initialized():
                dist.barrier()

            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                output = model(x_local, total_seq_len=total_seq_len, already_split=True)
                torch.cuda.synchronize()
            end_time = time.time()

            # Check output
            assert output.shape == (
                2,
                local_seq_len,
                768,
            ), f"Wrong output shape: {output.shape}"

            # Get memory
            mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

            if rank == 0:
                elapsed = end_time - start_time
                print(
                    f"✓ {label} ({total_seq_len:,} tokens): "
                    f"{mem_mb:.1f} MB, {elapsed:.2f}s, "
                    f"{total_seq_len / elapsed / 1e6:.2f}M tok/s"
                )

            max_successful = total_seq_len

            # Cleanup
            del x_local, model, output

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"✗ {label} ({total_seq_len:,} tokens): OOM")
            break
        except Exception as e:
            if rank == 0:
                print(f"✗ {label} ({total_seq_len:,} tokens): Error - {e}")
                import traceback

                traceback.print_exc()
            break

        # Small delay
        time.sleep(0.5)

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Maximum successful: {max_successful:,} tokens")
        print(f"Per GPU: {max_successful // world_size:,} tokens")

        if max_successful >= 200000:
            print("\n✅ Successfully processed 200K+ tokens with 2 GPUs!")
        else:
            print(f"\n❌ Could only process up to {max_successful:,} tokens")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
