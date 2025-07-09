#!/usr/bin/env python3
"""
Test ring attention with 2 GPUs - careful implementation.

Launch with:
torchrun --nproc_per_node=2 test_2gpu_ring_attention.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_correct import (
    RingDilatedAttentionHilbertOptimizedCorrect,
)


def main():
    """Test ring attention with 2 GPUs."""
    # Setup distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Testing ring attention with {world_size} GPUs")
        print("-" * 60)

    # Start with small sequence
    total_seq_len = 16384  # 16K tokens
    local_seq_len = total_seq_len // world_size
    batch_size = 1
    embed_dim = 768

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create local input - ensure contiguous
        x_local = torch.randn(
            batch_size, local_seq_len, embed_dim, device=device, dtype=torch.float32
        ).contiguous()

        if rank == 0:
            print(f"Created input: {x_local.shape}")
            print(f"Input is contiguous: {x_local.is_contiguous()}")

        # Synchronize before model creation
        if dist.is_initialized():
            dist.barrier()

        # Create model - let it detect distributed settings
        model = RingDilatedAttentionHilbertOptimizedCorrect(
            embed_dim=embed_dim,
            num_heads=12,
            segment_lengths=[2048, 4096],  # Smaller segments
            dilation_rates=[1, 2],
            dropout=0.0,
            use_hilbert=False,  # Disable Hilbert for now
            device=device,
            dtype=torch.float32,
            memory_efficient=True,
        )

        model = model.to(device)
        model.eval()

        if rank == 0:
            print(
                f"Model created with world_size={model.world_size}, rank={model.rank}"
            )

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Test forward pass
        with torch.no_grad():
            if rank == 0:
                print("Running forward pass...")

            # Ensure input is contiguous
            x_local = x_local.contiguous()

            output = model(x_local, total_seq_len=total_seq_len, already_split=True)

            torch.cuda.synchronize()

            if rank == 0:
                print(f"Output shape: {output.shape}")
                print(f"Output is contiguous: {output.is_contiguous()}")

        # Check memory
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        # Gather memory from all ranks
        mem_tensor = torch.tensor([mem_mb], device=device)
        if dist.is_initialized():
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)

        if rank == 0:
            print("\n✓ SUCCESS!")
            print(f"Processed {total_seq_len:,} tokens")
            print(f"Max memory per GPU: {mem_tensor.item():.1f} MB")
            print(f"Memory per token: {mem_tensor.item() / local_seq_len:.4f} MB")

        # Now try larger sequences
        if rank == 0:
            print("\nTesting larger sequences...")

        test_lengths = [32768, 65536, 100000, 150000, 200000]

        for seq_len in test_lengths:
            # Make divisible by world_size
            seq_len = (seq_len // world_size) * world_size
            local_seq = seq_len // world_size

            # Cleanup
            del x_local, output
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if dist.is_initialized():
                dist.barrier()

            try:
                # Create new input
                x_local = torch.randn(
                    batch_size, local_seq, embed_dim, device=device, dtype=torch.float32
                ).contiguous()

                # Forward pass
                with torch.no_grad():
                    output = model(x_local, total_seq_len=seq_len, already_split=True)
                    torch.cuda.synchronize()

                # Get memory
                mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

                if rank == 0:
                    print(f"✓ {seq_len:,} tokens: {mem_mb:.1f} MB per GPU")

                del x_local, output

            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"✗ {seq_len:,} tokens: OOM")
                break
            except Exception as e:
                if rank == 0:
                    print(f"✗ {seq_len:,} tokens: Error - {e}")
                break

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
