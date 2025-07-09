#!/usr/bin/env python3
"""
Test Ring Dilated Attention V3 with 2 GPUs.

Launch with:
torchrun --nproc_per_node=2 test_ring_v3_2gpu.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def setup_distributed():
    """Setup distributed training."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def test_sequence(
    total_seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    local_rank: int,
    test_backward: bool = False,
) -> tuple[bool, float, float]:
    """Test a specific sequence length."""
    device = torch.device(f"cuda:{local_rank}")
    local_seq_len = total_seq_len // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create local input
        x_local = torch.randn(
            batch_size,
            local_seq_len,
            768,
            device=device,
            dtype=torch.float32,
            requires_grad=test_backward,
        )

        # Create model
        model = RingDilatedAttentionV3(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            dropout=0.0,
            use_hilbert=False,  # Start without Hilbert
            device=device,
            dtype=torch.float32,
        )

        if test_backward:
            model.train()
        else:
            model.eval()

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Forward pass
        start_time = time.time()
        output = model(x_local, total_seq_len=total_seq_len, already_split=True)

        if test_backward:
            # Backward pass
            loss = output.mean()
            loss.backward()

        torch.cuda.synchronize()
        forward_time = time.time() - start_time

        # Get memory
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        # Cleanup
        del x_local, model, output
        if test_backward:
            del loss

        gc.collect()
        torch.cuda.empty_cache()

        return True, mem_mb, forward_time

    except Exception as e:
        if rank == 0:
            print(f"Error: {e}")
            traceback.print_exc()
        return False, 0, 0


def main():
    """Main test function."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print(f"Testing Ring Dilated Attention V3 with {world_size} GPUs")
        print("=" * 80)

    # Test configurations
    test_configs = [
        # (seq_len, batch_size, label, test_backward)
        (8192, 1, "8K", False),
        (16384, 1, "16K", False),
        (32768, 1, "32K", False),
        (65536, 1, "64K", False),
        (100000, 1, "100K", False),
        (150000, 1, "150K", False),
        (200000, 1, "200K", False),
        # Test with gradients
        (8192, 1, "8K+grad", True),
        (16384, 1, "16K+grad", True),
    ]

    max_successful = 0

    for seq_len, batch_size, label, test_backward in test_configs:
        # Make divisible by world_size
        seq_len = (seq_len // world_size) * world_size

        if dist.is_initialized():
            dist.barrier()

        success, memory, elapsed = test_sequence(
            seq_len, batch_size, rank, world_size, local_rank, test_backward
        )

        if rank == 0:
            if success:
                throughput = seq_len / elapsed / 1e6
                print(
                    f"✓ {label} ({seq_len:,} tokens): "
                    f"{memory:.1f} MB, {elapsed:.2f}s, "
                    f"{throughput:.2f}M tok/s"
                )
                if not test_backward:
                    max_successful = max(max_successful, seq_len)
            else:
                print(f"✗ {label} ({seq_len:,} tokens): Failed")
                if not test_backward:
                    break

        # Small delay
        time.sleep(0.5)

    # Test with Hilbert ordering
    if rank == 0:
        print("\nTesting with Hilbert ordering:")

    for seq_len in [8192, 16384]:
        seq_len = (seq_len // world_size) * world_size

        if dist.is_initialized():
            dist.barrier()

        # Create model with Hilbert
        try:
            device = torch.device(f"cuda:{local_rank}")
            local_seq_len = seq_len // world_size

            x_local = torch.randn(
                1, local_seq_len, 768, device=device, dtype=torch.float32
            )

            model = RingDilatedAttentionV3(
                embed_dim=768,
                num_heads=12,
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                dropout=0.0,
                use_hilbert=True,  # Enable Hilbert
                device=device,
                dtype=torch.float32,
            )
            model.eval()

            start_time = time.time()
            output = model(x_local, total_seq_len=seq_len, already_split=True)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            if rank == 0:
                print(f"✓ {seq_len:,} tokens with Hilbert: {elapsed:.2f}s")

            del x_local, model, output

        except Exception as e:
            if rank == 0:
                print(f"✗ {seq_len:,} tokens with Hilbert: {e}")

    if rank == 0:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Maximum successful sequence: {max_successful:,} tokens")
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
