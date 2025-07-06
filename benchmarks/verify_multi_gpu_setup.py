#!/usr/bin/env python3
"""
Quick verification that multi-GPU setup is working correctly.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def verify_gpu(rank, world_size):
    """Verify GPU setup for a single process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create a tensor on this GPU
    tensor = torch.ones(2, 2, device=device) * (rank + 1)

    print(f"[GPU {rank}] Device: {torch.cuda.get_device_name(rank)}")
    print(
        f"[GPU {rank}] Memory: {torch.cuda.get_device_properties(rank).total_memory / 1024**3:.1f} GB"
    )
    print(f"[GPU {rank}] Tensor: {tensor}")

    # Test all_reduce
    dist.all_reduce(tensor)
    print(f"[GPU {rank}] After all_reduce: {tensor}")

    # Test ring communication
    send_tensor = torch.ones(1000, 1000, device=device) * (rank + 1)
    recv_tensor = torch.zeros_like(send_tensor)

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Send to next, receive from previous
    if rank == 0:
        dist.send(send_tensor, dst=next_rank)
        dist.recv(recv_tensor, src=prev_rank)
    else:
        dist.recv(recv_tensor, src=prev_rank)
        dist.send(send_tensor, dst=next_rank)

    print(
        f"[GPU {rank}] Ring test: received from GPU {prev_rank}, sent to GPU {next_rank}"
    )

    # Cleanup
    dist.destroy_process_group()


def main():
    """Verify multi-GPU setup."""

    print("=" * 60)
    print("MULTI-GPU SETUP VERIFICATION")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s):")

    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            f"         Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
        )

    if num_gpus < 2:
        print(
            "\nWarning: Less than 2 GPUs available. Multi-GPU benchmarks will not work."
        )
        return

    print(f"\nTesting distributed communication with {num_gpus} GPUs...")
    print("-" * 60)

    # Test with all available GPUs
    try:
        mp.spawn(verify_gpu, args=(num_gpus,), nprocs=num_gpus, join=True)
        print("-" * 60)
        print("✓ Multi-GPU setup verified successfully!")
        print(f"✓ All {num_gpus} GPUs can communicate")
        print("✓ Ready for distributed benchmarks")
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        print("✗ Multi-GPU setup failed")

    print("=" * 60)


if __name__ == "__main__":
    # Set environment to see all GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(torch.cuda.device_count())
    )

    main()
