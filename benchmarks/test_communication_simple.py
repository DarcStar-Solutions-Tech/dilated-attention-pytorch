"""
Simple test to verify distributed communication works.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def simple_comm_test(rank: int, world_size: int):
    """Minimal communication test."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12362"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    print(f"\n[Rank {rank}] Starting...")

    try:
        # Initialize process group
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, init_method="env://"
        )

        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        print(f"[Rank {rank}] Process group initialized")

        # Test 1: Barrier
        dist.barrier()
        print(f"[Rank {rank}] ✓ Barrier successful")

        # Test 2: Small tensor exchange
        tensor = torch.tensor([rank], device=device, dtype=torch.float32)

        if rank == 0:
            dist.send(tensor, dst=1)
            print(f"[Rank {rank}] ✓ Sent tensor to rank 1")
        else:
            dist.recv(tensor, src=0)
            print(f"[Rank {rank}] ✓ Received tensor: {tensor.item()}")

        # Test 3: Cleanup
        dist.barrier()
        dist.destroy_process_group()
        print(f"[Rank {rank}] ✓ Clean shutdown")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Run simple communication test."""
    print("Simple Communication Test")
    print("=" * 50)

    world_size = 2

    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs for this test")
        return

    try:
        mp.spawn(simple_comm_test, args=(world_size,), nprocs=world_size, join=True)
        print("\n✅ Communication test PASSED")
    except Exception as e:
        print(f"\n❌ Communication test FAILED: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
