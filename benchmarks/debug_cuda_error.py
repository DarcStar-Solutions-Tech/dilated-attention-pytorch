"""
Debug CUDA error by testing each operation in isolation.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def debug_worker(rank: int, world_size: int):
    """Debug worker to isolate CUDA error."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12365"

    # Set device properly
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"\n[GPU {rank}] Debug worker started")
    print(
        f"[GPU {rank}] Device: {device}, Current device: {torch.cuda.current_device()}"
    )

    try:
        # Test 1: Local tensor operations
        print(f"\n[GPU {rank}] Test 1: Local operations")
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        c = torch.matmul(a, b)
        print(f"[GPU {rank}] ✓ Local matmul successful")

        # Test 2: Send/recv without using the result
        print(f"\n[GPU {rank}] Test 2: Send/recv without using result")
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size

        send_tensor = torch.ones(100, 100, device=device) * rank
        recv_tensor = torch.empty(100, 100, device=device)

        req_send = dist.isend(send_tensor, dst=send_rank)
        req_recv = dist.irecv(recv_tensor, src=recv_rank)
        req_send.wait()
        req_recv.wait()

        print(f"[GPU {rank}] ✓ Send/recv successful")
        print(f"[GPU {rank}] Received tensor device: {recv_tensor.device}")

        # Test 3: Use received tensor in computation
        print(f"\n[GPU {rank}] Test 3: Using received tensor")

        # First, just access the tensor
        print(f"[GPU {rank}] Recv tensor sum: {recv_tensor.sum().item()}")

        # Try simple operation
        recv_plus_one = recv_tensor + 1
        print(f"[GPU {rank}] ✓ Simple addition successful")

        # Try matmul with local tensor
        result = torch.matmul(a, recv_tensor)
        print(f"[GPU {rank}] ✓ Matmul with received tensor successful")

        # Test 4: Multiple exchanges
        print(f"\n[GPU {rank}] Test 4: Multiple exchanges")
        for i in range(3):
            # Exchange
            req_send = dist.isend(send_tensor, dst=send_rank)
            req_recv = dist.irecv(recv_tensor, src=recv_rank)
            req_send.wait()
            req_recv.wait()

            # Use immediately
            result = torch.matmul(send_tensor, recv_tensor.T)
            print(f"[GPU {rank}] ✓ Exchange {i} successful")

            # Update for next iteration
            send_tensor = recv_tensor.clone()

        print(f"\n[GPU {rank}] ✅ ALL TESTS PASSED")

    except Exception as e:
        print(f"\n[GPU {rank}] ❌ ERROR at: {e}")
        import traceback

        traceback.print_exc()

        # Additional debug info
        print(f"\n[GPU {rank}] Debug info:")
        print(f"  Current device: {torch.cuda.current_device()}")
        if "recv_tensor" in locals():
            print(f"  Recv tensor device: {recv_tensor.device}")
            print(f"  Recv tensor shape: {recv_tensor.shape}")
            print(f"  Recv tensor dtype: {recv_tensor.dtype}")

    finally:
        dist.barrier()
        dist.destroy_process_group()


def main():
    """Run CUDA error debug."""
    print("CUDA Error Debug")
    print("=" * 50)

    world_size = 2

    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs")
        return

    # Set this for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    try:
        mp.spawn(debug_worker, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"\nDebug failed: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
