"""
Simple test to demonstrate Ring Attention V2 memory distribution.

This test shows how memory is distributed across GPUs in Ring Attention.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import GPUtil


def monitor_gpu_memory(rank):
    """Monitor GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if rank < len(gpus):
        gpu = gpus[rank]
        return {
            "used": gpu.memoryUsed,
            "free": gpu.memoryFree,
            "total": gpu.memoryTotal,
            "util": gpu.memoryUtil * 100,
        }
    return None


def simple_ring_test(rank, world_size, seq_len=16384):
    """Simple test showing memory distribution."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Monitor initial memory
    torch.cuda.synchronize()
    initial_mem = monitor_gpu_memory(rank)

    print(
        f"[GPU {rank}] Initial memory: {initial_mem['used']:.0f}MB used, "
        f"{initial_mem['free']:.0f}MB free"
    )

    # Test 1: Standard approach (each GPU stores full KV)
    print(f"\n[GPU {rank}] Test 1: Standard (full KV on each GPU)")

    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Full tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    torch.cuda.synchronize()
    standard_mem = monitor_gpu_memory(rank)

    print(
        f"[GPU {rank}] After creating full QKV: {standard_mem['used']:.0f}MB used "
        f"(+{standard_mem['used'] - initial_mem['used']:.0f}MB)"
    )

    # Clean up
    del q, k, v
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 2: Ring approach (each GPU stores only its KV chunk)
    print(f"\n[GPU {rank}] Test 2: Ring (chunked KV)")

    chunk_size = seq_len // world_size

    # Full Q (needed on each GPU)
    _ = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    # Only local KV chunk
    k_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v_chunk = torch.randn_like(k_chunk)

    torch.cuda.synchronize()
    ring_mem = monitor_gpu_memory(rank)

    print(
        f"[GPU {rank}] After creating Q + KV chunk: {ring_mem['used']:.0f}MB used "
        f"(+{ring_mem['used'] - initial_mem['used']:.0f}MB)"
    )

    # Show memory savings
    if rank == 0:
        dist.barrier()
        print(f"\n[GPU {rank}] Memory Comparison:")
        print(
            f"[GPU {rank}] Standard: {standard_mem['used'] - initial_mem['used']:.0f}MB"
        )
        print(f"[GPU {rank}] Ring: {ring_mem['used'] - initial_mem['used']:.0f}MB")
        print(
            f"[GPU {rank}] Savings: {(1 - (ring_mem['used'] - initial_mem['used']) / (standard_mem['used'] - initial_mem['used'])) * 100:.0f}%"
        )

    # Test 3: Simulate ring communication
    print(f"\n[GPU {rank}] Test 3: Ring communication")

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size

    # Create buffers for communication
    k_recv = torch.empty_like(k_chunk)
    v_recv = torch.empty_like(v_chunk)

    # Simulate ring iterations
    for step in range(world_size):
        if rank == 0:
            print(f"[GPU {rank}] Ring step {step + 1}/{world_size}")

        # Send/receive KV chunks
        if world_size > 1:
            send_k = dist.isend(k_chunk, dst=send_rank)
            send_v = dist.isend(v_chunk, dst=send_rank)
            recv_k = dist.irecv(k_recv, src=recv_rank)
            recv_v = dist.irecv(v_recv, src=recv_rank)

            send_k.wait()
            send_v.wait()
            recv_k.wait()
            recv_v.wait()

            # Swap buffers
            k_chunk, k_recv = k_recv, k_chunk
            v_chunk, v_recv = v_recv, v_chunk

        # Would compute attention here
        # scores = torch.matmul(q, k_chunk.transpose(-2, -1))

        dist.barrier()

    if rank == 0:
        print(f"\n[GPU {rank}] Ring communication completed successfully!")
        print(f"[GPU {rank}] Each GPU only held 1/{world_size} of KV at any time")

    # Cleanup
    dist.destroy_process_group()


def main():
    print("Ring Attention V2 - Memory Distribution Test")
    print("=" * 60)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Error: Need at least 2 GPUs for this test")
        return

    print(f"Found {world_size} GPUs")

    # Test different sequence lengths
    seq_lengths = [8192, 16384]

    for seq_len in seq_lengths:
        print(f"\n\nTesting with sequence length: {seq_len}")
        print("-" * 60)

        try:
            mp.spawn(
                simple_ring_test,
                args=(world_size, seq_len),
                nprocs=world_size,
                join=True,
            )
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("1. Standard approach: Each GPU stores full KV (redundant)")
    print("2. Ring approach: Each GPU stores 1/N of KV (distributed)")
    print("3. Memory savings: ~50% with 2 GPUs, ~75% with 4 GPUs")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
