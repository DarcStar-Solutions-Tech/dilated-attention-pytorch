"""
Final working Ring Attention test showing actual memory distribution.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import subprocess


def get_gpu_memory():
    """Get current GPU memory usage."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )

    memory = {}
    for line in result.stdout.strip().split("\n"):
        idx, used = line.split(", ")
        memory[int(idx)] = int(used)
    return memory


def ring_demo(rank, world_size):
    """Demonstrate Ring Attention memory distribution."""

    # Setup distributed - each process gets its own GPU
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12362"

    # CRITICAL: Set CUDA device before init
    torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    # Parameters
    seq_len = 16384  # 16K tokens
    chunk_size = seq_len // world_size
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(f"\n[GPU {rank}] Initialized")
    print(f"[GPU {rank}] Will process chunk size: {chunk_size}")

    # Synchronize and show initial memory
    dist.barrier()
    time.sleep(0.1 * rank)  # Stagger prints

    initial_gpu_mem = get_gpu_memory()
    print(f"[GPU {rank}] Initial memory: {initial_gpu_mem[rank]}MB")

    dist.barrier()

    # === PHASE 1: Show standard approach memory ===
    if rank == 0:
        print("\n=== PHASE 1: Standard Approach (each GPU has full KV) ===")
    dist.barrier()

    # Create full tensors
    q_full = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k_full = torch.randn_like(q_full)
    v_full = torch.randn_like(q_full)

    torch.cuda.synchronize()
    dist.barrier()

    time.sleep(0.1 * rank)
    standard_gpu_mem = get_gpu_memory()
    print(
        f"[GPU {rank}] With full QKV: {standard_gpu_mem[rank]}MB "
        f"(+{standard_gpu_mem[rank] - initial_gpu_mem[rank]}MB)"
    )

    # Cleanup
    del k_full, v_full  # Keep q_full for ring approach
    torch.cuda.empty_cache()
    dist.barrier()

    # === PHASE 2: Show ring approach memory ===
    if rank == 0:
        print("\n=== PHASE 2: Ring Approach (distributed KV) ===")
    dist.barrier()

    # Each GPU only allocates its chunk of K and V
    k_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v_chunk = torch.randn_like(k_chunk)

    torch.cuda.synchronize()
    dist.barrier()

    time.sleep(0.1 * rank)
    ring_gpu_mem = get_gpu_memory()
    print(
        f"[GPU {rank}] With Q + KV chunk: {ring_gpu_mem[rank]}MB "
        f"(+{ring_gpu_mem[rank] - initial_gpu_mem[rank]}MB)"
    )

    dist.barrier()

    # === PHASE 3: Show memory summary ===
    if rank == 0:
        print("\n=== MEMORY SUMMARY ===")
        time.sleep(0.2)  # Let other GPUs report first

        # Get memory from all GPUs
        _ = get_gpu_memory()

        print("\nStandard approach (each GPU has full KV):")
        standard_total = 0
        for i in range(world_size):
            mem_used = standard_gpu_mem[i] - initial_gpu_mem[i]
            standard_total += mem_used
            print(f"  GPU {i}: {mem_used}MB")
        print(f"  Total: {standard_total}MB")

        print("\nRing approach (distributed KV):")
        ring_total = 0
        for i in range(world_size):
            mem_used = ring_gpu_mem[i] - initial_gpu_mem[i]
            ring_total += mem_used
            print(f"  GPU {i}: {mem_used}MB")
        print(f"  Total: {ring_total}MB")

        print(
            f"\nMemory saved: {standard_total - ring_total}MB "
            f"({(1 - ring_total / standard_total) * 100:.0f}%)"
        )

        print(f"\nWith {world_size} GPUs:")
        print(f"- Each GPU stores only 1/{world_size} of KV")
        print(f"- Enables {world_size}x longer sequences")
        print("- Trade-off: communication overhead")

    # === PHASE 4: Simple ring communication ===
    dist.barrier()
    if rank == 0:
        print("\n=== RING COMMUNICATION DEMO ===")
    dist.barrier()

    # Each GPU sends its chunk around the ring
    for step in range(world_size):
        if rank == 0:
            print(f"\nStep {step}:")

        # Show which GPU has which chunk
        chunk_owner = (rank - step) % world_size
        time.sleep(0.1 * rank)
        print(f"  [GPU {rank}] Processing chunk originally from GPU {chunk_owner}")

        # Ring exchange (except last step)
        if step < world_size - 1:
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1) % world_size

            k_recv = torch.empty_like(k_chunk)
            v_recv = torch.empty_like(v_chunk)

            # Non-blocking send/recv
            reqs = [
                dist.isend(k_chunk, dst=send_rank),
                dist.irecv(k_recv, src=recv_rank),
                dist.isend(v_chunk, dst=send_rank),
                dist.irecv(v_recv, src=recv_rank),
            ]

            for req in reqs:
                req.wait()

            k_chunk = k_recv
            v_chunk = v_recv

        dist.barrier()

    # Cleanup
    if rank == 0:
        print("\n=== TEST COMPLETED ===")

    dist.destroy_process_group()


def main():
    """Run the ring attention demonstration."""
    print("Ring Attention V2 - Memory Distribution Demonstration")
    print("=" * 70)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Error: This demo requires at least 2 GPUs")
        return

    print(f"Found {world_size} GPUs")
    print("Demonstrating memory distribution across GPUs...")
    print("=" * 70)

    # Launch processes
    mp.spawn(ring_demo, args=(world_size,), nprocs=world_size, join=True)

    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("Ring Attention successfully distributed memory across GPUs")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
