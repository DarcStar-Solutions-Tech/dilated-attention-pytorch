"""
Basic test to demonstrate Ring Attention memory distribution principle.

This shows the core idea: each GPU holds only part of KV, not all of it.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def ring_worker(rank, world_size):
    """Simple demonstration of ring attention memory usage."""

    # Initialize distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"  # Each process sees only its GPU

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:0")  # Will map to the correct GPU

    # Parameters
    seq_len = 8192
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(f"\n[Rank {rank}] Starting on GPU {rank}")

    # Show initial memory
    torch.cuda.synchronize()
    initial_mem = torch.cuda.memory_allocated(device) / 1024**2
    print(f"[Rank {rank}] Initial memory: {initial_mem:.1f}MB")

    # Standard approach: each GPU has full KV
    print(f"\n[Rank {rank}] === STANDARD APPROACH ===")
    q_full = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k_full = torch.randn_like(q_full)
    v_full = torch.randn_like(q_full)

    torch.cuda.synchronize()
    standard_mem = torch.cuda.memory_allocated(device) / 1024**2
    print(
        f"[Rank {rank}] After creating full QKV: {standard_mem:.1f}MB (used {standard_mem - initial_mem:.1f}MB)"
    )

    # Clean up
    del q_full, k_full, v_full
    torch.cuda.empty_cache()
    dist.barrier()

    # Ring approach: each GPU has only its chunk of KV
    print(f"\n[Rank {rank}] === RING APPROACH ===")
    chunk_size = seq_len // world_size

    # Full Q (needed on each GPU for computation)
    _ = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    # Only this GPU's chunk of K and V
    k_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    _ = torch.randn_like(k_chunk)

    torch.cuda.synchronize()
    ring_mem = torch.cuda.memory_allocated(device) / 1024**2
    print(
        f"[Rank {rank}] After creating Q + KV chunk: {ring_mem:.1f}MB (used {ring_mem - initial_mem:.1f}MB)"
    )

    # Show the memory distribution
    dist.barrier()
    if rank == 0:
        print(f"\n[Rank {rank}] === MEMORY SUMMARY ===")
        print(f"[Rank {rank}] Sequence length: {seq_len}")
        print(f"[Rank {rank}] Chunk size per GPU: {chunk_size}")
        print(
            f"[Rank {rank}] Standard approach: {standard_mem - initial_mem:.1f}MB per GPU"
        )
        print(f"[Rank {rank}] Ring approach: {ring_mem - initial_mem:.1f}MB per GPU")
        print(
            f"[Rank {rank}] Memory saved: {(1 - (ring_mem - initial_mem) / (standard_mem - initial_mem)) * 100:.0f}%"
        )

        # Calculate total memory across all GPUs
        total_standard = (standard_mem - initial_mem) * world_size
        total_ring = (ring_mem - initial_mem) * world_size
        print(f"\n[Rank {rank}] Total memory (all GPUs):")
        print(f"[Rank {rank}] Standard: {total_standard:.1f}MB (redundant KV)")
        print(f"[Rank {rank}] Ring: {total_ring:.1f}MB (distributed KV)")

    # Simple ring communication demonstration
    dist.barrier()
    if rank == 0:
        print(f"\n[Rank {rank}] === RING COMMUNICATION ===")

    # Each GPU sends its chunk to the next GPU in the ring
    for step in range(world_size):
        current_owner = (rank - step) % world_size
        if rank == 0:
            print(
                f"[Rank {rank}] Step {step}: Processing chunk from GPU {current_owner}"
            )

        # In real implementation, GPUs would exchange chunks here
        # For demo, just show that each GPU processes different chunks

    dist.barrier()

    # Cleanup
    dist.destroy_process_group()
    print(f"\n[Rank {rank}] Done!")


def main():
    print("Ring Attention - Basic Memory Distribution Demo")
    print("=" * 60)

    world_size = 2  # Use 2 GPUs

    print(f"Running on {world_size} GPUs")
    print("Each GPU will show its memory usage")
    print("=" * 60)

    # Launch processes
    mp.spawn(ring_worker, args=(world_size,), nprocs=world_size, join=True)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Standard: Each GPU stores full KV (redundant)")
    print("2. Ring: Each GPU stores 1/N of KV (distributed)")
    print("3. Memory savings increase with more GPUs")
    print("4. Enables longer sequences by distributing memory load")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
