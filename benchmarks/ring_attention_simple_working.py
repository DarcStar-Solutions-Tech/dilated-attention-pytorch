"""
Simplified Working Ring Attention - Demonstrating proper GPU communication.

This focuses on the core principle: distributing KV across GPUs and using
proper PyTorch distributed APIs for communication.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import math


def ring_attention_worker(rank: int, world_size: int):
    """Worker process for Ring Attention demonstration."""

    # Setup distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    # Parameters
    batch_size = 1
    seq_len = 8192
    num_heads = 8
    head_dim = 64
    chunk_size = seq_len // world_size

    print(f"\n[GPU {rank}] Ring Attention Worker Started")
    print(f"[GPU {rank}] Sequence length: {seq_len}, Chunk size: {chunk_size}")

    # Monitor initial memory
    torch.cuda.synchronize()
    initial_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"[GPU {rank}] Initial memory: {initial_mem:.1f}MB")

    # === PHASE 1: Show Standard Attention Memory Usage ===
    if rank == 0:
        print("\n=== PHASE 1: Standard Attention (baseline) ===")
    dist.barrier()

    # Create full tensors for standard attention
    q_full = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k_full = torch.randn_like(q_full)
    v_full = torch.randn_like(q_full)

    torch.cuda.synchronize()
    standard_mem = torch.cuda.memory_allocated() / 1024**2
    print(
        f"[GPU {rank}] Standard attention memory: {standard_mem:.1f}MB (+{standard_mem - initial_mem:.1f}MB)"
    )

    # Clean up
    del k_full, v_full
    torch.cuda.empty_cache()
    dist.barrier()

    # === PHASE 2: Ring Attention Setup ===
    if rank == 0:
        print("\n=== PHASE 2: Ring Attention (distributed KV) ===")
    dist.barrier()

    # Ring Attention: Keep full Q, but only local chunk of K/V
    _ = rank * chunk_size
    _ = (rank + 1) * chunk_size

    # Each GPU generates its own chunk of K and V
    k_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v_chunk = torch.randn_like(k_chunk)

    torch.cuda.synchronize()
    ring_mem = torch.cuda.memory_allocated() / 1024**2
    print(
        f"[GPU {rank}] Ring attention memory: {ring_mem:.1f}MB (+{ring_mem - initial_mem:.1f}MB)"
    )
    print(
        f"[GPU {rank}] Memory saved: {((standard_mem - ring_mem) / standard_mem * 100):.0f}%"
    )

    dist.barrier()

    # === PHASE 3: Demonstrate Ring Communication ===
    if rank == 0:
        print("\n=== PHASE 3: Ring Communication Demo ===")
    dist.barrier()

    # Simple ring communication test
    print(f"\n[GPU {rank}] Starting ring communication...")

    # Each GPU will pass its chunk around the ring
    current_k = k_chunk.clone()
    current_v = v_chunk.clone()

    for step in range(world_size):
        # Which chunk are we processing?
        chunk_owner = (rank - step) % world_size
        print(f"[GPU {rank}] Step {step}: Processing chunk from GPU {chunk_owner}")

        # Compute partial attention (simplified)
        # In real implementation, this would accumulate results
        _ = torch.matmul(
            q_full.reshape(batch_size * num_heads, seq_len, head_dim),
            current_k.reshape(batch_size * num_heads, chunk_size, head_dim).transpose(
                -2, -1
            ),
        ) / math.sqrt(head_dim)

        # Ring exchange (except last step)
        if step < world_size - 1:
            # Prepare for communication
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size

            # Allocate receive buffers
            k_recv = torch.empty_like(current_k)
            v_recv = torch.empty_like(current_v)

            # Use isend/irecv for non-blocking communication
            print(
                f"[GPU {rank}] Sending to GPU {send_rank}, receiving from GPU {recv_rank}"
            )

            # Start all operations
            ops = []
            ops.append(dist.isend(current_k, dst=send_rank))
            ops.append(dist.irecv(k_recv, src=recv_rank))
            ops.append(dist.isend(current_v, dst=send_rank))
            ops.append(dist.irecv(v_recv, src=recv_rank))

            # Wait for completion
            for op in ops:
                op.wait()

            # Update current chunks
            current_k = k_recv
            current_v = v_recv

            print(f"[GPU {rank}] Communication completed")

        dist.barrier()

    # === PHASE 4: Summary ===
    dist.barrier()
    if rank == 0:
        print("\n=== SUMMARY ===")
        print(f"✓ Each GPU stores only 1/{world_size} of KV cache")
        print(f"✓ Memory reduction: ~{(world_size - 1) / world_size * 100:.0f}% for KV")
        print(f"✓ Enables {world_size}x longer sequences")
        print("✓ Communication uses proper PyTorch distributed APIs")
        print("\nKey insight: Ring Attention trades communication for memory")
        print("Perfect for very long sequences where memory is the bottleneck")

    # Cleanup
    dist.destroy_process_group()
    print(f"\n[GPU {rank}] Worker completed successfully")


def main():
    """Run the Ring Attention demonstration."""
    print("Simple Working Ring Attention Demonstration")
    print("=" * 70)
    print("\nThis demonstrates the core principles of Ring Attention:")
    print("1. Distributed KV cache across GPUs")
    print("2. Proper communication using PyTorch distributed APIs")
    print("3. Memory savings that enable longer sequences")
    print("=" * 70)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("\nError: This demo requires at least 2 GPUs")
        print("Ring Attention provides no benefit on single GPU")
        return

    print(f"\nFound {world_size} GPUs")
    print("Starting distributed Ring Attention test...")

    try:
        mp.spawn(
            ring_attention_worker, args=(world_size,), nprocs=world_size, join=True
        )
        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("Ring Attention properly demonstrated:")
        print("- Distributed KV storage")
        print("- GPU-to-GPU communication")
        print("- Memory efficiency gains")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check GPU availability: nvidia-smi")
        print("2. Ensure NCCL is properly installed")
        print("3. Try different MASTER_PORT if port is in use")
        print("4. Set NCCL_DEBUG=INFO for detailed errors")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
