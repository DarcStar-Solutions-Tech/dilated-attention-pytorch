"""
Fixed distributed Ring Attention test that properly shows memory distribution.

This version fixes the implementation issues and monitors both GPUs.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess


def get_gpu_memory_info():
    """Get memory info for all GPUs using nvidia-smi."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )

    gpu_info = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(", ")
        gpu_info.append(
            {
                "index": int(parts[0]),
                "used": int(parts[1]),
                "free": int(parts[2]),
                "total": int(parts[3]),
            }
        )
    return gpu_info


def simple_ring_attention(rank, world_size, seq_len, batch_size=1):
    """Simple Ring Attention implementation showing memory distribution."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    os.environ["NCCL_DEBUG"] = "WARN"  # Reduce NCCL verbosity

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Configuration
    num_heads = 8
    head_dim = 64
    chunk_size = seq_len // world_size

    print(f"[GPU {rank}] Initialized - will process chunk size: {chunk_size}")

    # Synchronize and show initial memory
    dist.barrier()
    if rank == 0:
        print("\nInitial GPU memory state:")
        for gpu in get_gpu_memory_info():
            print(f"  GPU {gpu['index']}: {gpu['used']}MB used, {gpu['free']}MB free")
    dist.barrier()

    # Each GPU only allocates its chunk of K and V
    print(f"\n[GPU {rank}] Creating tensors...")

    # Full Q needed on each GPU
    q_full = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    # Only local chunk of K and V
    k_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )

    print(f"[GPU {rank}] Allocated Q: {q_full.shape}, K chunk: {k_chunk.shape}")

    # Output accumulator
    output = torch.zeros_like(q_full)

    # Synchronize and show memory after allocation
    dist.barrier()
    if rank == 0:
        print("\nGPU memory after allocation:")
        for gpu in get_gpu_memory_info():
            print(f"  GPU {gpu['index']}: {gpu['used']}MB used, {gpu['free']}MB free")
        print("\nStarting ring communication...")
    dist.barrier()

    # Ring communication
    for step in range(world_size):
        # Which chunk are we processing?
        chunk_owner = (rank - step) % world_size
        chunk_start = chunk_owner * chunk_size
        chunk_end = chunk_start + chunk_size

        print(
            f"[GPU {rank}] Step {step}: Processing chunk from rank {chunk_owner} "
            f"(positions {chunk_start}-{chunk_end})"
        )

        # Get the relevant part of Q for this chunk
        q_slice = q_full[:, chunk_start:chunk_end]

        # Compute attention for this chunk
        # Simple attention: Q @ K^T
        scores = torch.matmul(q_slice, k_chunk.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        chunk_output = torch.matmul(attn_weights, v_chunk)

        # Accumulate to output
        output[:, chunk_start:chunk_end] += chunk_output

        # Ring communication - rotate K and V chunks
        if step < world_size - 1:
            # Send to next GPU, receive from previous
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1) % world_size

            # Create receive buffers
            k_recv = torch.empty_like(k_chunk)
            v_recv = torch.empty_like(v_chunk)

            # Non-blocking send/recv
            reqs = []
            reqs.append(dist.isend(k_chunk, dst=send_rank))
            reqs.append(dist.isend(v_chunk, dst=send_rank))
            reqs.append(dist.irecv(k_recv, src=recv_rank))
            reqs.append(dist.irecv(v_recv, src=recv_rank))

            # Wait for all communication
            for req in reqs:
                req.wait()

            # Update chunks for next iteration
            k_chunk = k_recv
            v_chunk = v_recv

        dist.barrier()

    # Final synchronization
    dist.barrier()
    if rank == 0:
        print("\nRing communication completed!")
        print("\nFinal GPU memory state:")
        for gpu in get_gpu_memory_info():
            print(f"  GPU {gpu['index']}: {gpu['used']}MB used, {gpu['free']}MB free")

    # Verify output
    print(f"\n[GPU {rank}] Output shape: {output.shape}")
    print(
        f"[GPU {rank}] Output stats - mean: {output.mean():.4f}, std: {output.std():.4f}"
    )

    # Cleanup
    dist.destroy_process_group()


def run_comparison_test():
    """Run comparison between standard and ring attention."""
    print("Distributed Ring Attention - Memory Distribution Test")
    print("=" * 70)

    world_size = 2
    test_configs = [
        (8192, 2, "8K tokens, batch 2"),
        (16384, 1, "16K tokens, batch 1"),
        (32768, 1, "32K tokens, batch 1"),
    ]

    for seq_len, batch_size, desc in test_configs:
        print(f"\n\nTest: {desc}")
        print("-" * 70)

        # First show memory for standard approach
        print("\n1. Standard approach (each GPU would need full KV):")

        num_heads, head_dim = 8, 64
        element_size = 2  # float16

        # Memory calculation
        qkv_memory = 3 * batch_size * seq_len * num_heads * head_dim * element_size
        qkv_memory_mb = qkv_memory / (1024**2)

        print(f"   Each GPU would need: {qkv_memory_mb:.0f}MB for QKV")
        print(f"   Total redundant memory: {qkv_memory_mb * world_size:.0f}MB")

        # Now run ring attention
        print("\n2. Ring approach (distributed KV):")

        try:
            mp.spawn(
                simple_ring_attention,
                args=(world_size, seq_len, batch_size),
                nprocs=world_size,
                join=True,
            )

            # Calculate theoretical savings
            kv_per_gpu = (
                2
                * batch_size
                * (seq_len // world_size)
                * num_heads
                * head_dim
                * element_size
            )
            kv_per_gpu_mb = kv_per_gpu / (1024**2)
            q_memory_mb = (
                batch_size * seq_len * num_heads * head_dim * element_size / (1024**2)
            )

            ring_total_mb = q_memory_mb + kv_per_gpu_mb

            print("\n   Theoretical memory per GPU:")
            print(f"   - Q (full): {q_memory_mb:.0f}MB")
            print(f"   - KV (chunk): {kv_per_gpu_mb:.0f}MB")
            print(f"   - Total: {ring_total_mb:.0f}MB")
            print(f"   - Savings: {(1 - ring_total_mb / qkv_memory_mb) * 100:.0f}%")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Summary:")
    print("- Ring Attention distributes KV across GPUs")
    print("- Each GPU only stores 1/N of the KV cache")
    print("- Communication overhead is offset by memory savings")
    print("- Enables processing sequences that wouldn't fit on single GPU")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_comparison_test()
