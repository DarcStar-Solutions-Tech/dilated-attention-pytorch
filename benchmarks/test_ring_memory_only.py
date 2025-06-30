"""
Simple test showing Ring Attention memory distribution principle.
Just shows memory usage, no communication.
"""

import torch
import torch.multiprocessing as mp
import os
import subprocess
import time


def get_all_gpu_memory():
    """Get memory for all GPUs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )

    gpus = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split(", ")
        idx = int(parts[0])
        used = int(parts[1])
        free = int(parts[2])
        gpus[idx] = {"used": used, "free": free}

    return gpus


def worker(rank, world_size, seq_len):
    """Worker process for each GPU."""

    # Set this process to use only its GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda:0")

    # Configuration
    batch_size = 1
    num_heads = 8
    head_dim = 64
    chunk_size = seq_len // world_size

    print(f"\n[Process {rank}] Started on GPU {rank}")
    print(f"[Process {rank}] Sequence length: {seq_len}")
    print(f"[Process {rank}] Chunk size: {chunk_size}")

    # Initial state
    torch.cuda.synchronize()
    time.sleep(0.5)  # Let nvidia-smi update

    # Standard approach: full KV
    print(f"\n[Process {rank}] Creating FULL tensors (standard approach)...")
    q_full = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k_full = torch.randn_like(q_full)
    v_full = torch.randn_like(q_full)

    torch.cuda.synchronize()
    time.sleep(0.5)

    # Clean up K,V but keep Q
    del k_full, v_full
    torch.cuda.empty_cache()

    # Ring approach: chunked KV
    print(f"\n[Process {rank}] Creating CHUNKED KV (ring approach)...")
    k_chunk = torch.randn(
        batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    _ = torch.randn_like(k_chunk)

    torch.cuda.synchronize()
    time.sleep(0.5)

    print(f"[Process {rank}] Done")


def main():
    """Run the memory distribution test."""
    print("Ring Attention - Memory Distribution Test")
    print("=" * 70)

    world_size = 2
    seq_len = 16384  # 16K tokens

    print("Configuration:")
    print(f"  - GPUs: {world_size}")
    print(f"  - Sequence length: {seq_len:,}")
    print(f"  - Tokens per GPU: {seq_len // world_size:,}")
    print("=" * 70)

    # Get initial memory state
    print("\nInitial GPU memory:")
    initial = get_all_gpu_memory()
    for gpu_id in range(world_size):
        print(
            f"  GPU {gpu_id}: {initial[gpu_id]['used']}MB used, "
            f"{initial[gpu_id]['free']}MB free"
        )

    # Run processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, seq_len))
        p.start()
        processes.append(p)

    # Monitor memory while processes run
    print("\nMonitoring memory usage...")

    # Wait for processes to create standard tensors
    time.sleep(3)
    print("\n--- After creating FULL tensors (standard approach) ---")
    standard = get_all_gpu_memory()
    for gpu_id in range(world_size):
        used = standard[gpu_id]["used"] - initial[gpu_id]["used"]
        print(f"  GPU {gpu_id}: +{used}MB")

    # Wait for processes to switch to ring approach
    time.sleep(3)
    print("\n--- After creating CHUNKED tensors (ring approach) ---")
    ring = get_all_gpu_memory()
    for gpu_id in range(world_size):
        used = ring[gpu_id]["used"] - initial[gpu_id]["used"]
        print(f"  GPU {gpu_id}: +{used}MB")

    # Wait for processes to finish
    for p in processes:
        p.join()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Calculate totals
    standard_total = sum(
        standard[i]["used"] - initial[i]["used"] for i in range(world_size)
    )
    ring_total = sum(ring[i]["used"] - initial[i]["used"] for i in range(world_size))

    print("\nStandard approach (full KV on each GPU):")
    print(f"  Total memory used: {standard_total}MB")
    print(f"  Memory per GPU: ~{standard_total / world_size:.0f}MB")

    print("\nRing approach (chunked KV):")
    print(f"  Total memory used: {ring_total}MB")
    print(f"  Memory per GPU: ~{ring_total / world_size:.0f}MB")

    print("\nMemory savings:")
    print(f"  Absolute: {standard_total - ring_total}MB")
    print(f"  Relative: {(1 - ring_total / standard_total) * 100:.0f}%")

    print("\nKey insight:")
    print(f"  Each GPU stores only 1/{world_size} of KV cache")
    print(f"  This enables {world_size}x longer sequences!")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
