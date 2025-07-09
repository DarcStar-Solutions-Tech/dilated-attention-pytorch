#!/usr/bin/env python3
"""
Minimal test for ring attention with QKV projections.

Launch with:
torchrun --nproc_per_node=2 test_ring_attention_minimal.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import os
import gc


def ring_pass_kv(k, v, rank, world_size):
    """Simple ring pass for K,V tensors."""
    if world_size <= 1:
        return k, v

    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    # Ensure contiguous
    k = k.contiguous()
    v = v.contiguous()

    # Allocate receive buffers
    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    # Use blocking communication with proper ordering
    if rank % 2 == 0:
        # Even ranks: send K, recv K, send V, recv V
        dist.send(k, dst=dst)
        dist.recv(k_recv, src=src)
        dist.send(v, dst=dst)
        dist.recv(v_recv, src=src)
    else:
        # Odd ranks: recv K, send K, recv V, send V
        dist.recv(k_recv, src=src)
        dist.send(k, dst=dst)
        dist.recv(v_recv, src=src)
        dist.send(v, dst=dst)

    return k_recv, v_recv


def test_minimal():
    """Test minimal ring attention."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32  # Use FP32 for Pascal GPUs

    print(f"Rank {rank}: Starting minimal ring attention test")

    # Parameters
    batch_size = 1
    seq_len = 1024  # Small sequence
    embed_dim = 768
    num_heads = 12
    head_dim = embed_dim // num_heads

    # Create local input
    x_local = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Create QKV projection
    qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False).to(
        device=device, dtype=dtype
    )

    # Ensure CUDA operations complete
    torch.cuda.synchronize()

    print(f"Rank {rank}: Created tensors and model")

    # QKV projection
    qkv = qkv_proj(x_local)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

    q, k, v = qkv[0], qkv[1], qkv[2]

    print(f"Rank {rank}: QKV shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    if world_size > 1:
        # Synchronize before ring communication
        dist.barrier()
        print(f"Rank {rank}: Starting ring communication")

        # Test ring pass
        k_new, v_new = ring_pass_kv(k, v, rank, world_size)

        print(f"Rank {rank}: Ring pass complete")

        # Final synchronization
        dist.barrier()
        print(f"Rank {rank}: Final barrier passed")

        # Cleanup
        dist.destroy_process_group()

    # Clear memory
    del x_local, qkv_proj, qkv, q, k, v
    if world_size > 1:
        del k_new, v_new
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Rank {rank}: Test completed successfully!")


if __name__ == "__main__":
    test_minimal()
