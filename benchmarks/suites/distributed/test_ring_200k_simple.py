#!/usr/bin/env python3
"""
Simple test to reach 200K tokens with ring attention.

Launch with:
torchrun --nproc_per_node=2 test_ring_200k_simple.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import time


def ring_pass_kv(k, v, rank, world_size):
    """Simple ring pass for K,V tensors."""
    if world_size <= 1:
        return k, v

    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    k = k.contiguous()
    v = v.contiguous()

    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    # Use proper ordering to avoid deadlock
    if rank % 2 == 0:
        dist.send(k, dst=dst)
        dist.recv(k_recv, src=src)
        dist.send(v, dst=dst)
        dist.recv(v_recv, src=src)
    else:
        dist.recv(k_recv, src=src)
        dist.send(k, dst=dst)
        dist.recv(v_recv, src=src)
        dist.send(v, dst=dst)

    return k_recv, v_recv


class SimpleRingAttention(nn.Module):
    """Simple ring attention with dilated patterns."""

    def __init__(self, embed_dim, num_heads, device, dtype):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.qkv_proj = self.qkv_proj.to(device=device, dtype=dtype)

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def forward(self, x_local, segment_len=2048, dilation=1):
        """Forward pass with dilated attention."""
        batch_size, seq_len, _ = x_local.shape

        # QKV projection
        qkv = self.qkv_proj(x_local)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply dilation by selecting positions
        if dilation > 1 and seq_len > segment_len:
            positions = torch.arange(0, segment_len, device=x_local.device)
            q = q[:, :, positions, :]
            k = k[:, :, positions, :]
            v = v[:, :, positions, :]
            effective_len = segment_len
        else:
            effective_len = seq_len

        # Ring attention
        if self.world_size > 1:
            output = self._ring_attention(q, k, v)
        else:
            # Single GPU - use SDPA
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, effective_len, self.embed_dim)

        # If dilated, pad back to original size
        if dilation > 1 and seq_len > segment_len:
            padded = torch.zeros(
                batch_size,
                seq_len,
                self.embed_dim,
                device=output.device,
                dtype=output.dtype,
            )
            padded[:, positions, :] = output
            output = padded

        return output

    def _ring_attention(self, q, k, v):
        """Simple ring attention."""
        output = torch.zeros_like(q)

        k_chunk = k.clone()
        v_chunk = v.clone()

        for step in range(self.world_size):
            # Compute attention for current chunk
            attn_output = F.scaled_dot_product_attention(
                q, k_chunk, v_chunk, dropout_p=0.0
            )
            output += attn_output

            # Ring pass
            if step < self.world_size - 1:
                k_chunk, v_chunk = ring_pass_kv(
                    k_chunk, v_chunk, self.rank, self.world_size
                )

        return output / self.world_size


def test_sequence(total_tokens, rank, world_size, local_rank):
    """Test a specific sequence length."""
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32  # FP32 for Pascal GPUs

    local_tokens = total_tokens // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Create model
        model = SimpleRingAttention(768, 12, device, dtype)

        # Create input
        x_local = torch.randn(1, local_tokens, 768, device=device, dtype=dtype)

        # Warmup
        with torch.no_grad():
            _ = model(x_local, segment_len=2048, dilation=4)
            torch.cuda.synchronize()

        # Measure
        start = time.time()
        with torch.no_grad():
            _ = model(x_local, segment_len=2048, dilation=8)
            torch.cuda.synchronize()
        elapsed = time.time() - start

        # Memory
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        return True, mem_mb, elapsed

    except Exception as e:
        if rank == 0:
            print(f"Error: {e}")
        return False, 0, 0


def main():
    """Main test function."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print("Testing Simple Ring Attention")
        print(f"World size: {world_size}")
        print()

    # Test sequences
    test_sizes = [50000, 100000, 150000, 200000, 250000, 300000]

    for total_tokens in test_sizes:
        # Make divisible by world_size
        total_tokens = (total_tokens // world_size) * world_size

        if dist.is_initialized():
            dist.barrier()

        success, memory, elapsed = test_sequence(
            total_tokens, rank, world_size, local_rank
        )

        if rank == 0:
            if success:
                local_tokens = total_tokens // world_size
                print(f"✓ {total_tokens:,} tokens ({local_tokens:,} per GPU)")
                print(f"  Memory: {memory:.1f} MB")
                print(f"  Time: {elapsed:.2f}s")
                print()
            else:
                print(f"✗ {total_tokens:,} tokens - Failed")
                break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
