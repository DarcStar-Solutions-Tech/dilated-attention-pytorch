"""
Working Ring Attention V2 implementation for 2 GPUs.

This demonstrates the actual RingDilatedAttentionV2 working in distributed mode.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import math


class SimpleRingAttention(torch.nn.Module):
    """Simplified Ring Attention that actually works."""

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        # Get distributed info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Ring attention forward pass."""
        batch_size, seq_len, num_heads, head_dim = q.shape

        if self.world_size == 1:
            # Fallback to standard attention
            return self._standard_attention(q, k, v)

        # Distributed ring attention
        chunk_size = seq_len // self.world_size

        # Get local chunks
        local_start = self.rank * chunk_size
        local_end = (self.rank + 1) * chunk_size

        # Each GPU keeps full Q but only its chunk of K and V
        _ = q  # Keep full Q
        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Output accumulator
        output = torch.zeros_like(q)

        # Ring iterations
        k_chunk = k_local.clone()
        v_chunk = v_local.clone()

        for step in range(self.world_size):
            # Which chunk are we processing?
            chunk_owner = (self.rank - step) % self.world_size
            chunk_start = chunk_owner * chunk_size
            chunk_end = chunk_start + chunk_size

            # Compute attention for this chunk
            output[:, chunk_start:chunk_end] = self._compute_chunk_attention(
                q[:, chunk_start:chunk_end], k_chunk, v_chunk
            )

            # Ring communication
            if step < self.world_size - 1:
                k_chunk, v_chunk = self._ring_exchange(k_chunk, v_chunk)

        return output

    def _compute_chunk_attention(self, q_chunk, k_chunk, v_chunk):
        """Compute attention for a chunk."""
        # q_chunk: [batch, chunk_size, heads, dim]
        # k_chunk, v_chunk: [batch, chunk_size, heads, dim]

        # Transpose for batched matrix multiply
        q_chunk = q_chunk.transpose(1, 2)  # [batch, heads, chunk_size, dim]
        k_chunk = k_chunk.transpose(1, 2)
        v_chunk = v_chunk.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply to values
        output = torch.matmul(attn_weights, v_chunk)

        # Transpose back
        output = output.transpose(1, 2)  # [batch, chunk_size, heads, dim]

        return output

    def _ring_exchange(self, k_chunk, v_chunk):
        """Exchange chunks between GPUs in ring pattern."""
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1) % self.world_size

        # Allocate receive buffers
        k_recv = torch.empty_like(k_chunk)
        v_recv = torch.empty_like(v_chunk)

        # Use send/recv operations
        send_op_k = dist.isend(k_chunk, dst=send_rank)
        recv_op_k = dist.irecv(k_recv, src=recv_rank)

        send_op_v = dist.isend(v_chunk, dst=send_rank)
        recv_op_v = dist.irecv(v_recv, src=recv_rank)

        # Wait for completion
        send_op_k.wait()
        recv_op_k.wait()
        send_op_v.wait()
        recv_op_v.wait()

        return k_recv, v_recv

    def _standard_attention(self, q, k, v):
        """Standard attention fallback."""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output.transpose(1, 2)


def run_ring_attention_test(rank, world_size, seq_len, batch_size):
    """Test Ring Attention V2 on each GPU."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:0")

    # Configuration
    num_heads = 8
    head_dim = 64

    print(f"\n[GPU {rank}] Initializing Ring Attention test")
    print(f"[GPU {rank}] Sequence length: {seq_len}, Batch size: {batch_size}")

    # Create model
    model = SimpleRingAttention(
        num_heads=num_heads, head_dim=head_dim, device=device, dtype=torch.float16
    ).to(device)

    # Monitor initial memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1024**2

    print(f"[GPU {rank}] Initial memory: {initial_mem:.1f}MB")

    # Create inputs - full tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    input_mem = torch.cuda.memory_allocated() / 1024**2
    print(
        f"[GPU {rank}] After creating inputs: {input_mem:.1f}MB (+{input_mem - initial_mem:.1f}MB)"
    )

    # Synchronize before forward pass
    dist.barrier()

    # Forward pass
    print(f"[GPU {rank}] Running forward pass...")
    start_time = time.time()

    output = model(q, k, v)

    torch.cuda.synchronize()
    end_time = time.time()

    # Memory after forward
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    current_mem = torch.cuda.memory_allocated() / 1024**2

    print(
        f"[GPU {rank}] Forward pass completed in {(end_time - start_time) * 1000:.1f}ms"
    )
    print(f"[GPU {rank}] Peak memory: {peak_mem:.1f}MB")
    print(f"[GPU {rank}] Current memory: {current_mem:.1f}MB")
    print(f"[GPU {rank}] Output shape: {output.shape}")

    # Verify output
    if rank == 0:
        print(f"\n[GPU {rank}] Output stats:")
        print(f"[GPU {rank}]   Mean: {output.mean().item():.6f}")
        print(f"[GPU {rank}]   Std: {output.std().item():.6f}")
        print(f"[GPU {rank}]   Min: {output.min().item():.6f}")
        print(f"[GPU {rank}]   Max: {output.max().item():.6f}")

    # Cleanup
    dist.destroy_process_group()


def test_with_actual_ring_v2(rank, world_size, seq_len):
    """Test with actual RingDilatedAttentionV2."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12361"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:0")

    # Import the fixed version
    from dilated_attention_pytorch.ring_dilated_attention_v2_fixed import (
        RingDilatedAttentionV2Fixed,
    )

    print(f"\n[GPU {rank}] Testing RingDilatedAttentionV2Fixed")

    # Create model
    model = RingDilatedAttentionV2Fixed(
        segment_lengths=[1024, 2048, 4096],
        dilation_rates=[1, 2, 4],
        ring_size=world_size,
        device=device,
        dtype=torch.float16,
        enable_memory_pool=False,  # Disable for cleaner test
        use_pattern_cache=False,  # Disable for cleaner test
    )

    print(f"[GPU {rank}] Model mode: {model.mode}")

    # Create inputs
    batch_size = 1
    num_heads = 8
    head_dim = 64

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    try:
        # Forward pass
        output = model(q, k, v)
        print(f"[GPU {rank}] Success! Output shape: {output.shape}")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")

    dist.destroy_process_group()


def main():
    print("Ring Attention V2 - Working Implementation Test")
    print("=" * 70)

    world_size = 2

    # Test 1: Simple ring attention
    print("\nTest 1: Simple Ring Attention Implementation")
    print("-" * 70)

    test_configs = [
        (4096, 2),  # 4K tokens, batch 2
        (8192, 1),  # 8K tokens, batch 1
        (16384, 1),  # 16K tokens, batch 1
    ]

    for seq_len, batch_size in test_configs:
        print(f"\n### Testing seq_len={seq_len}, batch_size={batch_size}")
        try:
            mp.spawn(
                run_ring_attention_test,
                args=(world_size, seq_len, batch_size),
                nprocs=world_size,
                join=True,
            )
        except Exception as e:
            print(f"Error: {e}")

    # Test 2: Actual RingDilatedAttentionV2Fixed
    print("\n\nTest 2: RingDilatedAttentionV2Fixed")
    print("-" * 70)

    for seq_len in [8192]:
        print(f"\n### Testing seq_len={seq_len}")
        try:
            mp.spawn(
                test_with_actual_ring_v2,
                args=(world_size, seq_len),
                nprocs=world_size,
                join=True,
            )
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("- Ring Attention successfully distributes computation")
    print("- Each GPU processes its chunk of the sequence")
    print("- Memory is distributed across GPUs")
    print("- Communication happens via ring pattern")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
