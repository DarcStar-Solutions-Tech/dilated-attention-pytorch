"""
Simple fixed Ring Dilated Attention for testing distributed functionality.

This is a minimal implementation focused on fixing the ring communication issues.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def safe_ring_pass(
    k: torch.Tensor, v: torch.Tensor, rank: int, world_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe ring pass for K and V tensors.

    Special handling for 2 GPU case to avoid deadlock.
    """
    if world_size <= 1:
        return k, v

    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    # Ensure contiguous
    k = k.contiguous()
    v = v.contiguous()

    # Create receive buffers
    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    if world_size == 2:
        # Special case for 2 GPUs to avoid deadlock
        if rank == 0:
            # Rank 0: send K first
            dist.send(k, dst=dst, tag=0)
            dist.recv(k_recv, src=src, tag=0)
            # Then V
            dist.send(v, dst=dst, tag=1)
            dist.recv(v_recv, src=src, tag=1)
        else:
            # Rank 1: receive K first
            dist.recv(k_recv, src=src, tag=0)
            dist.send(k, dst=dst, tag=0)
            # Then V
            dist.recv(v_recv, src=src, tag=1)
            dist.send(v, dst=dst, tag=1)
    else:
        # For >2 GPUs, use non-blocking
        k_send_op = dist.isend(k, dst=dst, tag=0)
        k_recv_op = dist.irecv(k_recv, src=src, tag=0)
        v_send_op = dist.isend(v, dst=dst, tag=1)
        v_recv_op = dist.irecv(v_recv, src=src, tag=1)

        k_send_op.wait()
        k_recv_op.wait()
        v_send_op.wait()
        v_recv_op.wait()

    return k_recv, v_recv


class RingDilatedAttentionFixedSimple(nn.Module):
    """
    Simple fixed ring dilated attention for testing.

    Focuses on correct distributed communication without complex features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Device and dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        self.device = device
        self.dtype = dtype

        # Linear layers
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Move to device
        self.qkv_proj = self.qkv_proj.to(device=device, dtype=dtype)
        self.out_proj = self.out_proj.to(device=device, dtype=dtype)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Distributed info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        logger.info(
            f"Initialized RingDilatedAttentionFixedSimple (rank={self.rank}, world_size={self.world_size})"
        )

    def forward(
        self,
        x: torch.Tensor,
        total_seq_len: Optional[int] = None,
        already_split: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with fixed ring communication.

        Args:
            x: Input tensor [batch, seq, embed_dim]
            total_seq_len: Total sequence length across all GPUs
            already_split: If True, x is already the local chunk

        Returns:
            Output tensor [batch, local_seq, embed_dim]
        """
        batch_size = x.shape[0]

        # Ensure input is contiguous
        x = x.contiguous()

        # Handle splitting
        if self.world_size > 1 and not already_split:
            seq_len = x.shape[1]
            assert seq_len % self.world_size == 0

            local_seq_len = seq_len // self.world_size
            start = self.rank * local_seq_len
            end = start + local_seq_len

            x_local = x[:, start:end, :].contiguous()
            total_seq_len = seq_len
        else:
            x_local = x
            local_seq_len = x.shape[1]
            if total_seq_len is None:
                total_seq_len = local_seq_len * self.world_size

        # QKV projection
        qkv = self.qkv_proj(x_local)
        qkv = qkv.reshape(batch_size, local_seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]

        # Scale Q
        q_local = q_local * self.scale

        # Ring attention
        if self.world_size > 1:
            output = self._ring_forward(q_local, k_local, v_local)
        else:
            output = self._local_forward(q_local, k_local, v_local)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, local_seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output

    def _ring_forward(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
    ) -> torch.Tensor:
        """Ring attention forward pass."""
        batch_size, num_heads, seq_len, head_dim = q_local.shape

        # Initialize output
        output = torch.zeros_like(q_local)
        normalizer = torch.zeros(
            batch_size,
            num_heads,
            seq_len,
            1,
            device=q_local.device,
            dtype=q_local.dtype,
        )

        # Clone K,V for ring passing
        k_chunk = k_local.clone()
        v_chunk = v_local.clone()

        # Synchronize before starting
        if dist.is_initialized():
            dist.barrier()

        for step in range(self.world_size):
            # Compute attention scores
            scores = torch.matmul(q_local, k_chunk.transpose(-2, -1))

            # Softmax (simplified - in practice would use LSE trick)
            attn_weights = torch.softmax(scores, dim=-1)

            # Accumulate weighted values
            chunk_output = torch.matmul(attn_weights, v_chunk)
            output = output + chunk_output
            normalizer = normalizer + attn_weights.sum(dim=-1, keepdim=True)

            # Ring pass (except last step)
            if step < self.world_size - 1:
                k_chunk, v_chunk = safe_ring_pass(
                    k_chunk, v_chunk, self.rank, self.world_size
                )

        # Normalize (simplified - in practice would use LSE)
        output = output / (normalizer + 1e-8)

        # Final synchronization
        if dist.is_initialized():
            dist.barrier()

        return output

    def _local_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Local attention forward (single GPU)."""
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output
