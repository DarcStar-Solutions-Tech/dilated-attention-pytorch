"""
Ring Attention with custom autograd for proper gradient handling.

This module implements ring attention with both forward and backward passes,
including support for Hilbert SFC optimization.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

logger = logging.getLogger(__name__)


class RingAttentionFunction(Function):
    """
    Custom autograd function for ring attention.

    Handles both forward and backward passes with proper gradient communication.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: float,
        dropout_p: float,
        is_causal: bool,
        rank: int,
        world_size: int,
        use_hilbert: bool = False,
        hilbert_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of ring attention.

        Args:
            q: Query tensor [batch, heads, seq, dim]
            k: Key tensor [batch, heads, seq, dim]
            v: Value tensor [batch, heads, seq, dim]
            scale: Scaling factor (1/sqrt(dim))
            dropout_p: Dropout probability
            is_causal: Whether to use causal mask
            rank: Current rank
            world_size: Total world size
            use_hilbert: Whether to use Hilbert ordering
            hilbert_indices: Hilbert curve indices for reordering

        Returns:
            Attention output [batch, heads, seq, dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Apply Hilbert ordering if requested
        if use_hilbert and hilbert_indices is not None:
            q = q.gather(2, hilbert_indices.unsqueeze(-1).expand_as(q))
            k = k.gather(2, hilbert_indices.unsqueeze(-1).expand_as(k))
            v = v.gather(2, hilbert_indices.unsqueeze(-1).expand_as(v))

        # Scale queries
        q = q * scale

        # Initialize output and LSE
        output = torch.zeros_like(q)
        lse = torch.full(
            (batch_size, num_heads, seq_len),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )

        # Clone K,V for ring passing
        k_chunk = k.clone()
        v_chunk = v.clone()

        # Save for backward
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.scale = scale
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.use_hilbert = use_hilbert
        ctx.hilbert_indices = hilbert_indices

        # Ring attention forward
        for step in range(world_size):
            # Which rank's KV are we processing?
            kv_rank = (rank - step) % world_size

            # Compute attention scores
            scores = torch.matmul(q, k_chunk.transpose(-2, -1))

            # Apply causal mask if needed
            if is_causal and kv_rank >= rank:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1 if kv_rank == rank else -seq_len,
                )
                scores = scores + causal_mask

            # Compute new max for numerical stability
            scores_max = scores.max(dim=-1, keepdim=True).values
            scores_max = torch.maximum(scores_max.squeeze(-1), lse)

            # Update output with stable softmax
            exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
            exp_lse = torch.exp(lse - scores_max)

            # Update output
            output = output * exp_lse.unsqueeze(-1) + torch.matmul(exp_scores, v_chunk)

            # Update LSE
            lse = scores_max + torch.log(exp_lse + exp_scores.sum(dim=-1))

            # Ring pass KV (except last step)
            if step < world_size - 1:
                k_chunk, v_chunk = ring_pass_kv(k_chunk, v_chunk, rank, world_size)

        # Final normalization
        output = output / lse.unsqueeze(-1).clamp(min=1e-6)

        # Apply dropout if training
        if dropout_p > 0 and q.requires_grad:
            output = F.dropout(output, p=dropout_p, training=True)

        # Reverse Hilbert ordering if used
        if use_hilbert and hilbert_indices is not None:
            # Create inverse indices
            inverse_indices = torch.argsort(hilbert_indices)
            output = output.gather(2, inverse_indices.unsqueeze(-1).expand_as(output))

        # Save output stats for backward
        ctx.output_for_backward = output.clone()

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of ring attention.

        Args:
            grad_output: Gradient w.r.t output [batch, heads, seq, dim]

        Returns:
            Gradients for q, k, v, and None for other inputs
        """
        q, k, v, output, lse = ctx.saved_tensors
        scale = ctx.scale
        rank = ctx.rank
        world_size = ctx.world_size
        use_hilbert = ctx.use_hilbert
        hilbert_indices = ctx.hilbert_indices

        batch_size, num_heads, seq_len, head_dim = q.shape

        # Apply Hilbert ordering to saved tensors and gradients if needed
        if use_hilbert and hilbert_indices is not None:
            q = q.gather(2, hilbert_indices.unsqueeze(-1).expand_as(q))
            k = k.gather(2, hilbert_indices.unsqueeze(-1).expand_as(k))
            v = v.gather(2, hilbert_indices.unsqueeze(-1).expand_as(v))
            grad_output = grad_output.gather(
                2, hilbert_indices.unsqueeze(-1).expand_as(grad_output)
            )
            output = ctx.output_for_backward
            output = output.gather(2, hilbert_indices.unsqueeze(-1).expand_as(output))

        # Scale queries for backward
        q = q * scale

        # Initialize gradients
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        # Ring attention backward
        k_chunk = k.clone()
        v_chunk = v.clone()
        grad_k_chunk = grad_k.clone()
        grad_v_chunk = grad_v.clone()

        for step in range(world_size):
            # Which rank's KV are we processing?
            kv_rank = (rank - step) % world_size

            # Recompute attention scores
            scores = torch.matmul(q, k_chunk.transpose(-2, -1))

            # Apply causal mask if needed
            if ctx.is_causal and kv_rank >= rank:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1 if kv_rank == rank else -seq_len,
                )
                scores = scores + causal_mask

            # Compute attention weights
            attn_weights = F.softmax(scores, dim=-1)

            # Gradient w.r.t V
            grad_v_chunk = grad_v_chunk + torch.matmul(
                attn_weights.transpose(-2, -1), grad_output
            )

            # Gradient w.r.t attention weights
            grad_attn = torch.matmul(grad_output, v_chunk.transpose(-2, -1))

            # Gradient w.r.t scores (softmax backward)
            grad_scores = grad_attn * attn_weights
            sum_grad = grad_scores.sum(dim=-1, keepdim=True)
            grad_scores = grad_scores - attn_weights * sum_grad

            # Gradient w.r.t Q and K
            grad_q = grad_q + torch.matmul(grad_scores, k_chunk) * scale
            grad_k_chunk = grad_k_chunk + torch.matmul(grad_scores.transpose(-2, -1), q)

            # Ring pass KV and gradients (except last step)
            if step < world_size - 1:
                # Forward direction for KV
                k_chunk, v_chunk = ring_pass_kv(k_chunk, v_chunk, rank, world_size)
                # Backward direction for gradients
                grad_k_chunk, grad_v_chunk = ring_pass_gradients(
                    grad_k_chunk, grad_v_chunk, rank, world_size
                )

        # Final gradient accumulation
        grad_k = grad_k + grad_k_chunk
        grad_v = grad_v + grad_v_chunk

        # Reverse Hilbert ordering for gradients if used
        if use_hilbert and hilbert_indices is not None:
            inverse_indices = torch.argsort(hilbert_indices)
            grad_q = grad_q.gather(2, inverse_indices.unsqueeze(-1).expand_as(grad_q))
            grad_k = grad_k.gather(2, inverse_indices.unsqueeze(-1).expand_as(grad_k))
            grad_v = grad_v.gather(2, inverse_indices.unsqueeze(-1).expand_as(grad_v))

        # Return gradients (None for non-tensor inputs)
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None


def ring_pass_kv(
    k: Tensor, v: Tensor, rank: int, world_size: int
) -> Tuple[Tensor, Tensor]:
    """
    Ring pass for K and V tensors in forward direction.

    Each rank sends to next and receives from previous.
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
        # Special handling for 2 GPUs
        if rank == 0:
            dist.send(k, dst=dst, tag=0)
            dist.recv(k_recv, src=src, tag=0)
            dist.send(v, dst=dst, tag=1)
            dist.recv(v_recv, src=src, tag=1)
        else:
            dist.recv(k_recv, src=src, tag=0)
            dist.send(k, dst=dst, tag=0)
            dist.recv(v_recv, src=src, tag=1)
            dist.send(v, dst=dst, tag=1)
    else:
        # Non-blocking for >2 GPUs
        reqs = []
        reqs.append(dist.isend(k, dst=dst, tag=0))
        reqs.append(dist.irecv(k_recv, src=src, tag=0))
        reqs.append(dist.isend(v, dst=dst, tag=1))
        reqs.append(dist.irecv(v_recv, src=src, tag=1))

        for req in reqs:
            req.wait()

    return k_recv, v_recv


def ring_pass_gradients(
    grad_k: Tensor, grad_v: Tensor, rank: int, world_size: int
) -> Tuple[Tensor, Tensor]:
    """
    Ring pass for gradients in backward direction.

    Each rank sends to previous and receives from next (opposite of forward).
    """
    if world_size <= 1:
        return grad_k, grad_v

    # Reverse direction for gradients
    src = (rank + 1) % world_size
    dst = (rank - 1) % world_size

    # Ensure contiguous
    grad_k = grad_k.contiguous()
    grad_v = grad_v.contiguous()

    # Create receive buffers
    grad_k_recv = torch.empty_like(grad_k)
    grad_v_recv = torch.empty_like(grad_v)

    if world_size == 2:
        # Special handling for 2 GPUs
        if rank == 0:
            dist.send(grad_k, dst=dst, tag=2)
            dist.recv(grad_k_recv, src=src, tag=2)
            dist.send(grad_v, dst=dst, tag=3)
            dist.recv(grad_v_recv, src=src, tag=3)
        else:
            dist.recv(grad_k_recv, src=src, tag=2)
            dist.send(grad_k, dst=dst, tag=2)
            dist.recv(grad_v_recv, src=src, tag=3)
            dist.send(grad_v, dst=dst, tag=3)
    else:
        # Non-blocking for >2 GPUs
        reqs = []
        reqs.append(dist.isend(grad_k, dst=dst, tag=2))
        reqs.append(dist.irecv(grad_k_recv, src=src, tag=2))
        reqs.append(dist.isend(grad_v, dst=dst, tag=3))
        reqs.append(dist.irecv(grad_v_recv, src=src, tag=3))

        for req in reqs:
            req.wait()

    return grad_k_recv, grad_v_recv


def ring_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    use_hilbert: bool = False,
    hilbert_indices: Optional[Tensor] = None,
) -> Tensor:
    """
    Ring attention with custom autograd.

    Args:
        q: Query tensor [batch, heads, seq, dim]
        k: Key tensor [batch, heads, seq, dim]
        v: Value tensor [batch, heads, seq, dim]
        scale: Scaling factor (defaults to 1/sqrt(dim))
        dropout_p: Dropout probability
        is_causal: Whether to use causal mask
        use_hilbert: Whether to use Hilbert ordering
        hilbert_indices: Hilbert curve indices

    Returns:
        Attention output [batch, heads, seq, dim]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    return RingAttentionFunction.apply(
        q,
        k,
        v,
        scale,
        dropout_p,
        is_causal,
        rank,
        world_size,
        use_hilbert,
        hilbert_indices,
    )
