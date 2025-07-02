"""
Ring attention utilities based on lucidrains/ring-attention-pytorch.
Provides proper distributed coordination for ring communication.
"""

from typing import Optional, Tuple
from collections import namedtuple

import torch
import torch.distributed as dist
from torch import Tensor


# Helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# Ring topology functions


def circular_index_left(pos, ring_size, num=1):
    return (pos - num) % ring_size


def circular_index_right(pos, ring_size, num=1):
    return (pos + num) % ring_size


def circular_rank_left(rank=None, ring_size=None, num=1):
    rank = default(rank, dist.get_rank() if dist.is_initialized() else 0)
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )
    ring_rank = rank % ring_size
    return circular_index_left(ring_rank, ring_size, num)


def circular_rank_right(rank=None, ring_size=None, num=1):
    rank = default(rank, dist.get_rank() if dist.is_initialized() else 0)
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )
    ring_rank = rank % ring_size
    return circular_index_right(ring_rank, ring_size, num)


# Ring communication


def send_and_receive_(x, left_rank, right_rank, ring_size):
    """
    Send tensor to right rank and receive from left rank.
    Uses batch_isend_irecv for efficiency.
    """
    if not dist.is_initialized() or ring_size == 1:
        return x

    # Create P2P operations
    send_op = dist.P2POp(dist.isend, x, right_rank)
    recv_op = dist.P2POp(dist.irecv, x, left_rank)

    # Execute batch communication
    reqs = dist.batch_isend_irecv([send_op, recv_op])

    # Wait for completion
    for req in reqs:
        req.wait()

    # Synchronize
    dist.barrier()

    return x


def ring_pass(
    x: Tensor, receive_buffer: Optional[Tensor] = None, ring_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """
    Perform one ring pass - send to next rank, receive from previous.
    Returns (received_tensor, original_tensor_for_next_send).
    """
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )

    if ring_size == 1:
        return x, x

    # Ensure contiguous
    x = x.contiguous()

    # Allocate receive buffer if needed
    if not exists(receive_buffer):
        receive_buffer = torch.zeros_like(x)

    # Get neighboring ranks
    left = circular_rank_left(ring_size=ring_size)
    right = circular_rank_right(ring_size=ring_size)

    # Copy to receive buffer
    receive_buffer.copy_(x)

    # Perform ring pass
    send_and_receive_(receive_buffer, left, right, ring_size)

    return receive_buffer, x


# Ring iteration helpers

RingInfo = namedtuple("RingInfo", ["ring_rank", "iter"])


def null_ring_pass(x, ring_size=None):
    """Single pass iterator for non-distributed case."""
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )
    x = cast_tuple(x)
    yield (RingInfo(0, 0), x)


def all_ring_pass(x, receive_buffer=None, ring_size=None, max_iters=None, start_rank=0):
    """
    Iterate through all ring passes.
    Yields (RingInfo, tensors) for each ring position.
    """
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )
    max_iters = default(max_iters, ring_size)

    # Single device case
    if ring_size == 1:
        yield from null_ring_pass(x, ring_size)
        return

    # Ensure tuple
    x = cast_tuple(x)
    receive_buffer = cast_tuple(receive_buffer, len(x))

    # Get current ring rank
    rank = dist.get_rank() if dist.is_initialized() else 0
    ring_rank = rank % ring_size

    # Initial yield
    yield (RingInfo(ring_rank, 0), x)

    # Ring iterations
    for i in range(1, max_iters):
        # Perform ring pass for each tensor
        new_x = []
        new_receive = []

        for tensor, buffer in zip(x, receive_buffer):
            if exists(tensor):
                received, to_send = ring_pass(tensor, buffer, ring_size)
                new_x.append(received)
                new_receive.append(to_send)
            else:
                new_x.append(None)
                new_receive.append(None)

        x = tuple(new_x)
        receive_buffer = tuple(new_receive)

        # Calculate current ring position
        ring_pos = (ring_rank - i) % ring_size
        yield (RingInfo(ring_pos, i), x)


# Bucketing utilities for efficient communication


def split_by_rank(x, rank=None, ring_size=None):
    """Split tensor evenly across ranks."""
    rank = default(rank, dist.get_rank() if dist.is_initialized() else 0)
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )

    if ring_size == 1:
        return x

    # Assume sequence dimension is 1
    seq_len = x.shape[1]
    assert seq_len % ring_size == 0, (
        f"Sequence length {seq_len} must be divisible by ring size {ring_size}"
    )

    chunk_size = seq_len // ring_size
    start = rank * chunk_size
    end = start + chunk_size

    return x[:, start:end]


def gather_from_rank(x, rank=None, ring_size=None):
    """Gather tensor chunks from all ranks (for comparison/testing only)."""
    rank = default(rank, dist.get_rank() if dist.is_initialized() else 0)
    ring_size = default(
        ring_size, dist.get_world_size() if dist.is_initialized() else 1
    )

    if ring_size == 1:
        return x

    # Gather all chunks
    chunks = [torch.empty_like(x) for _ in range(ring_size)]
    dist.all_gather(chunks, x)

    # Concatenate along sequence dimension
    return torch.cat(chunks, dim=1)


# Attention-specific utilities


def create_causal_mask(q_pos, kv_pos, device):
    """Create causal mask for ring attention."""
    q_len = q_pos.shape[0] if hasattr(q_pos, "shape") else len(q_pos)
    kv_len = kv_pos.shape[0] if hasattr(kv_pos, "shape") else len(kv_pos)

    if isinstance(q_pos, int):
        q_indices = torch.arange(q_pos, q_pos + q_len, device=device)
    else:
        q_indices = q_pos

    if isinstance(kv_pos, int):
        kv_indices = torch.arange(kv_pos, kv_pos + kv_len, device=device)
    else:
        kv_indices = kv_pos

    # Causal mask: q can only attend to kv with index <= q
    mask = q_indices.unsqueeze(1) >= kv_indices.unsqueeze(0)
    return mask


def apply_ring_mask(
    scores, q_bucket_idx, kv_bucket_idx, ring_rank, iter_idx, causal=True
):
    """Apply appropriate masking for ring attention."""
    if not causal:
        return scores

    # Create position indices
    device = scores.device
    batch, heads, q_len, kv_len = scores.shape

    q_pos = q_bucket_idx * q_len
    kv_pos = kv_bucket_idx * kv_len

    # Adjust positions based on ring iteration
    kv_pos = kv_pos + (ring_rank - iter_idx) * kv_len

    # Create and apply mask
    mask = create_causal_mask(q_pos, kv_pos, device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims

    scores.masked_fill_(~mask, float("-inf"))
    return scores
