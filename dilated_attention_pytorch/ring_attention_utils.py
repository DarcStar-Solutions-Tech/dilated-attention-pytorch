"""
Ring attention utility functions.

This module provides common utility functions used by ring attention implementations.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def exists(val) -> bool:
    """Check if value exists (not None)."""
    return val is not None


def all_ring_pass(ring_size: int, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Perform all-to-all ring pass communication.

    Args:
        ring_size: Size of the ring
        x: Tensor to pass
        y: Optional second tensor to pass

    Returns:
        Passed tensor
    """
    if not dist.is_initialized() or ring_size <= 1:
        return x

    # Simple ring pass - each rank sends to next and receives from previous
    rank = dist.get_rank()
    next_rank = (rank + 1) % ring_size
    prev_rank = (rank - 1) % ring_size

    # Create send/recv buffers
    send_buffer = x.contiguous()
    recv_buffer = torch.empty_like(x)

    # Send to next, receive from previous
    send_op = dist.isend(send_buffer, dst=next_rank)
    recv_op = dist.irecv(recv_buffer, src=prev_rank)

    send_op.wait()
    recv_op.wait()

    return recv_buffer


def split_by_rank(tensor: Tensor, rank: int, world_size: int, dim: int = 0) -> Tensor:
    """
    Split tensor by rank for distributed processing.

    Args:
        tensor: Tensor to split
        rank: Current rank
        world_size: Total number of ranks
        dim: Dimension to split along

    Returns:
        Split tensor for current rank
    """
    if world_size <= 1:
        return tensor

    # Calculate chunk size
    total_size = tensor.shape[dim]
    chunk_size = total_size // world_size
    remainder = total_size % world_size

    # Calculate start and end indices for this rank
    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    # Create slice
    indices = [slice(None)] * tensor.ndim
    indices[dim] = slice(start, end)

    return tensor[tuple(indices)]


def create_causal_mask(
    seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Create a causal mask for attention.

    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type of mask

    Returns:
        Causal mask tensor
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return mask


__all__ = [
    "exists",
    "all_ring_pass",
    "split_by_rank",
    "create_causal_mask",
]
