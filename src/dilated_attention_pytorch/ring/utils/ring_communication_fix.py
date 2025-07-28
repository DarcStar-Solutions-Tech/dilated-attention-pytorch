"""
Fixed ring communication utilities based on lucidrains/ring-attention-pytorch.

This module provides corrected ring communication functions that ensure proper
tensor contiguity and synchronization for multi-GPU ring attention.
"""

import torch
import torch.distributed as dist
from typing import Tuple, Optional


def send_and_receive_(
    send_tensor: torch.Tensor,
    receive_buffer: torch.Tensor,
    send_to_rank: int,
    receive_from_rank: int,
) -> torch.Tensor:
    """
    Send and receive tensors using P2P operations with proper contiguity.

    Based on lucidrains/ring-attention-pytorch implementation.

    Args:
        send_tensor: Tensor to send
        receive_buffer: Buffer to receive into
        send_to_rank: Rank to send to
        receive_from_rank: Rank to receive from

    Returns:
        The receive buffer with received data
    """
    # Ensure tensors are contiguous - CRITICAL for avoiding CUDA errors
    send_tensor = send_tensor.contiguous()
    receive_buffer = receive_buffer.contiguous()

    # Create P2P operations
    ops = []
    ops.append(dist.P2POp(dist.isend, send_tensor, send_to_rank))
    ops.append(dist.P2POp(dist.irecv, receive_buffer, receive_from_rank))

    # Execute all operations
    reqs = dist.batch_isend_irecv(ops)

    # Wait for completion
    for req in reqs:
        req.wait()

    # Synchronize all processes - CRITICAL for avoiding race conditions
    dist.barrier()

    return receive_buffer


def ring_pass_fixed(
    tensor: torch.Tensor,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Perform one ring pass with fixed communication.

    Args:
        tensor: Tensor to pass
        rank: Current rank (auto-detected if None)
        world_size: World size (auto-detected if None)

    Returns:
        Tensor received from previous rank
    """
    if not dist.is_initialized():
        return tensor

    if rank is None:
        rank = dist.get_rank()
    if world_size is None:
        world_size = dist.get_world_size()

    if world_size <= 1:
        return tensor

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Create receive buffer
    receive_buffer = torch.empty_like(tensor)

    # Send and receive
    return send_and_receive_(tensor, receive_buffer, next_rank, prev_rank)


def ring_pass_kv_fixed(
    k: torch.Tensor,
    v: torch.Tensor,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ring pass for both K and V tensors with fixed communication.

    Args:
        k: Key tensor
        v: Value tensor
        rank: Current rank (auto-detected if None)
        world_size: World size (auto-detected if None)

    Returns:
        Tuple of (k_received, v_received)
    """
    if not dist.is_initialized():
        return k, v

    if rank is None:
        rank = dist.get_rank()
    if world_size is None:
        world_size = dist.get_world_size()

    if world_size <= 1:
        return k, v

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Ensure contiguity
    k = k.contiguous()
    v = v.contiguous()

    # Create receive buffers
    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    # Create P2P operations for both tensors
    ops = []
    ops.append(dist.P2POp(dist.isend, k, next_rank, tag=0))
    ops.append(dist.P2POp(dist.irecv, k_recv, prev_rank, tag=0))
    ops.append(dist.P2POp(dist.isend, v, next_rank, tag=1))
    ops.append(dist.P2POp(dist.irecv, v_recv, prev_rank, tag=1))

    # Execute all operations
    reqs = dist.batch_isend_irecv(ops)

    # Wait for completion
    for req in reqs:
        req.wait()

    # Synchronize all processes
    dist.barrier()

    return k_recv, v_recv


def all_ring_pass_fixed(
    ring_size: int,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fixed version of all_ring_pass with proper contiguity handling.

    Args:
        ring_size: Size of the ring
        x: Tensor to pass
        y: Optional second tensor to pass

    Returns:
        Passed tensor
    """
    if not dist.is_initialized() or ring_size <= 1:
        return x

    # Simply use the fixed ring_pass
    return ring_pass_fixed(x)


__all__ = [
    "send_and_receive_",
    "ring_pass_fixed",
    "ring_pass_kv_fixed",
    "all_ring_pass_fixed",
]
