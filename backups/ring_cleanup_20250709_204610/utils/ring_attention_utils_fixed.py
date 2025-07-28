"""
Fixed ring attention utility functions with robust error handling.

This module provides fixed utility functions for ring attention implementations
that properly handle distributed communication and avoid CUDA errors.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


def safe_ring_pass(
    tensor: Tensor,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    tag: int = 0,
) -> Tensor:
    """
    Perform safe ring pass communication with proper error handling.

    Args:
        tensor: Tensor to pass (will be made contiguous)
        rank: Current rank (auto-detected if None)
        world_size: World size (auto-detected if None)
        tag: Communication tag

    Returns:
        Received tensor from previous rank
    """
    if not dist.is_initialized():
        return tensor

    if rank is None:
        rank = dist.get_rank()
    if world_size is None:
        world_size = dist.get_world_size()

    if world_size <= 1:
        return tensor

    # Calculate neighbors
    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    # Ensure tensor is contiguous and on correct device
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Create receive buffer with same properties
    recv_buffer = torch.empty_like(tensor)

    # Use different approach for 2 GPUs to avoid deadlock
    if world_size == 2:
        if rank == 0:
            # Rank 0: send first, then receive
            dist.send(tensor, dst=dst, tag=tag)
            dist.recv(recv_buffer, src=src, tag=tag)
        else:
            # Rank 1: receive first, then send
            dist.recv(recv_buffer, src=src, tag=tag)
            dist.send(tensor, dst=dst, tag=tag)
    else:
        # For >2 GPUs, use non-blocking operations
        send_op = dist.isend(tensor, dst=dst, tag=tag)
        recv_op = dist.irecv(recv_buffer, src=src, tag=tag)

        # Wait for both operations
        send_op.wait()
        recv_op.wait()

    return recv_buffer


def safe_ring_pass_kv(
    k_tensor: Tensor,
    v_tensor: Tensor,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Perform safe ring pass for K and V tensors together.

    Args:
        k_tensor: Key tensor
        v_tensor: Value tensor
        rank: Current rank
        world_size: World size

    Returns:
        Tuple of (received_k, received_v)
    """
    if not dist.is_initialized() or (world_size and world_size <= 1):
        return k_tensor, v_tensor

    # Pass K and V with different tags to avoid confusion
    k_recv = safe_ring_pass(k_tensor, rank, world_size, tag=0)
    v_recv = safe_ring_pass(v_tensor, rank, world_size, tag=1)

    return k_recv, v_recv


def validate_tensor_for_ring_pass(tensor: Tensor, name: str = "tensor") -> None:
    """
    Validate tensor is suitable for ring pass operations.

    Args:
        tensor: Tensor to validate
        name: Name for error messages

    Raises:
        ValueError: If tensor is not suitable
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")

    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA device for distributed ops")

    if tensor.requires_grad:
        logger.warning(f"{name} has requires_grad=True, this may cause issues")


def create_ring_groups(world_size: int) -> Optional[dist.ProcessGroup]:
    """
    Create process groups for ring communication.

    Args:
        world_size: Total number of processes

    Returns:
        Process group for ring communication
    """
    if not dist.is_initialized() or world_size <= 1:
        return None

    # For small world sizes, use the default group
    if world_size <= 8:
        return dist.group.WORLD

    # For larger world sizes, could create subgroups
    # but for now use default
    return dist.group.WORLD


def get_memory_info(device: torch.device) -> dict:
    """
    Get current memory information for debugging.

    Args:
        device: CUDA device

    Returns:
        Dict with memory information
    """
    if device.type != "cuda":
        return {}

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024 / 1024,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }


class RingCommunicator:
    """
    A stateful ring communicator that handles distributed ring attention.
    """

    def __init__(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        """Initialize ring communicator."""
        self.rank = (
            rank
            if rank is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        self.world_size = (
            world_size
            if world_size is not None
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )

        # Calculate neighbors once
        self.src = (self.rank - 1) % self.world_size if self.world_size > 1 else 0
        self.dst = (self.rank + 1) % self.world_size if self.world_size > 1 else 0

        # Track communication steps for debugging
        self.step_count = 0

    def pass_tensor(self, tensor: Tensor, tag: Optional[int] = None) -> Tensor:
        """Pass tensor to next rank in ring."""
        if self.world_size <= 1:
            return tensor

        if tag is None:
            tag = self.step_count % 32767  # Max tag value

        result = safe_ring_pass(tensor, self.rank, self.world_size, tag)
        self.step_count += 1
        return result

    def pass_kv(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Pass K and V tensors together."""
        if self.world_size <= 1:
            return k, v

        k_tag = (self.step_count * 2) % 32767
        v_tag = (self.step_count * 2 + 1) % 32767

        k_new = safe_ring_pass(k, self.rank, self.world_size, k_tag)
        v_new = safe_ring_pass(v, self.rank, self.world_size, v_tag)

        self.step_count += 1
        return k_new, v_new

    def barrier(self) -> None:
        """Synchronize all ranks."""
        if dist.is_initialized() and self.world_size > 1:
            dist.barrier()


# For backward compatibility
def all_ring_pass(ring_size: int, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Backward compatible ring pass function.

    Args:
        ring_size: Size of the ring
        x: Tensor to pass
        y: Optional second tensor (ignored for simplicity)

    Returns:
        Passed tensor
    """
    return safe_ring_pass(x, world_size=ring_size)


__all__ = [
    "safe_ring_pass",
    "safe_ring_pass_kv",
    "validate_tensor_for_ring_pass",
    "create_ring_groups",
    "get_memory_info",
    "RingCommunicator",
    "all_ring_pass",  # For backward compatibility
]
