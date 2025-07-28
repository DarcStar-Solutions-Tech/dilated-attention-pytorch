"""Mixin class providing ring communication functionality.

This module provides reusable ring communication patterns for distributed
attention implementations.
"""

from typing import Tuple, Union, Any
import warnings
import time

import torch
import torch.distributed as dist
from torch import Tensor


class RingCommunicationMixin:
    """Mixin providing ring communication patterns.

    This class provides standardized communication methods for ring attention,
    including retry logic, error handling, and performance monitoring.
    """

    def __init__(self):
        """Initialize communication mixin."""
        self._comm_buffers = {}
        self._comm_stats = {
            "sends": 0,
            "recvs": 0,
            "failures": 0,
            "retries": 0,
            "total_bytes": 0,
            "total_time": 0.0,
        }

    def ring_pass_forward(
        self,
        tensor: Tensor,
        tag: int = 0,
        async_op: bool = False,
        max_retries: int = 3,
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Pass tensor forward in ring (rank i sends to rank i+1).

        Args:
            tensor: Tensor to send
            tag: Communication tag
            async_op: Whether to return async handle
            max_retries: Maximum retry attempts

        Returns:
            Received tensor (or tuple with async handle if async_op=True)
        """
        if not self.is_distributed:
            return (tensor.clone(), None) if async_op else tensor.clone()

        src = (self.rank - 1) % self.world_size
        dst = (self.rank + 1) % self.world_size

        return self._ring_communication_with_retry(
            tensor, src, dst, tag, async_op, max_retries
        )

    def ring_pass_backward(
        self,
        tensor: Tensor,
        tag: int = 0,
        async_op: bool = False,
        max_retries: int = 3,
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Pass tensor backward in ring (rank i sends to rank i-1).

        Args:
            tensor: Tensor to send
            tag: Communication tag
            async_op: Whether to return async handle
            max_retries: Maximum retry attempts

        Returns:
            Received tensor (or tuple with async handle if async_op=True)
        """
        if not self.is_distributed:
            return (tensor.clone(), None) if async_op else tensor.clone()

        src = (self.rank + 1) % self.world_size
        dst = (self.rank - 1) % self.world_size

        return self._ring_communication_with_retry(
            tensor, src, dst, tag, async_op, max_retries
        )

    def _ring_communication_with_retry(
        self,
        tensor: Tensor,
        src: int,
        dst: int,
        tag: int,
        async_op: bool,
        max_retries: int,
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Perform ring communication with retry logic.

        Args:
            tensor: Tensor to communicate
            src: Source rank
            dst: Destination rank
            tag: Communication tag
            async_op: Whether to use async operations
            max_retries: Maximum retry attempts

        Returns:
            Received tensor (or tuple with async handle)
        """
        # Get or allocate receive buffer
        recv_buffer = self._get_comm_buffer(tensor.shape, tensor.dtype, "recv")

        # Try communication with retries
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Ensure tensor is contiguous
                send_tensor = tensor.contiguous()

                # Perform communication
                if async_op:
                    send_handle = dist.isend(send_tensor, dst, tag=tag)
                    recv_handle = dist.irecv(recv_buffer, src, tag=tag)

                    # Update stats
                    self._update_comm_stats(tensor, time.time() - start_time)

                    return recv_buffer, (send_handle, recv_handle)
                else:
                    send_req = dist.isend(send_tensor, dst, tag=tag)
                    recv_req = dist.irecv(recv_buffer, src, tag=tag)

                    send_req.wait()
                    recv_req.wait()

                    # Update stats
                    self._update_comm_stats(tensor, time.time() - start_time)

                    return recv_buffer

            except Exception as e:
                self._comm_stats["failures"] += 1
                if attempt < max_retries - 1:
                    self._comm_stats["retries"] += 1
                    warnings.warn(
                        f"Ring communication failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    # Small delay before retry
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Ring communication failed after {max_retries} attempts: {e}"
                    )

    def _get_comm_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        buffer_type: str,
    ) -> Tensor:
        """Get or allocate communication buffer.

        Args:
            shape: Buffer shape
            dtype: Buffer dtype
            buffer_type: "send" or "recv"

        Returns:
            Buffer tensor
        """
        key = (shape, dtype, buffer_type)

        if key not in self._comm_buffers:
            # Allocate new buffer
            self._comm_buffers[key] = torch.empty(
                shape, dtype=dtype, device=self.device
            )

        return self._comm_buffers[key]

    def _update_comm_stats(self, tensor: Tensor, comm_time: float):
        """Update communication statistics.

        Args:
            tensor: Communicated tensor
            comm_time: Communication time in seconds
        """
        self._comm_stats["sends"] += 1
        self._comm_stats["recvs"] += 1
        self._comm_stats["total_bytes"] += tensor.numel() * tensor.element_size()
        self._comm_stats["total_time"] += comm_time

    def get_communication_stats(self) -> dict:
        """Get communication statistics.

        Returns:
            Dictionary of communication stats
        """
        stats = self._comm_stats.copy()

        # Calculate derived stats
        if stats["sends"] > 0:
            stats["avg_time_per_comm"] = stats["total_time"] / stats["sends"]
            stats["avg_bytes_per_comm"] = stats["total_bytes"] / stats["sends"]

            if stats["total_time"] > 0:
                stats["bandwidth_gbps"] = (
                    stats["total_bytes"] / stats["total_time"] / 1e9 * 8
                )

        return stats

    def reset_communication_stats(self):
        """Reset communication statistics."""
        self._comm_stats = {
            "sends": 0,
            "recvs": 0,
            "failures": 0,
            "retries": 0,
            "total_bytes": 0,
            "total_time": 0.0,
        }

    def _synchronize_ring(self):
        """Synchronize all processes in the ring."""
        if self.is_distributed:
            dist.barrier()

    def validate_ring_setup(self) -> bool:
        """Validate ring communication setup.

        Returns:
            True if setup is valid
        """
        if not self.is_distributed:
            return True

        try:
            # Test communication with neighbors
            test_tensor = torch.ones(1, device=self.device, dtype=self.dtype)

            # Forward pass test
            recv_forward = self.ring_pass_forward(test_tensor, tag=99999)
            if not torch.allclose(recv_forward, test_tensor):
                return False

            # Backward pass test
            recv_backward = self.ring_pass_backward(test_tensor, tag=99998)
            if not torch.allclose(recv_backward, test_tensor):
                return False

            # Synchronize
            self._synchronize_ring()

            return True

        except Exception as e:
            warnings.warn(f"Ring validation failed: {e}")
            return False

    def _get_ring_neighbors(self) -> Tuple[int, int]:
        """Get forward and backward neighbor ranks.

        Returns:
            Tuple of (forward_neighbor, backward_neighbor)
        """
        if not self.is_distributed:
            return (0, 0)

        forward = (self.rank + 1) % self.world_size
        backward = (self.rank - 1) % self.world_size

        return forward, backward
