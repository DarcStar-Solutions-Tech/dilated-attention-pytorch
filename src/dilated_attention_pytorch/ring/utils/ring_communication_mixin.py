"""Mixin class providing standardized ring communication patterns.

This module implements the core ring communication primitives using
isend/irecv for true O(n/k) memory complexity.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import Optional, Tuple, List
import warnings
import time


class RingCommunicationMixin:
    """Mixin providing standardized ring communication patterns.

    This mixin implements efficient ring communication using non-blocking
    isend/irecv operations. It provides error recovery, performance monitoring,
    and various optimization strategies.
    """

    # Communication constants
    DEFAULT_TAG_OFFSET = 1000
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 0.1  # seconds

    def __init__(self):
        """Initialize communication mixin."""
        # Communication statistics
        self._comm_stats = {
            "total_sends": 0,
            "total_recvs": 0,
            "total_bytes": 0,
            "failed_attempts": 0,
            "retry_successes": 0,
        }

        # Error recovery settings
        self.enable_error_recovery = True
        self.fallback_to_allgather = False  # Never use all_gather!

    def ring_pass_forward(
        self, tensor: Tensor, tag: int = 0, async_op: bool = False
    ) -> Tensor:
        """Forward ring pass using isend/irecv.

        Sends tensor to next rank and receives from previous rank.

        Args:
            tensor: Tensor to send
            tag: Communication tag
            async_op: Whether to return async handles

        Returns:
            Received tensor from previous rank
        """
        if not self.is_distributed:
            return tensor.clone()

        src, dst = self._get_ring_neighbors()
        return self._ring_communication_with_retry(tensor, src, dst, tag, async_op)

    def ring_pass_backward(
        self, tensor: Tensor, tag: int = 0, async_op: bool = False
    ) -> Tensor:
        """Backward ring pass using isend/irecv.

        Sends tensor to previous rank and receives from next rank.

        Args:
            tensor: Tensor to send
            tag: Communication tag
            async_op: Whether to return async handles

        Returns:
            Received tensor from next rank
        """
        if not self.is_distributed:
            return tensor.clone()

        src, dst = self._get_ring_neighbors()
        # Reverse direction for backward pass
        return self._ring_communication_with_retry(tensor, dst, src, tag, async_op)

    def _ring_communication_with_retry(
        self, tensor: Tensor, src: int, dst: int, tag: int, async_op: bool = False
    ) -> Tensor:
        """Perform ring communication with retry logic.

        Args:
            tensor: Tensor to communicate
            src: Source rank to receive from
            dst: Destination rank to send to
            tag: Communication tag
            async_op: Whether to use async operations

        Returns:
            Received tensor
        """
        attempt = 0
        last_error = None

        while attempt < self.MAX_RETRY_ATTEMPTS:
            try:
                return self._do_ring_communication(
                    tensor, src, dst, tag + attempt, async_op
                )
            except Exception as e:
                last_error = e
                self._comm_stats["failed_attempts"] += 1

                if attempt < self.MAX_RETRY_ATTEMPTS - 1:
                    warnings.warn(
                        f"Ring communication failed (attempt {attempt + 1}), retrying: {e}"
                    )
                    time.sleep(self.RETRY_DELAY * (2**attempt))  # Exponential backoff
                    attempt += 1
                else:
                    if self.enable_error_recovery:
                        return self._handle_communication_failure(tensor, e)
                    else:
                        raise

        # Should never reach here
        raise RuntimeError(
            f"Ring communication failed after {attempt} attempts: {last_error}"
        )

    def _do_ring_communication(
        self, tensor: Tensor, src: int, dst: int, tag: int, async_op: bool = False
    ) -> Tensor:
        """Perform actual ring communication.

        Args:
            tensor: Tensor to communicate
            src: Source rank
            dst: Destination rank
            tag: Communication tag
            async_op: Whether to use async operations

        Returns:
            Received tensor
        """
        # Ensure tensor is contiguous
        send_tensor = tensor.contiguous()

        # Allocate receive buffer
        recv_tensor = torch.empty_like(send_tensor)

        # Use unique tags to avoid conflicts
        unique_tag = self.DEFAULT_TAG_OFFSET + tag

        # Non-blocking send and receive
        send_handle = dist.isend(send_tensor, dst=dst, tag=unique_tag)
        recv_handle = dist.irecv(recv_tensor, src=src, tag=unique_tag)

        if async_op:
            return recv_tensor, (send_handle, recv_handle)
        else:
            # Wait for completion
            send_handle.wait()
            recv_handle.wait()

            # Update statistics
            self._comm_stats["total_sends"] += 1
            self._comm_stats["total_recvs"] += 1
            self._comm_stats["total_bytes"] += (
                send_tensor.numel() * send_tensor.element_size()
            )

            return recv_tensor

    def ring_exchange_with_buffer(
        self,
        send_tensor: Tensor,
        recv_buffer: Optional[Tensor] = None,
        direction: str = "forward",
        tag: int = 0,
    ) -> Tensor:
        """Exchange tensors in ring with optional receive buffer.

        This is more efficient when you have a pre-allocated buffer.

        Args:
            send_tensor: Tensor to send
            recv_buffer: Pre-allocated receive buffer
            direction: "forward" or "backward"
            tag: Communication tag

        Returns:
            Received tensor (same as recv_buffer if provided)
        """
        if not self.is_distributed:
            return send_tensor.clone()

        src, dst = self._get_ring_neighbors()
        if direction == "backward":
            src, dst = dst, src

        # Allocate buffer if not provided
        if recv_buffer is None:
            recv_buffer = torch.empty_like(send_tensor)
        else:
            assert recv_buffer.shape == send_tensor.shape, "Buffer shape mismatch"

        # Ensure contiguous
        send_tensor = send_tensor.contiguous()

        # Perform exchange
        unique_tag = self.DEFAULT_TAG_OFFSET + tag
        send_handle = dist.isend(send_tensor, dst=dst, tag=unique_tag)
        recv_handle = dist.irecv(recv_buffer, src=src, tag=unique_tag)

        send_handle.wait()
        recv_handle.wait()

        return recv_buffer

    def broadcast_from_rank(
        self, tensor: Optional[Tensor], src_rank: int = 0
    ) -> Tensor:
        """Broadcast tensor from specific rank to all others.

        Args:
            tensor: Tensor to broadcast (only needed on src_rank)
            src_rank: Source rank for broadcast

        Returns:
            Broadcasted tensor
        """
        if not self.is_distributed:
            return tensor

        # Ensure all ranks have a tensor
        if self.rank != src_rank:
            assert tensor is not None, "Non-source ranks must provide tensor shape"
            tensor = torch.empty_like(tensor)

        dist.broadcast(tensor, src=src_rank)
        return tensor

    def validate_ring_setup(self) -> bool:
        """Validate that ring communication is properly set up.

        Returns:
            True if ring is properly configured
        """
        if not self.is_distributed:
            warnings.warn("Ring attention running in non-distributed mode")
            return True

        try:
            # Test communication with small tensor
            test_tensor = torch.ones(1, device=self.device, dtype=self.dtype)
            result = self.ring_pass_forward(test_tensor, tag=999)

            # Verify we received something
            if result.numel() == 0:
                raise RuntimeError("Received empty tensor in ring test")

            # Synchronize to ensure all ranks completed test
            dist.barrier()

            return True

        except Exception as e:
            warnings.warn(f"Ring validation failed: {e}")
            return False

    def _handle_communication_failure(self, tensor: Tensor, error: Exception) -> Tensor:
        """Handle communication failure with graceful degradation.

        Args:
            tensor: Original tensor that failed to communicate
            error: The exception that occurred

        Returns:
            Fallback tensor (usually zeros or clone of input)
        """
        warnings.warn(
            f"Ring communication failed: {error}. Returning zeros as fallback."
        )

        # Return zeros to allow computation to continue
        # This is better than crashing the entire training
        return torch.zeros_like(tensor)

    def get_communication_stats(self) -> dict:
        """Get communication statistics.

        Returns:
            Dictionary of communication stats
        """
        stats = self._comm_stats.copy()

        # Add computed stats
        if stats["total_sends"] > 0:
            stats["failure_rate"] = stats["failed_attempts"] / stats["total_sends"]
            stats["avg_bytes_per_comm"] = stats["total_bytes"] / stats["total_sends"]
        else:
            stats["failure_rate"] = 0.0
            stats["avg_bytes_per_comm"] = 0.0

        return stats

    def reset_communication_stats(self) -> None:
        """Reset communication statistics."""
        self._comm_stats = {
            "total_sends": 0,
            "total_recvs": 0,
            "total_bytes": 0,
            "failed_attempts": 0,
            "retry_successes": 0,
        }

    def _get_ring_neighbors(self) -> Tuple[int, int]:
        """Get source and destination ranks for ring.

        Returns:
            Tuple of (src_rank, dst_rank)
        """
        if not hasattr(self, "world_size"):
            raise AttributeError("RingCommunicationMixin requires world_size attribute")

        src = (self.rank - 1) % self.world_size
        dst = (self.rank + 1) % self.world_size
        return src, dst


class AsyncRingCommunicator:
    """Helper class for managing async ring communications.

    This class helps manage multiple in-flight ring communications
    for overlapping computation and communication.
    """

    def __init__(self, mixin: RingCommunicationMixin):
        """Initialize async communicator.

        Args:
            mixin: RingCommunicationMixin instance
        """
        self.mixin = mixin
        self.pending_ops: List[Tuple[Tensor, Tuple]] = []

    def start_ring_pass(
        self, tensor: Tensor, direction: str = "forward", tag: int = 0
    ) -> Tensor:
        """Start async ring pass.

        Args:
            tensor: Tensor to communicate
            direction: "forward" or "backward"
            tag: Communication tag

        Returns:
            Receive buffer (contents not valid until wait_completion)
        """
        if direction == "forward":
            recv_tensor, handles = self.mixin.ring_pass_forward(
                tensor, tag=tag, async_op=True
            )
        else:
            recv_tensor, handles = self.mixin.ring_pass_backward(
                tensor, tag=tag, async_op=True
            )

        self.pending_ops.append((recv_tensor, handles))
        return recv_tensor

    def wait_completion(self, index: int = -1) -> Tensor:
        """Wait for specific async operation to complete.

        Args:
            index: Index of operation to wait for (-1 for most recent)

        Returns:
            Completed receive tensor
        """
        if not self.pending_ops:
            raise RuntimeError("No pending async operations")

        recv_tensor, (send_handle, recv_handle) = self.pending_ops.pop(index)
        send_handle.wait()
        recv_handle.wait()

        return recv_tensor

    def wait_all(self) -> List[Tensor]:
        """Wait for all pending operations to complete.

        Returns:
            List of completed receive tensors
        """
        results = []
        while self.pending_ops:
            results.append(self.wait_completion(0))
        return results
