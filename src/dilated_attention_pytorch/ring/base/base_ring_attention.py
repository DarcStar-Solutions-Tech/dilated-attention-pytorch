"""Abstract base class for all ring attention implementations.

This module provides the foundation for ring attention variants that achieve
O(n/k) memory complexity where n is sequence length and k is world size.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import warnings

import torch
import torch.distributed as dist
from torch import Tensor

from ...core.base import BaseDilatedAttention
from ...core.config import DilatedAttentionConfig


class BaseRingAttention(BaseDilatedAttention, ABC):
    """Abstract base class for all ring attention implementations.

    This class defines the interface and common functionality for ring attention
    variants that distribute sequence processing across multiple GPUs/processes
    to achieve O(n/k) memory complexity.

    Key Design Principles:
    1. Sequences are split BEFORE QKV projection to maintain O(n/k) memory
    2. Only isend/irecv operations are used for true ring communication
    3. Results are accumulated using log-sum-exp for numerical stability
    4. Supports both forward and backward gradient flow
    """

    def __init__(
        self,
        config: DilatedAttentionConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize base ring attention.

        Args:
            config: Dilated attention configuration
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        # Update config with device and dtype if provided
        if device is not None:
            config.device = device
        if dtype is not None:
            config.dtype = dtype

        super().__init__(config)

        # Ring-specific attributes
        self.world_size = 1
        self.rank = 0
        self.ring_size = 1
        self.is_distributed = False

        # Communication buffers (allocated on demand)
        self._send_buffer: Optional[Tensor] = None
        self._recv_buffer: Optional[Tensor] = None

        # Initialize distributed environment if available
        self._init_distributed()

    def _init_distributed(self) -> None:
        """Initialize distributed environment and ring parameters."""
        if dist.is_available() and dist.is_initialized():
            self.is_distributed = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.ring_size = self.world_size

            # Validate ring setup
            try:
                self._validate_ring_setup()
            except ValueError as e:
                warnings.warn(f"Ring setup validation failed: {e}")
                self.is_distributed = False
                self.world_size = 1
                self.rank = 0
                self.ring_size = 1

    @abstractmethod
    def _split_sequence(
        self, x: Tensor, already_split: bool = False
    ) -> Tuple[Tensor, int, int]:
        """Split sequence for local processing.

        Args:
            x: Input tensor of shape (batch, seq_len, ...)
            already_split: Whether sequence is already split

        Returns:
            Tuple of (local_chunk, start_idx, end_idx)
        """
        pass

    @abstractmethod
    def _ring_communication(
        self, tensor: Tensor, direction: str = "forward", tag: int = 0
    ) -> Tensor:
        """Perform ring communication using isend/irecv.

        Args:
            tensor: Tensor to communicate
            direction: "forward" (rank -> rank+1) or "backward" (rank -> rank-1)
            tag: Communication tag for MPI

        Returns:
            Received tensor from neighbor
        """
        pass

    @abstractmethod
    def _local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        segment_idx: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention on local chunk.

        Args:
            q: Query tensor (local chunk)
            k: Key tensor (current ring position)
            v: Value tensor (current ring position)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            segment_idx: Current segment index for dilated attention

        Returns:
            Tuple of (attention_output, lse) where lse is log-sum-exp for stability
        """
        pass

    @abstractmethod
    def _accumulate_results(
        self,
        local_out: Tensor,
        local_lse: Tensor,
        remote_out: Tensor,
        remote_lse: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Accumulate results from ring passes using log-sum-exp.

        Args:
            local_out: Local attention output
            local_lse: Local log-sum-exp
            remote_out: Remote attention output
            remote_lse: Remote log-sum-exp

        Returns:
            Tuple of (accumulated_output, accumulated_lse)
        """
        pass

    def _get_communication_buffer(
        self, shape: Tuple[int, ...], buffer_type: str = "send"
    ) -> Tensor:
        """Get or allocate communication buffer.

        Args:
            shape: Shape of buffer needed
            buffer_type: "send" or "recv"

        Returns:
            Buffer tensor
        """
        buffer_attr = f"_{buffer_type}_buffer"
        buffer = getattr(self, buffer_attr)

        if buffer is None or buffer.shape != shape:
            # Allocate new buffer
            buffer = torch.empty(shape, device=self.device, dtype=self.dtype)
            setattr(self, buffer_attr, buffer)

        return buffer

    def _validate_ring_setup(self) -> None:
        """Validate ring setup for distributed training.

        Raises:
            ValueError: If ring setup is invalid
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized but ring_size > 1")

        if self.world_size < 2:
            raise ValueError(
                f"Ring attention requires world_size >= 2, got {self.world_size}"
            )

        if self.rank >= self.world_size:
            raise ValueError(f"Rank {self.rank} >= world_size {self.world_size}")

    def _validate_sequence_length(self, seq_len: int) -> None:
        """Validate sequence length for ring attention.

        Args:
            seq_len: Sequence length to validate

        Raises:
            ValueError: If sequence length is invalid
        """
        if self.is_distributed and seq_len % self.world_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by world size {self.world_size}"
            )

        # Also validate against segment lengths
        max_segment = max(self.segment_lengths)
        if seq_len % max_segment != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by largest segment {max_segment}"
            )

    def _get_ring_neighbors(self) -> Tuple[int, int]:
        """Get source and destination ranks for ring communication.

        Returns:
            Tuple of (source_rank, dest_rank)
        """
        if not self.is_distributed:
            return 0, 0

        src = (self.rank - 1) % self.world_size
        dst = (self.rank + 1) % self.world_size
        return src, dst

    def _synchronize_ring(self) -> None:
        """Synchronize all processes in the ring."""
        if self.is_distributed:
            dist.barrier()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        already_split: bool = False,
    ) -> Tensor:
        """Forward pass of ring attention.

        This method should be overridden by subclasses but can call super()
        for common validation and setup.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            already_split: Whether inputs are already split for ring

        Returns:
            Attention output
        """
        # Validate inputs
        batch_size, seq_len = query.shape[:2]
        self._validate_sequence_length(seq_len)

        # Let subclasses implement the actual forward logic
        raise NotImplementedError("Subclasses must implement forward()")

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        base_repr = super().extra_repr()
        ring_info = f", ring_size={self.ring_size}, rank={self.rank}"
        return base_repr + ring_info


class RingAttentionState:
    """Container for ring attention state during forward pass.

    This class helps manage the complex state needed during ring attention
    computation, making the code cleaner and easier to understand.
    """

    def __init__(
        self,
        batch_size: int,
        local_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize ring attention state."""
        self.batch_size = batch_size
        self.local_seq_len = local_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Accumulation tensors
        self.output = torch.zeros(
            (batch_size, local_seq_len, num_heads, head_dim), device=device, dtype=dtype
        )
        self.lse = torch.full(
            (batch_size, num_heads, local_seq_len),
            fill_value=-float("inf"),
            device=device,
            dtype=dtype,
        )

        # Current ring position tensors (will be set during computation)
        self.current_k: Optional[Tensor] = None
        self.current_v: Optional[Tensor] = None

    def update(self, new_output: Tensor, new_lse: Tensor) -> None:
        """Update accumulated state with new results."""
        # Stable accumulation using log-sum-exp
        max_lse = torch.maximum(self.lse, new_lse)

        self.output = self.output * torch.exp(
            self.lse.unsqueeze(-1) - max_lse.unsqueeze(-1)
        ) + new_output * torch.exp(new_lse.unsqueeze(-1) - max_lse.unsqueeze(-1))

        self.lse = max_lse + torch.log(
            torch.exp(self.lse - max_lse) + torch.exp(new_lse - max_lse)
        )
