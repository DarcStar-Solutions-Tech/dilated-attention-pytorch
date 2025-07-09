"""Standard ring attention implementation using the base class.

This is the reference implementation that other ring variants can build upon.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from .base_ring_attention import BaseRingAttention, RingAttentionState
from .ring_communication_mixin import RingCommunicationMixin
from .ring_config import RingAttentionConfig
from ...utils.attention_utils import create_causal_mask


class StandardRingAttention(BaseRingAttention, RingCommunicationMixin):
    """Standard ring attention implementation.

    This class provides a clean, efficient implementation of ring attention
    that serves as the reference for other variants. It uses isend/irecv
    for true O(n/k) memory complexity.
    """

    def __init__(
        self,
        config: RingAttentionConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize standard ring attention.

        Args:
            config: Ring attention configuration
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        # Initialize base class
        BaseRingAttention.__init__(self, config, device, dtype)

        # Initialize communication mixin
        RingCommunicationMixin.__init__(self)

        # Store config
        self.config = config

        # Validate setup if distributed
        if self.is_distributed and config.validate_gradients:
            if not self.validate_ring_setup():
                raise RuntimeError("Ring setup validation failed")

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
        if already_split or not self.is_distributed:
            # Sequence already split or not in distributed mode
            seq_len = x.shape[1]
            return x, 0, seq_len

        batch_size, seq_len = x.shape[:2]

        # Calculate local sequence length
        assert seq_len % self.world_size == 0, (
            f"Sequence length {seq_len} must be divisible by world size {self.world_size}"
        )

        local_seq_len = seq_len // self.world_size
        start_idx = self.rank * local_seq_len
        end_idx = start_idx + local_seq_len

        # Extract local chunk
        local_chunk = x[:, start_idx:end_idx].contiguous()

        return local_chunk, start_idx, end_idx

    def _ring_communication(
        self, tensor: Tensor, direction: str = "forward", tag: int = 0
    ) -> Tensor:
        """Perform ring communication using isend/irecv.

        Args:
            tensor: Tensor to communicate
            direction: "forward" or "backward"
            tag: Communication tag

        Returns:
            Received tensor from neighbor
        """
        if direction == "forward":
            return self.ring_pass_forward(tensor, tag=tag)
        else:
            return self.ring_pass_backward(tensor, tag=tag)

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
            q: Query tensor of shape (batch, q_len, num_heads, head_dim)
            k: Key tensor of shape (batch, kv_len, num_heads, head_dim)
            v: Value tensor of shape (batch, kv_len, num_heads, head_dim)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            segment_idx: Current segment index

        Returns:
            Tuple of (attention_output, lse) where lse is log-sum-exp
        """
        batch_size, q_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]

        # Reshape for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, q_len, head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, kv_len, head_dim)
        v = v.transpose(1, 2)  # (batch, num_heads, kv_len, head_dim)

        # Compute attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply causal mask if needed
        if is_causal:
            causal_mask = create_causal_mask(
                q_len, kv_len, device=self.device, dtype=scores.dtype
            )
            scores = scores + causal_mask

        # Compute log-sum-exp for numerical stability
        lse = torch.logsumexp(scores, dim=-1)  # (batch, num_heads, q_len)

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout if configured
        if self.dropout_p > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        # Compute attention output
        attn_output = torch.matmul(
            attn_weights, v
        )  # (batch, num_heads, q_len, head_dim)

        # Transpose back
        attn_output = attn_output.transpose(1, 2)  # (batch, q_len, num_heads, head_dim)

        return attn_output, lse

    def _accumulate_results(
        self,
        local_out: Tensor,
        local_lse: Tensor,
        remote_out: Tensor,
        remote_lse: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Accumulate results from ring passes using log-sum-exp.

        Args:
            local_out: Local attention output (batch, seq, heads, dim)
            local_lse: Local log-sum-exp (batch, heads, seq)
            remote_out: Remote attention output
            remote_lse: Remote log-sum-exp

        Returns:
            Tuple of (accumulated_output, accumulated_lse)
        """
        # Ensure correct shapes
        # local_lse shape: (batch, heads, seq)
        # local_out shape: (batch, seq, heads, dim)

        # Compute stable accumulation using log-sum-exp trick
        max_lse = torch.maximum(local_lse, remote_lse)

        # Compute weights
        local_weight = torch.exp(local_lse - max_lse)
        remote_weight = torch.exp(remote_lse - max_lse)

        # Reshape weights for broadcasting
        # Need to match output shape: (batch, seq, heads, dim)
        local_weight = local_weight.transpose(1, 2).unsqueeze(-1)
        remote_weight = remote_weight.transpose(1, 2).unsqueeze(-1)

        # Weighted combination
        accumulated_out = (local_out * local_weight + remote_out * remote_weight) / (
            local_weight + remote_weight
        )

        # Update LSE
        accumulated_lse = max_lse + torch.log(
            torch.exp(local_lse - max_lse) + torch.exp(remote_lse - max_lse)
        )

        return accumulated_out, accumulated_lse

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        already_split: bool = False,
    ) -> Tensor:
        """Forward pass of standard ring attention.

        Args:
            query: Query tensor (batch, seq_len, num_heads, head_dim)
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            already_split: Whether inputs are already split

        Returns:
            Attention output
        """
        # Input validation
        batch_size, seq_len, num_heads, head_dim = query.shape
        self._validate_sequence_length(seq_len)

        # Split sequences for local processing
        q_local, q_start, q_end = self._split_sequence(query, already_split)
        k_local, k_start, k_end = self._split_sequence(key, already_split)
        v_local, v_start, v_end = self._split_sequence(value, already_split)

        local_seq_len = q_local.shape[1]

        # Initialize state
        state = RingAttentionState(
            batch_size=batch_size,
            local_seq_len=local_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Current K and V start as local chunks
        current_k = k_local
        current_v = v_local

        # Perform ring passes
        for ring_step in range(self.world_size):
            # Compute local attention
            local_out, local_lse = self._local_attention(
                q_local,
                current_k,
                current_v,
                attention_mask=attention_mask,
                is_causal=is_causal,
                segment_idx=0,  # TODO: Handle segment index properly
            )

            # Accumulate results
            if ring_step == 0:
                state.output = local_out
                state.lse = local_lse
            else:
                state.output, state.lse = self._accumulate_results(
                    state.output, state.lse, local_out, local_lse
                )

            # Ring communication for next iteration (except last)
            if ring_step < self.world_size - 1:
                current_k = self._ring_communication(
                    current_k, direction="forward", tag=ring_step * 2
                )
                current_v = self._ring_communication(
                    current_v, direction="forward", tag=ring_step * 2 + 1
                )

        # Synchronize before returning
        if self.is_distributed:
            self._synchronize_ring()

        # Log stats if enabled
        if self.config.log_communication_stats:
            stats = self.get_communication_stats()
            print(f"Rank {self.rank} communication stats: {stats}")

        return state.output

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        base_repr = super().extra_repr()
        return f"{base_repr}, backend={self.config.communication_backend}"
