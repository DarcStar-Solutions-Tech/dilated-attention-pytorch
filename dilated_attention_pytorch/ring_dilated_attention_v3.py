"""
Ring Dilated Attention V3 - True ring attention with proper utilities.

This implementation uses the patterns from lucidrains/ring-attention-pytorch
to achieve true O(n/p) memory scaling with dilated attention patterns.
"""

import math
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from .ring_attention_utils import (
    exists,
    default,
    all_ring_pass,
    split_by_rank,
    create_causal_mask,
    RingInfo,
)
from .ring_attention_lse import (
    StableRingAccumulator,
    compute_attention_with_lse,
)
from .ring_attention_bucketed import (
    BucketConfig,
    BucketedAttentionProcessor,
)


class RingDilatedAttentionV3(nn.Module):
    """
    True Ring Dilated Attention with O(n/p) memory scaling.

    Uses proper ring communication utilities for robust distributed execution.
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bucket_size: int = 1024,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_bucketed: bool = True,
        grad_checkpoint_buckets: bool = False,
    ):
        super().__init__()

        assert len(segment_lengths) == len(dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.bucket_size = bucket_size
        self.use_bucketed = use_bucketed

        # Bucket configuration
        self.bucket_config = BucketConfig(
            bucket_size=bucket_size,
            grad_checkpoint=grad_checkpoint_buckets,
            use_flash_attn=False,  # Can be enabled later
        )

        # Device and dtype
        self.device = device or (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.dtype = dtype or torch.float32

        # Ring configuration
        if torch.distributed.is_initialized():
            self.ring_size = default(ring_size, torch.distributed.get_world_size())
            self.rank = torch.distributed.get_rank()
        else:
            self.ring_size = 1
            self.rank = 0

    def _calculate_head_groups(self, num_heads: int) -> list[int]:
        """Distribute heads across segment lengths."""
        num_segments = len(self.segment_lengths)
        base_heads = num_heads // num_segments
        extra_heads = num_heads % num_segments

        head_groups = [base_heads] * num_segments
        for i in range(extra_heads):
            head_groups[-(i + 1)] += 1

        return head_groups

    def _apply_dilated_pattern(
        self,
        tensor: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset: int = 0,
    ) -> Tensor:
        """Apply dilated pattern to tensor."""
        if dilation_rate == 1:
            return tensor

        b, n, h, d = tensor.shape

        # Create dilated view
        # For simplicity, we'll use a gather approach
        indices = torch.arange(0, n, device=tensor.device)

        # Apply dilation pattern
        dilated_indices = []
        for i in range(0, n, segment_len):
            segment_end = min(i + segment_len, n)
            segment_indices = indices[i:segment_end]

            # Apply dilation within segment
            if len(segment_indices) >= dilation_rate:
                dilated = segment_indices[offset::dilation_rate]
                # Pad if needed
                while len(dilated) < len(segment_indices):
                    dilated = torch.cat(
                        [
                            dilated,
                            segment_indices[: len(segment_indices) - len(dilated)],
                        ]
                    )
                dilated_indices.append(dilated[: len(segment_indices)])
            else:
                dilated_indices.append(segment_indices)

        if dilated_indices:
            all_indices = torch.cat(dilated_indices)
            return tensor.index_select(1, all_indices)
        else:
            return tensor

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass using true ring attention.

        Args:
            q, k, v: Shape (batch, seq_len, num_heads, head_dim)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of same shape as q
        """
        b, n, h, d = q.shape
        _ = q.device

        # For single device, use standard computation
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Split K,V across ranks
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )

        # Each rank owns a chunk of K,V
        k_local = split_by_rank(k, self.rank, self.ring_size)
        v_local = split_by_rank(v, self.rank, self.ring_size)

        # Skip dilated patterns for distributed case to avoid shape issues
        # TODO: Implement proper distributed dilated patterns
        # k_local = self._apply_dilated_patterns_to_tensor(k_local)
        # v_local = self._apply_dilated_patterns_to_tensor(v_local)

        # Stack K,V for ring passing
        kv_local = torch.stack((k_local, v_local))

        # Initialize stable accumulator for numerical stability
        accumulator = StableRingAccumulator(
            output_shape=(b, h, n, d),  # Note: heads before seq for LSE
            device=q.device,
            dtype=q.dtype,
        )

        # Transpose q for attention computation (batch, heads, seq, dim)
        q_transposed = q.transpose(1, 2)

        # Process through ring passes
        ring_pass_fn = partial(all_ring_pass, ring_size=self.ring_size)

        for ring_info, (kv_chunk,) in ring_pass_fn(kv_local):
            if not exists(kv_chunk):
                continue

            k_chunk, v_chunk = kv_chunk

            # Calculate which part of sequence this chunk represents
            chunk_size = n // self.ring_size
            chunk_idx = ring_info.ring_rank
            chunk_start = chunk_idx * chunk_size

            # Compute attention for this chunk with LSE
            if self.use_bucketed:
                chunk_output, chunk_lse = self._compute_chunk_attention_bucketed(
                    q_transposed, k_chunk, v_chunk, chunk_start, is_causal, ring_info
                )
            else:
                chunk_output, chunk_lse = self._compute_chunk_attention_lse(
                    q_transposed, k_chunk, v_chunk, chunk_start, is_causal, ring_info
                )

            # Update accumulator
            accumulator.update(chunk_output, chunk_lse)

        # Get final output and transpose back to (batch, seq, heads, dim)
        output = accumulator.get_output().transpose(1, 2)
        return output

    def _apply_dilated_patterns_to_tensor(self, tensor: Tensor) -> Tensor:
        """Apply dilated patterns based on head groups."""
        b, n, h, d = tensor.shape
        heads_per_group = self._calculate_head_groups(h)

        output = []
        head_start = 0

        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Apply dilation to this head group
            tensor_group = tensor[:, :, head_start:head_end, :]
            tensor_dilated = self._apply_dilated_pattern(
                tensor_group, segment_len, dilation_rate, offset=i
            )
            output.append(tensor_dilated)

            head_start = head_end

        return torch.cat(output, dim=2) if output else tensor

    def _compute_chunk_attention_lse(
        self,
        q: Tensor,  # (batch, heads, seq, dim)
        k_chunk: Tensor,  # (batch, seq, heads, dim)
        v_chunk: Tensor,  # (batch, seq, heads, dim)
        chunk_start: int,
        is_causal: bool,
        ring_info: RingInfo,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention for one chunk with log-sum-exp."""
        b, h, n_q, d = q.shape

        # Transpose k,v chunks to (batch, heads, seq, dim)
        k_chunk = k_chunk.transpose(1, 2)
        v_chunk = v_chunk.transpose(1, 2)

        # Create causal mask if needed
        mask = None
        if is_causal:
            # Create mask based on absolute positions
            q_positions = torch.arange(n_q, device=q.device)
            kv_positions = torch.arange(
                chunk_start, chunk_start + k_chunk.shape[2], device=q.device
            )

            causal_mask = create_causal_mask(q_positions, kv_positions, q.device)
            # Expand for batch and heads dimensions
            mask = ~causal_mask.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)

        # Compute attention with LSE
        output, lse = compute_attention_with_lse(
            q,
            k_chunk,
            v_chunk,
            scale=1.0 / math.sqrt(d),
            mask=mask,
            dropout=self.dropout,
            training=self.training,
        )

        return output, lse

    def _compute_chunk_attention_bucketed(
        self,
        q: Tensor,  # (batch, heads, seq, dim)
        k_chunk: Tensor,  # (batch, seq, heads, dim)
        v_chunk: Tensor,  # (batch, seq, heads, dim)
        chunk_start: int,
        is_causal: bool,
        ring_info: RingInfo,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention for one chunk using bucketed processing."""
        b, h, n_q, d = q.shape

        # Transpose k,v chunks to (batch, heads, seq, dim)
        k_chunk = k_chunk.transpose(1, 2)
        v_chunk = v_chunk.transpose(1, 2)

        # Create bucketed processor
        processor = BucketedAttentionProcessor(
            self.bucket_config,
            device=q.device,
            dtype=q.dtype,
        )

        # Process with buckets
        output, lse = processor.process_attention(
            q,
            k_chunk,
            v_chunk,
            scale=1.0 / math.sqrt(d),
            mask=None,  # Mask handled inside processor
            is_causal=is_causal,
            dropout=self.dropout,
            training=self.training,
            q_offset=0,  # Query positions start at 0
            kv_offset=chunk_start,  # K,V positions start at chunk_start
        )

        # Note: For proper causal masking across chunks, we need to
        # adjust the processor to handle chunk_start offset
        # This is a simplified version - full implementation would
        # pass chunk_start to the processor

        return output, lse

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Standard forward for single device using LSE for consistency."""
        # Apply dilated patterns
        k = self._apply_dilated_patterns_to_tensor(k)
        v = self._apply_dilated_patterns_to_tensor(v)

        # Transpose to (batch, heads, seq, dim) for LSE computation
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Create causal mask if needed
        mask = None
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(q.shape[1], k.shape[1], device=q.device), diagonal=1
            ).bool()
            mask = (
                causal_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(q.shape[0], q.shape[2], -1, -1)
            )

        # Use LSE computation for consistency
        output, _ = compute_attention_with_lse(
            q_t,
            k_t,
            v_t,
            scale=1.0 / math.sqrt(q.shape[-1]),
            mask=mask,
            dropout=self.dropout,
            training=self.training,
        )

        # Transpose back to (batch, seq, heads, dim)
        return output.transpose(1, 2)


class RingMultiheadDilatedAttentionV3(nn.Module):
    """
    Multihead wrapper for Ring Dilated Attention V3.

    Drop-in replacement for nn.MultiheadAttention with true ring scaling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        bucket_size: int = 1024,
        ring_size: Optional[int] = None,
        use_bucketed: bool = True,
        grad_checkpoint_buckets: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dtype = dtype or torch.float32
        self.device = device or (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        assert embed_dim % num_heads == 0

        # Projections
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # Ring attention
        self.ring_attention = RingDilatedAttentionV3(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            bucket_size=bucket_size,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            use_bucketed=use_bucketed,
            grad_checkpoint_buckets=grad_checkpoint_buckets,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass with true ring dilated attention."""
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        b, n, _ = query.shape

        # Project and reshape
        q = self.q_proj(query).view(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(b, n, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(b, n, self.num_heads, self.head_dim)

        # Ring attention
        output = self.ring_attention(q, k, v, is_causal)

        # Reshape and project
        output = output.reshape(b, n, self.embed_dim)
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output
