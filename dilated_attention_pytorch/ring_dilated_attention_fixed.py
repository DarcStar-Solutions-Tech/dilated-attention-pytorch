"""
Fixed Ring Dilated Attention - Simple working implementation without model recreation.

This is a minimal implementation that works correctly without the complex
optimizations that are causing issues.
"""

import math
from typing import Optional, List
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

# Import ring utilities
from .ring_attention_utils import (
    exists,
    all_ring_pass,
    split_by_rank,
)

# Import LSE utilities
from .ring_attention_lse import (
    StableRingAccumulator,
)


class RingDilatedAttentionFixed(nn.Module):
    """
    Fixed Ring Dilated Attention without model recreation.

    This is a clean, working implementation that avoids the issues
    in the original codebase.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Configuration
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout

        # Device and dtype
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.dtype = dtype or torch.float32

        # Ring configuration
        if dist.is_initialized():
            self.ring_size = ring_size or dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.ring_size = ring_size or 1
            self.rank = 0

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # Pre-allocate buffers
        self._kv_receive_buffer = None

        # Simple pattern cache (just a dict)
        self._pattern_cache = {}

        print("RingDilatedAttentionFixed: Initialized")

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass."""
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)
        else:
            return self._multi_device_forward(q, k, v, is_causal)

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward - simple dilated attention."""
        b, n, h, d = q.shape

        # For simplicity, just do standard attention with dilated patterns
        # Process each segment
        output = torch.zeros_like(q)

        for seg_idx, (seg_len, dilation) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            seg_start = sum(self.segment_lengths[:seg_idx])
            seg_end = seg_start + seg_len

            if seg_start >= n:
                break

            seg_end = min(seg_end, n)

            # Get Q segment
            q_seg = q[:, seg_start:seg_end]

            # For dilated attention, we sample K and V with dilation
            # Simple approach: use strided sampling
            if dilation > 1:
                # Sample every 'dilation' positions
                k_indices = torch.arange(0, n, dilation, device=self.device)
                k_indices = k_indices[k_indices < n]

                k_dilated = k[:, k_indices]
                v_dilated = v[:, k_indices]
            else:
                k_dilated = k
                v_dilated = v

            # Compute attention
            output_seg = self._compute_attention(
                q_seg, k_dilated, v_dilated, is_causal, seg_start
            )

            output[:, seg_start:seg_end] = output_seg

        return output

    def _multi_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Multi-GPU forward with ring communication."""
        b, n, h, d = q.shape

        # Ensure divisibility
        assert n % self.ring_size == 0

        chunk_size = n // self.ring_size

        # Split K,V across GPUs
        k_local = split_by_rank(k, self.rank, self.ring_size)
        v_local = split_by_rank(v, self.rank, self.ring_size)

        # Stack for ring passing
        kv_local = torch.stack((k_local, v_local))

        # Pre-allocate receive buffer
        if (
            self._kv_receive_buffer is None
            or self._kv_receive_buffer.shape != kv_local.shape
        ):
            self._kv_receive_buffer = torch.empty_like(kv_local)

        # Initialize accumulator
        accumulator = StableRingAccumulator(
            output_shape=(b, h, n, d),
            device=q.device,
            dtype=q.dtype,
        )

        # Ring attention
        ring_pass_fn = partial(
            all_ring_pass,
            receive_buffer=self._kv_receive_buffer,
            ring_size=self.ring_size,
        )

        for ring_info, (kv_chunk,) in ring_pass_fn(kv_local):
            if not exists(kv_chunk):
                continue

            k_chunk, v_chunk = kv_chunk

            # Simple attention computation
            # Transpose to (b, h, n, d) format
            q_t = q.transpose(1, 2)
            k_chunk_t = k_chunk.transpose(1, 2)
            v_chunk_t = v_chunk.transpose(1, 2)

            # Compute scores
            scores = torch.matmul(q_t, k_chunk_t.transpose(-2, -1)) / math.sqrt(d)

            # Apply causal mask if needed
            if is_causal:
                # Simple causal mask
                mask = torch.ones(n, chunk_size, device=self.device, dtype=torch.bool)
                chunk_start = ring_info.ring_rank * chunk_size
                for i in range(n):
                    for j in range(chunk_size):
                        if i < chunk_start + j:
                            mask[i, j] = False
                scores = scores.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Get max for numerical stability
            max_scores = scores.amax(dim=-1, keepdim=True)
            max_scores = torch.where(
                torch.isfinite(max_scores), max_scores, torch.zeros_like(max_scores)
            )

            # Exp scores
            exp_scores = torch.exp(scores - max_scores)

            # Sum of exp scores (for denominator)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)

            # Weighted values
            weighted_values = torch.matmul(exp_scores, v_chunk_t)

            # LSE for this chunk
            lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

            # Update accumulator
            accumulator.update(weighted_values, lse)

        # Get final output
        output = accumulator.get_output().transpose(1, 2)

        return output

    def _compute_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
        q_offset: int = 0,
    ) -> Tensor:
        """Compute standard attention."""
        b, q_len, h, d = q.shape
        _, k_len, _, _ = k.shape

        # Transpose to (b, h, len, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        # Apply causal mask if needed
        if is_causal:
            mask = torch.ones(q_len, k_len, device=self.device, dtype=torch.bool)
            for i in range(q_len):
                for j in range(k_len):
                    if q_offset + i < j:
                        mask[i, j] = False
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Dropout
        if self.dropout_layer is not None and self.training:
            attn_weights = self.dropout_layer(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Transpose back
        return output.transpose(1, 2)
