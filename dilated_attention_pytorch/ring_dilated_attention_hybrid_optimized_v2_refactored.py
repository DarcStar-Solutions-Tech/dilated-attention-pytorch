"""
Refactored Ring Dilated Attention Hybrid Optimized V2.

This fixes the model recreation issue by properly implementing the single device
forward pass without creating temporary models.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

# Import base refactored implementation
from .ring_dilated_attention_refactored import RingDilatedAttentionRefactored

# Import ring utilities

# Import LSE utilities
from .ring_attention_lse import (
    compute_attention_with_lse,
)

# Import optimized LSE with backend fallbacks
try:
    from .ring_attention_lse_optimized import compute_attention_with_lse_optimized

    HAS_OPTIMIZED_LSE = True
except ImportError:
    HAS_OPTIMIZED_LSE = False
    compute_attention_with_lse_optimized = compute_attention_with_lse

# Import optimized attention computation
try:
    from .utils.attention_utils import optimize_attention_computation

    HAS_OPTIMIZE_ATTENTION = True
except ImportError:
    HAS_OPTIMIZE_ATTENTION = False


class RingDilatedAttentionHybridOptimizedV2Refactored(RingDilatedAttentionRefactored):
    """
    Refactored V2 implementation without model recreation.

    This maintains all the optimizations from V2 while fixing the
    fundamental issue of creating temporary models in the forward pass.
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # Optimization features
        enable_memory_pool: bool = True,
        enable_profiling: bool = False,
        use_pattern_cache: bool = True,
        use_flash_attention: bool = False,
    ):
        # Initialize base class
        super().__init__(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            enable_memory_pool=enable_memory_pool,
            use_pattern_cache=use_pattern_cache,
        )

        # V2 specific features
        self.enable_profiling = enable_profiling
        self.use_flash_attention = use_flash_attention

        # Pre-compute segment info
        self._setup_segment_info()

        print("RingDilatedAttentionHybridOptimizedV2Refactored: Initialized")

    def _setup_segment_info(self):
        """Pre-compute segment information for efficient processing."""
        self.total_segments = len(self.segment_lengths)
        self.segment_starts = [0]
        for seg_len in self.segment_lengths[:-1]:
            self.segment_starts.append(self.segment_starts[-1] + seg_len)

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Optimized single device forward without model recreation."""
        b, n, h, d = q.shape

        # Try to use Flash Attention if available and requested
        if self.use_flash_attention and HAS_OPTIMIZE_ATTENTION:
            try:
                # Transpose for Flash Attention format
                q_t = q.transpose(1, 2)  # (b, h, n, d)
                k_t = k.transpose(1, 2)
                v_t = v.transpose(1, 2)

                output_t = optimize_attention_computation(
                    q_t,
                    k_t,
                    v_t,
                    is_causal=is_causal,
                    dropout_p=self.dropout if self.training else 0.0,
                )

                return output_t.transpose(1, 2)  # Back to (b, n, h, d)
            except Exception:
                # Fall back to standard implementation
                pass

        # Use dilated attention implementation
        return self._compute_dilated_attention_single_device(q, k, v, is_causal)

    def _compute_dilated_attention_single_device(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Compute dilated attention on single device."""
        b, n, h, d = q.shape

        # Initialize output
        output = torch.zeros_like(q)

        # Process each segment
        for seg_idx in range(self.total_segments):
            seg_len = self.segment_lengths[seg_idx]
            seg_start = self.segment_starts[seg_idx]
            seg_end = seg_start + seg_len
            dilation = self.dilation_rates[seg_idx]

            if seg_start >= n:
                break

            # Adjust segment end if needed
            seg_end = min(seg_end, n)
            actual_seg_len = seg_end - seg_start

            # Get Q for this segment
            q_seg = q[:, seg_start:seg_end]

            # Compute dilated indices
            indices = self._compute_dilated_indices(
                seg_idx, seg_start, actual_seg_len, n, dilation
            )

            # Gather K and V
            k_dilated = self._gather_dilated_tensors(k, indices, b, h, d)
            v_dilated = self._gather_dilated_tensors(v, indices, b, h, d)

            # Compute attention
            output_seg = self._compute_segment_attention(
                q_seg, k_dilated, v_dilated, is_causal, seg_start, indices
            )

            output[:, seg_start:seg_end] = output_seg

        return output

    def _compute_dilated_indices(
        self,
        seg_idx: int,
        seg_start: int,
        seg_len: int,
        total_len: int,
        dilation: int,
    ) -> Tensor:
        """Compute dilated indices for a segment."""
        # Try cache first
        cache_key = (seg_idx, seg_start, seg_len, total_len, dilation)
        if self._pattern_cache is not None:
            cached = self._pattern_cache.get(cache_key)
            if cached is not None:
                return cached.to(self.device)

        # Create dilated pattern
        indices = []
        for i in range(seg_len):
            pos = seg_start + i

            # Get dilated positions
            dilated_positions = []

            # Look backward with dilation
            for j in range(dilation):
                idx = pos - j * seg_len
                while idx >= 0:
                    dilated_positions.append(idx)
                    idx -= dilation * seg_len

            # Look forward with dilation
            for j in range(1, dilation):
                idx = pos + j * seg_len
                while idx < total_len:
                    dilated_positions.append(idx)
                    idx += dilation * seg_len

            # Sort and truncate
            dilated_positions = sorted(dilated_positions)[:seg_len]

            # Pad if necessary
            while len(dilated_positions) < seg_len:
                dilated_positions.append(
                    dilated_positions[-1] if dilated_positions else pos
                )

            indices.append(torch.tensor(dilated_positions, device=self.device))

        indices = torch.stack(indices)

        # Cache result
        if self._pattern_cache is not None:
            self._pattern_cache.put(cache_key, indices.cpu())

        return indices

    def _gather_dilated_tensors(
        self,
        tensor: Tensor,
        indices: Tensor,
        b: int,
        h: int,
        d: int,
    ) -> Tensor:
        """Gather tensor values according to dilated indices."""
        seg_len, pattern_len = indices.shape

        # Expand indices for batch and head dimensions
        indices_expanded = indices.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        indices_expanded = indices_expanded.expand(b, -1, h, d, -1)

        # Reshape tensor for gathering
        tensor_reshaped = tensor.transpose(1, 2).unsqueeze(3)  # (b, h, n, 1, d)

        # Gather
        gathered = torch.gather(
            tensor_reshaped, 2, indices_expanded.transpose(3, 4).transpose(2, 3)
        )

        # Reshape back
        return gathered.squeeze(3).transpose(1, 2).transpose(2, 3)

    def _compute_segment_attention(
        self,
        q_seg: Tensor,
        k_pattern: Tensor,
        v_pattern: Tensor,
        is_causal: bool,
        seg_start: int,
        indices: Tensor,
    ) -> Tensor:
        """Compute attention for a segment."""
        b, seg_len, h, d = q_seg.shape

        # Reshape for attention computation
        q_seg = q_seg.transpose(1, 2)  # (b, h, seg_len, d)
        k_pattern = k_pattern.transpose(1, 2)  # (b, h, pattern_len, d)
        v_pattern = v_pattern.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q_seg, k_pattern.transpose(-2, -1)) / math.sqrt(d)

        # Apply causal mask if needed
        if is_causal:
            q_pos = torch.arange(
                seg_start, seg_start + seg_len, device=self.device
            ).unsqueeze(1)
            mask = q_pos < indices
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Dropout
        if self.dropout_layer is not None and self.training:
            attn_weights = self.dropout_layer(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v_pattern)

        # Transpose back
        return output.transpose(1, 2)  # (b, seg_len, h, d)

    def _compute_dilated_chunk_attention(
        self,
        q: Tensor,
        k_chunk: Tensor,
        v_chunk: Tensor,
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute dilated attention for a chunk in ring mode."""
        b, n, h, d = q.shape

        # Process each segment
        outputs = []
        lses = []

        for seg_idx in range(self.total_segments):
            seg_len = self.segment_lengths[seg_idx]
            seg_start = self.segment_starts[seg_idx]
            dilation = self.dilation_rates[seg_idx]

            if seg_start >= n:
                break

            # Get relevant Q positions for this chunk
            q_seg_start = max(seg_start, chunk_start)
            q_seg_end = min(seg_start + seg_len, n)

            if q_seg_start >= q_seg_end:
                continue

            q_seg = q[:, q_seg_start:q_seg_end]

            # Compute which K/V positions from chunk are relevant
            # This is complex due to dilation patterns
            relevant_k, relevant_v = self._get_relevant_chunk_positions(
                k_chunk,
                v_chunk,
                seg_idx,
                q_seg_start,
                q_seg_end,
                chunk_start,
                chunk_size,
                dilation,
            )

            if relevant_k.shape[1] == 0:
                continue

            # Compute attention for this segment
            if HAS_OPTIMIZED_LSE:
                seg_output, seg_lse = compute_attention_with_lse_optimized(
                    q_seg.transpose(1, 2),  # (b, h, seg_len, d)
                    relevant_k.transpose(1, 2),
                    relevant_v.transpose(1, 2),
                    is_causal=is_causal,
                    chunk_idx=chunk_start // chunk_size,
                    total_chunks=self.ring_size,
                )
            else:
                seg_output, seg_lse = compute_attention_with_lse(
                    q_seg.transpose(1, 2),
                    relevant_k.transpose(1, 2),
                    relevant_v.transpose(1, 2),
                    is_causal=is_causal,
                    chunk_idx=chunk_start // chunk_size,
                    total_chunks=self.ring_size,
                )

            outputs.append((q_seg_start, q_seg_end, seg_output))
            lses.append((q_seg_start, q_seg_end, seg_lse))

        # Combine outputs
        combined_output = torch.zeros(b, h, n, d, device=q.device, dtype=q.dtype)
        combined_lse = torch.full(
            (b, h, n), float("-inf"), device=q.device, dtype=q.dtype
        )

        for start, end, output in outputs:
            combined_output[:, :, start:end] = output

        for start, end, lse in lses:
            combined_lse[:, :, start:end] = lse

        return combined_output, combined_lse

    def _get_relevant_chunk_positions(
        self,
        k_chunk: Tensor,
        v_chunk: Tensor,
        seg_idx: int,
        q_start: int,
        q_end: int,
        chunk_start: int,
        chunk_size: int,
        dilation: int,
    ) -> Tuple[Tensor, Tensor]:
        """Get relevant K/V positions from chunk for dilated attention."""
        # This is a simplified version - in practice you'd need to
        # carefully compute which positions in the chunk are accessed
        # by the dilated pattern for the given Q positions

        # For now, return the full chunk
        # TODO: Implement proper dilated pattern filtering
        return k_chunk, v_chunk
