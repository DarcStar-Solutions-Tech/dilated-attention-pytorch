"""
Fixed RingDilatedAttentionHilbertOptimized with standardized API.

This implementation fixes the import issues and provides the standardized API.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict

from .core.standardized_api import (
    StandardizedRingConfig,
    StandardizedRingAttentionMixin,
)


class RingDilatedAttentionHilbertOptimizedFixed(
    nn.Module, StandardizedRingAttentionMixin
):
    """
    Fixed version of RingDilatedAttentionHilbertOptimized with standardized API.

    This implementation includes Hilbert curve optimizations for better cache locality
    while fixing the import and API issues.
    """

    def __init__(
        self,
        # Accept both StandardizedRingConfig and individual parameters
        config: Optional[StandardizedRingConfig] = None,
        # Individual parameters for backward compatibility
        dim: Optional[int] = None,
        heads: Optional[int] = None,
        segment_lengths: Optional[List[int]] = None,
        dilation_rates: Optional[List[int]] = None,
        ring_size: Optional[int] = None,
        dropout: float = 0.0,
        use_hilbert: bool = True,
        hilbert_chunk_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Handle different initialization patterns
        if config is not None:
            # StandardizedRingConfig provided
            self.dim = config.dim
            self.heads = config.heads
            self.segment_lengths = config.segment_lengths
            self.dilation_rates = config.dilation_rates
            self.ring_size = config.ring_size or 1
            self.dropout_p = config.dropout
            self.use_hilbert = config.use_hilbert or use_hilbert
            self.hilbert_chunk_size = config.hilbert_chunk_size or hilbert_chunk_size
        else:
            # Individual parameters provided
            if segment_lengths is None or dilation_rates is None:
                raise ValueError("segment_lengths and dilation_rates are required")

            self.dim = dim or kwargs.get("head_dim", 64)
            self.heads = heads or kwargs.get("num_heads", 8)
            self.segment_lengths = segment_lengths
            self.dilation_rates = dilation_rates
            self.ring_size = ring_size or 1
            self.dropout_p = dropout
            self.use_hilbert = use_hilbert
            self.hilbert_chunk_size = hilbert_chunk_size

        # Validate inputs
        assert len(self.segment_lengths) == len(self.dilation_rates), (
            "segment_lengths and dilation_rates must have same length"
        )

        # Initialize components
        self.scale = self.dim**-0.5
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

        # Set default Hilbert chunk size
        if self.hilbert_chunk_size is None:
            self.hilbert_chunk_size = max(self.segment_lengths)

        # Cache for Hilbert mappings
        self._hilbert_cache: Dict[int, torch.Tensor] = {}
        self._inverse_hilbert_cache: Dict[int, torch.Tensor] = {}

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_hilbert_indices(self, n: int) -> torch.Tensor:
        """Generate Hilbert curve indices for sequence length n."""
        if n in self._hilbert_cache:
            return self._hilbert_cache[n]

        # Find the smallest power of 2 >= sqrt(n)
        dim = 1
        while dim * dim < n:
            dim *= 2

        # Generate 2D Hilbert curve
        indices = []
        for i in range(min(n, dim * dim)):
            x, y = self._hilbert_index_to_xy(i, dim)
            if x * dim + y < n:
                indices.append(x * dim + y)

        # Handle remaining indices if n is not a perfect square
        if len(indices) < n:
            remaining = list(range(len(indices), n))
            indices.extend(remaining)

        hilbert_indices = torch.tensor(
            indices[:n], dtype=torch.long, device=self.device
        )
        self._hilbert_cache[n] = hilbert_indices

        # Also cache the inverse mapping
        inverse_indices = torch.zeros_like(hilbert_indices)
        inverse_indices[hilbert_indices] = torch.arange(n, device=self.device)
        self._inverse_hilbert_cache[n] = inverse_indices

        return hilbert_indices

    def _hilbert_index_to_xy(self, index: int, n: int) -> Tuple[int, int]:
        """Convert Hilbert curve index to 2D coordinates."""
        x = y = 0
        s = 1

        while s < n:
            rx = 1 & (index // 2)
            ry = 1 & (index ^ rx)

            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - y, s - 1 - x
                x, y = y, x

            x += s * rx
            y += s * ry
            index //= 4
            s *= 2

        return x, y

    def _apply_hilbert_ordering(
        self, tensor: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Apply or inverse Hilbert ordering to tensor."""
        batch_size, seq_len, num_heads, head_dim = tensor.shape

        if not self.use_hilbert or seq_len <= self.hilbert_chunk_size:
            return tensor

        # Get indices
        if inverse:
            indices = self._inverse_hilbert_cache.get(seq_len)
            if indices is None:
                self._generate_hilbert_indices(seq_len)
                indices = self._inverse_hilbert_cache[seq_len]
        else:
            indices = self._generate_hilbert_indices(seq_len)

        # Apply ordering
        return tensor[:, indices]

    def _apply_segment_hilbert_ordering(
        self, tensor: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Apply Hilbert ordering to a segment tensor."""
        if not self.use_hilbert:
            return tensor

        batch_size, seg_len, num_heads, head_dim = tensor.shape

        # Skip if segment is too small
        if seg_len <= 64:  # Minimum size for meaningful Hilbert curve
            return tensor

        # Get or generate indices for this segment length
        if inverse:
            indices = self._inverse_hilbert_cache.get(seg_len)
            if indices is None:
                self._generate_hilbert_indices(seg_len)
                indices = self._inverse_hilbert_cache[seg_len]
        else:
            indices = self._hilbert_cache.get(seg_len)
            if indices is None:
                indices = self._generate_hilbert_indices(seg_len)

        # Apply ordering to segment
        return tensor[:, indices]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass with Hilbert curve optimization.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to use causal masking
            return_attention_weights: Whether to return attention weights

        Returns:
            output: Attention output [batch, seq_len, num_heads, head_dim]
            attention_weights: Optional attention weights if requested
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Initialize output (no global Hilbert ordering)
        output = torch.zeros_like(q)

        # Process each segment with its dilation rate
        for seg_idx, (seg_len, dilation_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            if seq_len < seg_len:
                seg_len = seq_len

            # Number of segments
            num_segments = (seq_len + seg_len - 1) // seg_len

            # Assign different segments to different heads for load balancing
            heads_per_segment = max(1, num_heads // len(self.segment_lengths))
            head_start = seg_idx * heads_per_segment
            head_end = min(head_start + heads_per_segment, num_heads)

            if head_start >= num_heads:
                continue

            for seg_idx in range(num_segments):
                # Calculate segment boundaries
                start_idx = seg_idx * seg_len
                end_idx = min(start_idx + seg_len, seq_len)
                actual_seg_len = end_idx - start_idx

                # Get segment queries, keys, values for assigned heads
                q_seg = q[:, start_idx:end_idx, head_start:head_end]
                k_seg = k[:, start_idx:end_idx, head_start:head_end]
                v_seg = v[:, start_idx:end_idx, head_start:head_end]

                # Apply Hilbert ordering to this segment
                q_seg_ordered = self._apply_segment_hilbert_ordering(
                    q_seg, inverse=False
                )
                k_seg_ordered = self._apply_segment_hilbert_ordering(
                    k_seg, inverse=False
                )
                v_seg_ordered = self._apply_segment_hilbert_ordering(
                    v_seg, inverse=False
                )

                # Apply local dilation within the segment
                if dilation_rate > 1:
                    # Get dilated indices within this segment
                    local_indices = torch.arange(
                        0, actual_seg_len, dilation_rate, device=q.device
                    )

                    # Apply local dilation on Hilbert-ordered segments
                    k_seg_dilated = k_seg_ordered[:, local_indices]
                    v_seg_dilated = v_seg_ordered[:, local_indices]
                else:
                    # No dilation, use Hilbert-ordered segments directly
                    k_seg_dilated = k_seg_ordered
                    v_seg_dilated = v_seg_ordered

                # Compute attention scores
                # Need to handle head subset dimension
                # q_seg_ordered: [batch, actual_seg_len, num_heads_subset, head_dim]
                # k_seg_dilated: [batch, dilated_len, num_heads_subset, head_dim]

                # Transpose for attention computation
                q_seg_t = q_seg_ordered.transpose(
                    1, 2
                )  # [batch, num_heads_subset, actual_seg_len, head_dim]
                k_seg_t = k_seg_dilated.transpose(
                    1, 2
                )  # [batch, num_heads_subset, dilated_len, head_dim]

                scores = torch.matmul(q_seg_t, k_seg_t.transpose(-2, -1)) * self.scale

                # Apply causal mask if needed
                if is_causal:
                    q_len = actual_seg_len
                    k_len = k_seg_dilated.shape[1]

                    if dilation_rate > 1:
                        # For dilated attention, we need a special causal mask
                        mask = torch.zeros(
                            q_len, k_len, device=q.device, dtype=torch.bool
                        )
                        local_indices_cpu = (
                            local_indices.cpu().numpy() if dilation_rate > 1 else None
                        )
                        for i in range(q_len):
                            for j, k_pos in enumerate(
                                local_indices_cpu if dilation_rate > 1 else range(k_len)
                            ):
                                if k_pos > i:
                                    mask[i, j] = True
                    else:
                        mask = torch.triu(
                            torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                            diagonal=1,
                        )

                    scores = scores.masked_fill(
                        mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                # Apply softmax
                attn_weights = torch.softmax(scores, dim=-1)

                # Handle NaN values
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                # Apply dropout if needed
                if self.dropout is not None and self.training:
                    attn_weights = self.dropout(attn_weights)

                # Apply attention to values
                v_seg_t = v_seg_dilated.transpose(
                    1, 2
                )  # [batch, num_heads_subset, dilated_len, head_dim]
                seg_output_t = torch.matmul(
                    attn_weights, v_seg_t
                )  # [batch, num_heads_subset, actual_seg_len, head_dim]

                # Transpose back
                seg_output = seg_output_t.transpose(
                    1, 2
                )  # [batch, actual_seg_len, num_heads_subset, head_dim]

                # Apply inverse Hilbert ordering to restore original segment order
                seg_output_restored = self._apply_segment_hilbert_ordering(
                    seg_output, inverse=True
                )

                # Add to output for assigned heads
                output[:, start_idx:end_idx, head_start:head_end] = seg_output_restored

        # No need for global inverse Hilbert ordering since we're doing per-segment

        if return_attention_weights:
            return output, None
        else:
            return output

    @classmethod
    def from_config(cls, config: StandardizedRingConfig, **kwargs):
        """Create instance from standardized config."""
        return cls(config=config, **kwargs)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"dim={self.dim}, heads={self.heads}, "
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"ring_size={self.ring_size}, dropout={self.dropout_p}, "
            f"use_hilbert={self.use_hilbert}, hilbert_chunk_size={self.hilbert_chunk_size}"
        )
