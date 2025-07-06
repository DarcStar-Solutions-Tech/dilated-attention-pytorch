"""
Ring Dilated Attention with Hilbert SFC applied to dilated access patterns.

This implementation applies Hilbert curves to the dilated attention patterns
AFTER segmentation, providing better cache locality for the actual memory
access patterns used during attention computation.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

from .ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)


class RingDilatedAttentionHilbertV2(RingDilatedAttentionHybridOptimizedV2):
    """
    Improved Hilbert Ring Attention that applies SFC to dilated patterns.

    Key improvement: Hilbert ordering is applied to the actual dilated
    access patterns within each segment, not to the full sequence.
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
        # New optimization flags
        precompute_patterns: bool = True,
        overlap_communication: bool = False,
        # Hilbert-specific
        use_hilbert: bool = True,
        hilbert_mode: str = "dilated",  # "dilated" or "segment"
        **kwargs,  # Catch any extra arguments
    ):
        """Initialize Hilbert V2 Ring Dilated Attention."""
        super().__init__(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            enable_memory_pool=enable_memory_pool,
            enable_profiling=enable_profiling,
            use_pattern_cache=use_pattern_cache,
            use_flash_attention=use_flash_attention,
            precompute_patterns=precompute_patterns,
            overlap_communication=overlap_communication,
        )

        self.use_hilbert = use_hilbert
        self.hilbert_mode = hilbert_mode
        self._hilbert_pattern_cache = {}

    def _generate_hilbert_indices(self, n: int) -> torch.Tensor:
        """Generate Hilbert curve indices for n points."""
        if n in self._hilbert_pattern_cache:
            return self._hilbert_pattern_cache[n]

        # For small n, use identity
        if n <= 4:
            indices = torch.arange(n, device=self.device, dtype=torch.long)
            self._hilbert_pattern_cache[n] = indices
            return indices

        # Find smallest power of 2 that fits
        size = 1
        while size * size < n:
            size *= 2

        def hilbert_d2xy(size: int, d: int) -> Tuple[int, int]:
            """Convert distance along curve to (x,y) coordinates."""
            x = y = 0
            s = 1
            while s < size:
                rx = 1 if (d // 2) & 1 else 0
                ry = 1 if (d ^ rx) & 1 else 0
                if ry == 0:
                    if rx == 1:
                        x, y = size - 1 - x, size - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d //= 4
                s *= 2
            return x, y

        # Generate Hilbert ordering
        indices = []
        for d in range(min(n, size * size)):
            x, y = hilbert_d2xy(size, d)
            linear_idx = y * size + x
            if linear_idx < n:
                indices.append(linear_idx)

        # Handle any remaining indices
        indices.extend(range(len(indices), n))

        result = torch.tensor(indices, device=self.device, dtype=torch.long)
        self._hilbert_pattern_cache[n] = result
        return result

    def _apply_hilbert_to_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply Hilbert ordering to a dilation pattern."""
        if not self.use_hilbert or self.hilbert_mode != "dilated":
            return pattern

        n = pattern.shape[0]
        if n <= 1:
            return pattern

        # Get Hilbert indices for pattern length
        hilbert_indices = self._generate_hilbert_indices(n)

        # Reorder pattern according to Hilbert curve
        return pattern[hilbert_indices]

    def _get_segment_dilation_pattern(
        self, seg_len: int, dilation_rate: int, offset: int
    ) -> Tensor:
        """Get dilation pattern with optional Hilbert ordering."""
        # Get base pattern from parent
        pattern = super()._get_segment_dilation_pattern(seg_len, dilation_rate, offset)

        # Apply Hilbert ordering to the dilated indices
        if self.use_hilbert and self.hilbert_mode == "dilated":
            # The pattern contains indices to attend to
            # We want to reorder these indices for better cache locality
            pattern_sorted = torch.sort(pattern)[0]
            hilbert_ordered = self._apply_hilbert_to_pattern(
                torch.arange(pattern.shape[0], device=pattern.device)
            )
            pattern = pattern_sorted[hilbert_ordered]

        return pattern

    def _compute_attention_with_causality(
        self,
        q: Tensor,  # (b, gh, q_len, d)
        k: Tensor,  # (b, gh, k_len, d)
        v: Tensor,  # (b, gh, k_len, d)
        q_positions: Tensor,  # Global positions of Q
        k_start: int,  # Start position of K chunk
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention with optional Hilbert reordering of K,V access.
        """
        if self.use_hilbert and self.hilbert_mode == "segment":
            # Apply Hilbert ordering to K,V within this segment
            k_len = k.shape[2]
            hilbert_indices = self._generate_hilbert_indices(k_len)

            # Reorder K,V according to Hilbert curve
            k = k[:, :, hilbert_indices]
            v = v[:, :, hilbert_indices]

        # Use parent's attention computation
        return super()._compute_attention_with_causality(
            q, k, v, q_positions, k_start, is_causal
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass with improved Hilbert ordering."""
        # Don't apply Hilbert to full sequence - let parent handle splitting
        # Hilbert will be applied to dilated patterns or segments
        return super().forward(q, k, v, is_causal)
