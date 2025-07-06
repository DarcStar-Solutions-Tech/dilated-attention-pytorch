"""
Hilbert-optimized Ring Dilated Attention - Refactored version.

This properly inherits from the refactored base class without model recreation issues.
"""

import math
from typing import Optional, List, Dict
import torch
from torch import Tensor

# Import refactored base
from .ring_dilated_attention_hybrid_optimized_v2_refactored import (
    RingDilatedAttentionHybridOptimizedV2Refactored,
)


class RingDilatedAttentionHilbertRefactored(
    RingDilatedAttentionHybridOptimizedV2Refactored
):
    """
    Hilbert-optimized Ring Dilated Attention without model recreation.

    This adds Hilbert space-filling curve optimization on top of the
    refactored base implementation.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # Optimization features
        enable_memory_pool: bool = True,
        enable_profiling: bool = False,
        use_pattern_cache: bool = True,
        use_flash_attention: bool = False,
        # Hilbert-specific
        cache_hilbert_mappings: bool = True,
        apply_hilbert_to_kv: bool = True,
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
            enable_profiling=enable_profiling,
            use_pattern_cache=use_pattern_cache,
            use_flash_attention=use_flash_attention,
        )

        # Hilbert-specific attributes
        self.cache_hilbert_mappings = cache_hilbert_mappings
        self.apply_hilbert_to_kv = apply_hilbert_to_kv

        # Hilbert mapping cache
        self._hilbert_cache: Dict[int, Tensor] = {}

        print(
            "RingDilatedAttentionHilbertRefactored: Initialized with Hilbert optimization"
        )

    def _generate_hilbert_mapping(self, seq_len: int) -> Tensor:
        """Generate Hilbert curve mapping for sequence reordering."""
        if self.cache_hilbert_mappings and seq_len in self._hilbert_cache:
            return self._hilbert_cache[seq_len]

        # Use bit-reversal permutation for power-of-2 sequences
        indices = torch.arange(seq_len, device=self.device, dtype=torch.long)

        n_bits = int(math.log2(seq_len))
        if 2**n_bits == seq_len:
            # Bit reversal for better cache locality
            reversed_indices = torch.zeros_like(indices)
            for i in range(seq_len):
                rev = 0
                num = i
                for _ in range(n_bits):
                    rev = (rev << 1) | (num & 1)
                    num >>= 1
                reversed_indices[i] = rev
            indices = reversed_indices
        else:
            # For non-power-of-2, use a different permutation
            # Gray code inspired permutation
            gray_indices = torch.zeros_like(indices)
            for i in range(seq_len):
                gray_indices[i] = i ^ (i >> 1)
            # Map to valid range
            indices = gray_indices % seq_len

        if self.cache_hilbert_mappings:
            self._hilbert_cache[seq_len] = indices

        return indices

    def _apply_hilbert_ordering(self, tensor: Tensor, inverse: bool = False) -> Tensor:
        """Apply or reverse Hilbert curve ordering to tensor."""
        batch_size, seq_len = tensor.shape[:2]
        rest_dims = list(tensor.shape[2:])

        # Generate mapping
        mapping = self._generate_hilbert_mapping(seq_len)

        if inverse:
            # Create inverse mapping
            inverse_mapping = torch.empty_like(mapping)
            inverse_mapping[mapping] = torch.arange(seq_len, device=self.device)
            mapping = inverse_mapping

        # Reshape tensor for easier manipulation
        tensor_flat = tensor.reshape(batch_size, seq_len, -1)

        # Apply mapping
        mapping_expanded = (
            mapping.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, tensor_flat.shape[-1])
        )

        # Gather along sequence dimension
        reordered = tensor_flat.gather(1, mapping_expanded)

        # Reshape back to original shape
        final_shape = [batch_size, seq_len] + rest_dims
        return reordered.reshape(final_shape)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with Hilbert optimization.

        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            is_causal: whether to apply causal masking

        Returns:
            output: (batch, seq_len, num_heads, head_dim)
        """
        # Apply Hilbert ordering to inputs
        q_hilbert = self._apply_hilbert_ordering(q)
        k_hilbert = self._apply_hilbert_ordering(k) if self.apply_hilbert_to_kv else k
        v_hilbert = self._apply_hilbert_ordering(v) if self.apply_hilbert_to_kv else v

        # Call parent's forward method with Hilbert-ordered tensors
        output_hilbert = super().forward(q_hilbert, k_hilbert, v_hilbert, is_causal)

        # Reverse Hilbert ordering on output
        output = self._apply_hilbert_ordering(output_hilbert, inverse=True)

        return output
