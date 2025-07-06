"""
Fixed Ring Dilated Attention with Hilbert SFC Optimization.

This fixes the performance issue where new model instances were created
during each forward pass.
"""

import math
from typing import Optional, Dict
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

# Import the base implementation
from .ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)


class RingDilatedAttentionHilbertOptimizedFixed(nn.Module):
    """
    Fixed Ring Dilated Attention with Hilbert Space-Filling Curve optimization.

    This fixes the performance issue where the base class was being instantiated
    on every forward pass.
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
        # Hilbert-specific
        hilbert_chunk_size: Optional[int] = None,
        cache_hilbert_mappings: bool = True,
        apply_hilbert_to_kv: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.hilbert_chunk_size = hilbert_chunk_size or max(segment_lengths)
        self.cache_hilbert_mappings = cache_hilbert_mappings
        self.apply_hilbert_to_kv = apply_hilbert_to_kv

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

        # Create the base implementation ONCE
        self.base_impl = RingDilatedAttentionHybridOptimizedV2(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=self.ring_size,
            device=self.device,
            dtype=self.dtype,
            enable_memory_pool=enable_memory_pool,
            enable_profiling=enable_profiling,
            use_pattern_cache=use_pattern_cache,
            use_flash_attention=use_flash_attention,
        )

        # Hilbert mapping cache
        self._hilbert_cache: Dict[int, Tensor] = {}

        # Memory optimization
        self._kv_receive_buffer = None
        self._memory_pool = (
            self.base_impl._memory_pool
            if hasattr(self.base_impl, "_memory_pool")
            else None
        )

        print(
            "RingDilatedAttentionHilbertOptimizedFixed: Initialized with base implementation"
        )

    def _generate_hilbert_mapping(self, seq_len: int) -> Tensor:
        """Generate Hilbert curve mapping for sequence reordering."""
        if self.cache_hilbert_mappings and seq_len in self._hilbert_cache:
            return self._hilbert_cache[seq_len]

        # Find the appropriate level for Hilbert curve
        levels = int(math.log2(seq_len))
        assert 2**levels == seq_len, (
            f"Sequence length must be power of 2, got {seq_len}"
        )

        # Generate Hilbert curve indices
        from .utils.hilbert_curve import generate_hilbert_indices

        indices = generate_hilbert_indices(levels)
        mapping = torch.tensor(indices, device=self.device, dtype=torch.long)

        if self.cache_hilbert_mappings:
            self._hilbert_cache[seq_len] = mapping

        return mapping

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
        Forward pass with Hilbert-optimized ring attention.

        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            is_causal: whether to apply causal masking

        Returns:
            output: (batch, seq_len, num_heads, head_dim)
        """
        b, n, h, d = q.shape

        # Single device fallback
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Multi-GPU ring attention with Hilbert optimization
        return self._ring_forward_with_hilbert(q, k, v, is_causal)

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward with Hilbert ordering."""
        # Apply Hilbert ordering to improve cache locality
        q_hilbert = self._apply_hilbert_ordering(q)
        k_hilbert = self._apply_hilbert_ordering(k)
        v_hilbert = self._apply_hilbert_ordering(v)

        # Use the base implementation
        output_hilbert = self.base_impl(q_hilbert, k_hilbert, v_hilbert, is_causal)

        # Reverse Hilbert ordering
        output = self._apply_hilbert_ordering(output_hilbert, inverse=True)

        return output

    def _ring_forward_with_hilbert(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """
        Ring attention with Hilbert curve optimization.

        The key insight: apply Hilbert ordering to improve cache efficiency
        during attention computation while maintaining correct ring communication.
        """
        b, n, h, d = q.shape

        # Ensure divisibility
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )

        # Apply Hilbert ordering to Q for better cache locality
        q_hilbert = self._apply_hilbert_ordering(q)

        # Split K,V across GPUs
        chunk_size = n // self.ring_size

        # Optionally apply Hilbert to K,V chunks for consistency
        if self.apply_hilbert_to_kv:
            k_hilbert = self._apply_hilbert_ordering(k)
            v_hilbert = self._apply_hilbert_ordering(v)
            k_local = split_by_rank(k_hilbert, self.rank, self.ring_size)
            v_local = split_by_rank(v_hilbert, self.rank, self.ring_size)
        else:
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

        # Initialize LSE accumulator (in Hilbert space)
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
            chunk_idx = ring_info.ring_rank
            chunk_start = chunk_idx * chunk_size

            # Compute dilated attention for this chunk (in Hilbert space)
            # Use the base implementation's method directly
            chunk_output, chunk_lse = self.base_impl._compute_dilated_chunk_attention(
                q_hilbert, k_chunk, v_chunk, chunk_start, chunk_size, is_causal
            )

            # Accumulate with LSE
            accumulator.update(chunk_output, chunk_lse)

        # Get final output and transpose back
        output_hilbert = accumulator.get_output().transpose(1, 2)  # (b, n, h, d)

        # Reverse Hilbert ordering to get back to original space
        output = self._apply_hilbert_ordering(output_hilbert, inverse=True)

        return output
