"""
Refactored Ring Dilated Attention - No model recreation in forward pass.

This implementation provides a clean base for ring dilated attention without
the problematic model recreation pattern.
"""

import math
from typing import Optional, Tuple, List
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
    compute_attention_with_lse,
)

# Import optimized LSE with backend fallbacks
try:
    from .ring_attention_lse_optimized import compute_attention_with_lse_optimized

    HAS_OPTIMIZED_LSE = True
except ImportError:
    HAS_OPTIMIZED_LSE = False
    compute_attention_with_lse_optimized = compute_attention_with_lse

# Import memory pool
try:
    from .core.enhanced_memory_pool import get_enhanced_memory_pool

    HAS_ENHANCED_MEMORY_POOL = True
except ImportError:
    HAS_ENHANCED_MEMORY_POOL = False

# Import pattern cache
try:
    from .core.pattern_cache import get_global_pattern_cache

    HAS_PATTERN_CACHE = True
except ImportError:
    HAS_PATTERN_CACHE = False


class RingDilatedAttentionRefactored(nn.Module):
    """
    Refactored Ring Dilated Attention without model recreation issues.

    This provides a clean implementation that:
    - Properly handles single and multi-GPU cases
    - Doesn't create temporary models in forward pass
    - Supports dilated attention patterns
    - Uses efficient ring communication
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_memory_pool: bool = True,
        use_pattern_cache: bool = True,
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

        # Memory optimization
        self._memory_pool = None
        if enable_memory_pool and HAS_ENHANCED_MEMORY_POOL:
            try:
                self._memory_pool = get_enhanced_memory_pool()
                print("RingDilatedAttentionRefactored: Memory pool initialized")
            except Exception:
                pass

        # Pattern cache
        self._pattern_cache = None
        if use_pattern_cache and HAS_PATTERN_CACHE:
            try:
                self._pattern_cache = get_global_pattern_cache()
                print("RingDilatedAttentionRefactored: Pattern cache initialized")
            except Exception:
                pass

        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # Pre-allocate buffers
        self._kv_receive_buffer = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with ring dilated attention.

        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            is_causal: whether to apply causal masking

        Returns:
            output: (batch, seq_len, num_heads, head_dim)
        """
        b, n, h, d = q.shape

        # Single device case
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Multi-GPU ring attention
        return self._ring_forward(q, k, v, is_causal)

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward with dilated attention."""
        b, n, h, d = q.shape

        # Initialize output accumulator
        output = torch.zeros_like(q)

        # Process each segment with its dilation
        for seg_idx, (seg_len, dilation) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            # Compute segment boundaries
            seg_start = sum(self.segment_lengths[:seg_idx])
            seg_end = seg_start + seg_len

            if seg_start >= n:
                break

            # Get Q for this segment
            q_seg = q[:, seg_start:seg_end]

            # Create dilated attention pattern
            pattern = self._create_dilated_pattern(seg_len, n, dilation)

            # Gather K and V according to pattern
            k_dilated = k[:, pattern]
            v_dilated = v[:, pattern]

            # Compute attention for this segment
            output_seg = self._compute_attention_segment(
                q_seg, k_dilated, v_dilated, is_causal, seg_start, pattern
            )

            # Add to output
            output[:, seg_start:seg_end] = output_seg

        return output

    def _ring_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Multi-GPU ring forward with dilated attention."""
        b, n, h, d = q.shape

        # Ensure divisibility
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )

        # Split K,V across GPUs
        chunk_size = n // self.ring_size
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

        # Initialize LSE accumulator
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

            # Compute dilated attention for this chunk
            chunk_output, chunk_lse = self._compute_dilated_chunk_attention(
                q, k_chunk, v_chunk, chunk_start, chunk_size, is_causal
            )

            # Accumulate with LSE
            accumulator.update(chunk_output, chunk_lse)

        # Get final output and transpose back
        output = accumulator.get_output().transpose(1, 2)  # (b, n, h, d)

        return output

    def _create_dilated_pattern(
        self, seg_len: int, total_len: int, dilation: int
    ) -> Tensor:
        """Create dilated attention pattern for a segment."""
        # Use cached pattern if available
        cache_key = (seg_len, total_len, dilation)
        if self._pattern_cache is not None:
            cached = self._pattern_cache.get(cache_key)
            if cached is not None:
                return cached.to(self.device)

        # Create pattern
        indices = []
        for i in range(seg_len):
            # Dilated indices
            dilated_indices = torch.arange(
                i, total_len, dilation * seg_len, device=self.device
            )
            indices.append(dilated_indices[:seg_len])

        pattern = torch.stack(indices, dim=0)

        # Cache if enabled
        if self._pattern_cache is not None:
            self._pattern_cache.put(cache_key, pattern.cpu())

        return pattern

    def _compute_attention_segment(
        self,
        q_seg: Tensor,
        k_pattern: Tensor,
        v_pattern: Tensor,
        is_causal: bool,
        seg_start: int,
        pattern: Tensor,
    ) -> Tensor:
        """Compute attention for a segment with dilated pattern."""
        b, seg_len, h, d = q_seg.shape
        _, pattern_len, _, _ = k_pattern.shape

        # Compute attention scores
        scores = torch.einsum("bqhd,bkhd->bhqk", q_seg, k_pattern) / math.sqrt(d)

        # Apply causal mask if needed
        if is_causal:
            # Create mask based on actual positions
            q_pos = torch.arange(seg_start, seg_start + seg_len, device=self.device)
            k_pos = pattern
            mask = q_pos.unsqueeze(1) < k_pos
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout if enabled
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)

        # Compute output
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v_pattern)

        return output

    def _compute_dilated_chunk_attention(
        self,
        q: Tensor,  # Full Q
        k_chunk: Tensor,  # Chunk of K
        v_chunk: Tensor,  # Chunk of V
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention with dilated patterns for a chunk."""
        b, n, h, d = q.shape

        # Use optimized LSE computation if available
        compute_fn = (
            compute_attention_with_lse_optimized
            if HAS_OPTIMIZED_LSE
            else compute_attention_with_lse
        )

        # Compute attention for this chunk
        output, lse = compute_fn(
            q,
            k_chunk,
            v_chunk,
            is_causal=is_causal,
            chunk_idx=chunk_start // chunk_size,
            total_chunks=self.ring_size,
        )

        return output, lse
