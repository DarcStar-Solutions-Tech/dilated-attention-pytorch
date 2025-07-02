"""
Ring Dilated Attention Hybrid - Fixed Implementation

This is the corrected version that properly implements dilated attention
by segmenting sequences first, then applying dilation within segments.
Combines V3's true ring attention (O(n/p) memory) with V2's correct
dilated attention semantics.
"""

import math
from typing import Optional, List, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

# Import ring utilities from V3
from .ring_attention_utils import (
    all_ring_pass,
    split_by_rank,
)

# Import LSE accumulator from V3
from .ring_attention_lse import StableRingAccumulator, compute_attention_with_lse

# Import V2's utilities
from .utils.gpu_utils import get_optimal_dtype
from .core.pattern_cache import get_global_pattern_cache


# Simple validation function
def validate_inputs(segment_lengths, dilation_rates):
    """Validate segment lengths and dilation rates."""
    if not segment_lengths or not dilation_rates:
        raise ValueError("segment_lengths and dilation_rates cannot be empty")
    if len(segment_lengths) != len(dilation_rates):
        raise ValueError(
            f"segment_lengths and dilation_rates must have same length: "
            f"{len(segment_lengths)} != {len(dilation_rates)}"
        )


# Import create_causal_mask

# Flash attention imports
HAS_FLASH_UTILS = False
try:
    from .utils.attention_utils import (
        detect_available_backend,
    )
    from flash_attn import flash_attn_func as flash_attention_forward

    HAS_FLASH_UTILS = True
except ImportError:
    pass


# Helper function
def exists(val):
    return val is not None


class RingDilatedAttentionHybridFixed(nn.Module):
    """
    Fixed Ring Dilated Attention implementation that correctly applies
    dilated attention semantics with true ring communication.

    Key fix: Properly segments sequences and applies dilation within
    segments, not globally across the entire sequence.
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
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # Validate inputs
        validate_inputs(segment_lengths, dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout

        # Device and dtype setup
        self.device = device or torch.cuda.current_device()
        if dtype is None:
            # Use GPU utilities to select optimal dtype
            self.dtype = get_optimal_dtype(
                self.device, prefer_fp16=True, warn_pascal=False
            )
        else:
            self.dtype = dtype

        # Ring setup
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.ring_size = ring_size or torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.ring_size = 1

        # Features
        self.enable_memory_pool = enable_memory_pool
        self.use_pattern_cache = use_pattern_cache
        self.use_flash_attention = use_flash_attention and HAS_FLASH_UTILS

        # Pattern cache setup
        self._pattern_cache = get_global_pattern_cache() if use_pattern_cache else None
        self._local_pattern_cache = {}

        # Flash attention setup
        self._can_use_flash = False
        self.flash_backend = None
        if self.use_flash_attention and HAS_FLASH_UTILS:
            try:
                self.flash_backend = detect_available_backend(self.device)
                self._can_use_flash = self.flash_backend is not None
            except Exception:
                pass

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
        Forward pass with correct dilated attention semantics.

        The key fix: We now properly segment sequences and apply
        dilation within segments during attention computation.
        """
        b, n, h, d = q.shape

        # Single device fallback
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Multi-GPU ring attention with proper dilated attention
        return self._ring_forward_with_dilated_segments(q, k, v, is_causal)

    def _ring_forward_with_dilated_segments(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """
        Ring attention that properly handles dilated attention within segments.

        Key insight: Each GPU processes its chunk of the sequence, but when
        computing attention, we apply dilation within logical segments.
        """
        b, n, h, d = q.shape

        # Ensure divisibility
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )

        # Split K,V across GPUs (standard ring attention)
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

        # Initialize accumulator
        accumulator = StableRingAccumulator(
            output_shape=(b, h, n, d),
            device=q.device,
            dtype=q.dtype,
        )

        # Ring passing with dilated attention computation
        ring_pass_fn = partial(
            all_ring_pass,
            receive_buffer=self._kv_receive_buffer,
            ring_size=self.ring_size,
        )

        # Process each ring position
        for ring_info, (kv_chunk,) in ring_pass_fn(kv_local):
            if not exists(kv_chunk):
                continue

            k_chunk, v_chunk = kv_chunk
            chunk_idx = ring_info.ring_rank
            chunk_start = chunk_idx * chunk_size

            # Compute dilated attention for this chunk
            chunk_output, chunk_lse = self._compute_dilated_chunk_attention(
                q,
                k_chunk,
                v_chunk,
                chunk_start,
                chunk_size,
                is_causal,
            )

            # Accumulate
            accumulator.update(chunk_output, chunk_lse)

        # Get final output
        output = accumulator.get_output().transpose(1, 2)
        return output

    def _compute_dilated_chunk_attention(
        self,
        q: Tensor,  # (b, n, h, d) - full Q
        k_chunk: Tensor,  # (b, chunk_size, h, d) - chunk of K
        v_chunk: Tensor,  # (b, chunk_size, h, d) - chunk of V
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention with proper dilated attention semantics.

        This is the key fix: We segment the sequences and apply
        dilation within segments, not globally.
        """
        b, n, h, d = q.shape

        # Transpose to attention format
        q_t = q.transpose(1, 2)  # (b, h, n, d)
        k_chunk_t = k_chunk.transpose(1, 2)  # (b, h, chunk_size, d)
        v_chunk_t = v_chunk.transpose(1, 2)  # (b, h, chunk_size, d)

        # Calculate head groups
        heads_per_group = self._calculate_head_groups(h)

        # Pre-allocate output
        output = torch.zeros(b, h, n, d, device=q.device, dtype=q.dtype)
        lse = torch.full((b, h, n), float("-inf"), device=q.device, dtype=q.dtype)

        # Process each head group with its segment configuration
        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Process this head group
            group_output, group_lse = self._process_head_group_segments(
                q_t[:, head_start:head_end],  # Q for this head group
                k_chunk_t[:, head_start:head_end],  # K chunk for this head group
                v_chunk_t[:, head_start:head_end],  # V chunk for this head group
                segment_len,
                dilation_rate,
                i,  # offset index
                chunk_start,
                chunk_size,
                is_causal,
            )

            output[:, head_start:head_end] = group_output
            lse[:, head_start:head_end] = group_lse

            head_start = head_end

        return output, lse

    def _process_head_group_segments(
        self,
        q_group: Tensor,  # (b, group_heads, n, d)
        k_chunk_group: Tensor,  # (b, group_heads, chunk_size, d)
        v_chunk_group: Tensor,  # (b, group_heads, chunk_size, d)
        segment_len: int,
        dilation_rate: int,
        offset_idx: int,
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process segments with dilation for a head group.

        This implements the correct algorithm:
        1. Determine which segments overlap with this chunk
        2. For each segment, apply dilation and compute attention
        """
        b, gh, n, d = q_group.shape

        # Initialize output for this head group
        output = torch.zeros_like(q_group)
        lse = torch.full((b, gh, n), float("-inf"), device=q_group.device)

        # Determine segment boundaries
        num_segments = (n + segment_len - 1) // segment_len

        # Process each segment
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_len
            seg_end = min(seg_start + segment_len, n)
            actual_seg_len = seg_end - seg_start

            # Check if this segment overlaps with the current chunk
            if seg_end <= chunk_start or seg_start >= chunk_start + chunk_size:
                continue  # No overlap

            # Determine overlap region
            overlap_start = max(seg_start, chunk_start)
            overlap_end = min(seg_end, chunk_start + chunk_size)

            # Get Q for this segment
            q_seg = q_group[:, :, seg_start:seg_end, :]

            # Get K,V from chunk that corresponds to this segment's overlap
            k_overlap_start = overlap_start - chunk_start
            k_overlap_end = overlap_end - chunk_start
            k_seg = k_chunk_group[:, :, k_overlap_start:k_overlap_end, :]
            v_seg = v_chunk_group[:, :, k_overlap_start:k_overlap_end, :]

            # Apply dilation within segment
            if dilation_rate > 1:
                # Calculate offset for this segment
                offset = (offset_idx + seg_idx) % dilation_rate

                # Get dilation pattern
                pattern = self._get_segment_dilation_pattern(
                    actual_seg_len, dilation_rate, offset
                )

                # Apply dilation to Q (segment-local indices)
                q_seg_dilated = q_seg.index_select(2, pattern)

                # For K,V we need to map the pattern to chunk-local indices
                overlap_len = overlap_end - overlap_start
                k_pattern = self._map_pattern_to_overlap(
                    pattern, seg_start, overlap_start, overlap_len
                )

                if k_pattern is not None and len(k_pattern) > 0:
                    k_seg_dilated = k_seg.index_select(2, k_pattern)
                    v_seg_dilated = v_seg.index_select(2, k_pattern)
                else:
                    continue  # No valid positions in this chunk
            else:
                q_seg_dilated = q_seg
                k_seg_dilated = k_seg
                v_seg_dilated = v_seg

            # Compute attention for this dilated segment
            seg_output, seg_lse = self._compute_segment_attention(
                q_seg_dilated,
                k_seg_dilated,
                v_seg_dilated,
                seg_start,
                overlap_start,
                is_causal,
            )

            # Map output back to full positions
            if dilation_rate > 1:
                # Place dilated output back in correct positions
                for i, idx in enumerate(pattern):
                    if seg_start + idx < n:
                        output[:, :, seg_start + idx, :] = seg_output[:, :, i, :]
                        lse[:, :, seg_start + idx] = seg_lse[:, :, i]
            else:
                output[:, :, seg_start:seg_end, :] = seg_output
                lse[:, :, seg_start:seg_end] = seg_lse

        return output, lse

    def _get_segment_dilation_pattern(
        self, seg_len: int, dilation_rate: int, offset: int
    ) -> Tensor:
        """Get dilation pattern for a segment."""
        cache_key = (seg_len, dilation_rate, offset)

        if cache_key in self._local_pattern_cache:
            return self._local_pattern_cache[cache_key]

        # Create pattern
        indices = []
        for i in range(0, seg_len, dilation_rate):
            idx = (i + offset) % seg_len
            if idx < seg_len:
                indices.append(idx)

        # Ensure we have enough indices
        if len(indices) < seg_len // dilation_rate:
            # Cycle through to fill
            cycle_len = len(indices)
            while len(indices) < seg_len:
                for i in range(cycle_len):
                    if len(indices) < seg_len:
                        indices.append(indices[i])

        pattern = torch.tensor(indices, device=self.device, dtype=torch.long)
        self._local_pattern_cache[cache_key] = pattern

        return pattern

    def _map_pattern_to_overlap(
        self,
        pattern: Tensor,
        seg_start: int,
        overlap_start: int,
        overlap_len: int,
    ) -> Optional[Tensor]:
        """Map segment-local pattern to chunk-local indices."""
        # Filter pattern to only include positions in overlap
        _ = overlap_start - seg_start
        valid_indices = []

        for idx in pattern:
            global_pos = seg_start + idx
            if overlap_start <= global_pos < overlap_start + overlap_len:
                chunk_local_idx = global_pos - overlap_start
                valid_indices.append(chunk_local_idx)

        if not valid_indices:
            return None

        return torch.tensor(valid_indices, device=pattern.device, dtype=torch.long)

    def _compute_segment_attention(
        self,
        q_seg: Tensor,  # (b, h, seg_len_dilated, d)
        k_seg: Tensor,  # (b, h, overlap_len_dilated, d)
        v_seg: Tensor,  # (b, h, overlap_len_dilated, d)
        seg_start: int,
        overlap_start: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention for a dilated segment."""
        # Use Flash Attention if available
        if self._can_use_flash:
            try:
                # Transpose for Flash
                q_flash = q_seg.transpose(1, 2)  # (b, seg_len, h, d)
                k_flash = k_seg.transpose(1, 2)
                v_flash = v_seg.transpose(1, 2)

                output = flash_attention_forward(
                    q_flash,
                    k_flash,
                    v_flash,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal and seg_start == 0,
                    backend=self.flash_backend,
                )

                # Compute LSE separately
                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) / math.sqrt(
                    q_seg.shape[-1]
                )
                if is_causal:
                    # Apply causal mask
                    q_len, k_len = q_seg.shape[2], k_seg.shape[2]
                    for i in range(q_len):
                        for j in range(k_len):
                            if seg_start + i < overlap_start + j:
                                scores[:, :, i, j] = float("-inf")

                lse = scores.logsumexp(dim=-1)
                return output.transpose(1, 2), lse

            except Exception:
                pass  # Fall through

        # Standard attention computation
        return compute_attention_with_lse(
            q_seg,
            k_seg,
            v_seg,
            scale=1.0 / math.sqrt(q_seg.shape[-1]),
            mask=self._get_segment_causal_mask(
                q_seg.shape[2], k_seg.shape[2], seg_start, overlap_start, is_causal
            ),
            dropout=self.dropout,
            training=self.training,
        )

    def _get_segment_causal_mask(
        self,
        q_len: int,
        k_len: int,
        seg_start: int,
        overlap_start: int,
        is_causal: bool,
    ) -> Optional[Tensor]:
        """Get causal mask for segment attention."""
        if not is_causal:
            return None

        mask = torch.ones(q_len, k_len, device=self.device, dtype=torch.bool)

        for i in range(q_len):
            for j in range(k_len):
                q_pos = seg_start + i
                k_pos = overlap_start + j
                if q_pos < k_pos:
                    mask[i, j] = False

        return mask

    def _calculate_head_groups(self, num_heads: int) -> List[int]:
        """Calculate how many heads belong to each segment/dilation configuration."""
        num_groups = len(self.segment_lengths)
        base_heads = num_heads // num_groups
        extra_heads = num_heads % num_groups

        heads_per_group = [base_heads] * num_groups
        for i in range(extra_heads):
            heads_per_group[i] += 1

        return heads_per_group

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward with proper dilated attention."""
        b, n, h, d = q.shape

        # Transpose to attention format
        q_t = q.transpose(1, 2)  # (b, h, n, d)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Calculate head groups
        heads_per_group = self._calculate_head_groups(h)

        # Process each head group
        output = torch.zeros_like(q_t)
        head_start = 0

        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Process segments for this head group
            group_output = self._process_single_device_segments(
                q_t[:, head_start:head_end],
                k_t[:, head_start:head_end],
                v_t[:, head_start:head_end],
                segment_len,
                dilation_rate,
                i,  # offset index
                is_causal,
            )

            output[:, head_start:head_end] = group_output
            head_start = head_end

        return output.transpose(1, 2)  # (b, n, h, d)

    def _process_single_device_segments(
        self,
        q_group: Tensor,
        k_group: Tensor,
        v_group: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset_idx: int,
        is_causal: bool,
    ) -> Tensor:
        """Process segments on single device."""
        b, gh, n, d = q_group.shape
        output = torch.zeros_like(q_group)

        # Process each segment
        num_segments = (n + segment_len - 1) // segment_len

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_len
            seg_end = min(seg_start + segment_len, n)
            actual_seg_len = seg_end - seg_start

            # Get segment
            q_seg = q_group[:, :, seg_start:seg_end, :]
            k_seg = k_group[:, :, seg_start:seg_end, :]
            v_seg = v_group[:, :, seg_start:seg_end, :]

            # Apply dilation
            if dilation_rate > 1:
                offset = (offset_idx + seg_idx) % dilation_rate
                pattern = self._get_segment_dilation_pattern(
                    actual_seg_len, dilation_rate, offset
                )

                q_seg = q_seg.index_select(2, pattern)
                k_seg = k_seg.index_select(2, pattern)
                v_seg = v_seg.index_select(2, pattern)

            # Compute attention
            seg_output, _ = compute_attention_with_lse(
                q_seg,
                k_seg,
                v_seg,
                scale=1.0 / math.sqrt(d),
                mask=self._get_segment_causal_mask(
                    q_seg.shape[2], k_seg.shape[2], seg_start, seg_start, is_causal
                ),
                dropout=self.dropout,
                training=self.training,
            )

            # Place output back
            if dilation_rate > 1:
                for i, idx in enumerate(pattern):
                    if seg_start + idx < n:
                        output[:, :, seg_start + idx, :] = seg_output[:, :, i, :]
            else:
                output[:, :, seg_start:seg_end, :] = seg_output

        return output


# Alias for compatibility
RingDilatedAttentionHybridCorrected = RingDilatedAttentionHybridFixed
