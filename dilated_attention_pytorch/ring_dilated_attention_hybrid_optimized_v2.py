"""
Ring Dilated Attention Hybrid Optimized V2 - Fixed multi-GPU support.

This implementation fixes the multi-GPU issues by:
- Properly handling chunk boundaries in ring communication
- Correct pattern mapping between segments and chunks
- Efficient memory usage with proper bounds checking
"""

import math
from typing import Optional, Tuple, List
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

# Import ring utilities from V3
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

# Import optimized attention computation
try:
    from .utils.attention_utils import optimize_attention_computation

    HAS_OPTIMIZE_ATTENTION = True
except ImportError:
    HAS_OPTIMIZE_ATTENTION = False


class RingDilatedAttentionHybridOptimizedV2(nn.Module):
    """
    Fixed Hybrid Ring Dilated Attention for multi-GPU.

    Key fixes:
    - Proper chunk boundary handling in ring communication
    - Correct pattern mapping for dilated segments across chunks
    - Efficient memory management for distributed scenarios
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
        use_flash_attention: bool = False,  # Disabled for Pascal
        # New optimization flags
        precompute_patterns: bool = True,
        overlap_communication: bool = False,
    ):
        """Initialize Fixed Hybrid Ring Dilated Attention."""
        super().__init__()

        assert len(segment_lengths) == len(dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout if dropout is not None else 0.0
        self.precompute_patterns = precompute_patterns
        self.overlap_communication = overlap_communication

        # Device and dtype setup
        self.device = device or torch.cuda.current_device()
        self.dtype = dtype or torch.float32

        # Ring setup
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.ring_size = ring_size or dist.get_world_size()
        else:
            self.rank = 0
            self.ring_size = 1

        # Optimization features
        self.enable_memory_pool = enable_memory_pool
        self.enable_profiling = enable_profiling
        self.use_pattern_cache = use_pattern_cache
        self.use_flash_attention = use_flash_attention

        # Initialize memory pool
        self._memory_pool = None
        if self.enable_memory_pool and HAS_ENHANCED_MEMORY_POOL:
            try:
                self._memory_pool = get_enhanced_memory_pool(
                    enable_profiling=self.enable_profiling,
                )
                if self.rank == 0:
                    print(
                        "RingDilatedAttentionHybridOptimizedV2: Memory pool initialized"
                    )
            except Exception as e:
                if self.rank == 0:
                    print(f"Failed to initialize memory pool: {e}")
                self._memory_pool = None

        # Initialize pattern cache
        self._pattern_cache = None
        if self.use_pattern_cache and HAS_PATTERN_CACHE:
            try:
                self._pattern_cache = get_global_pattern_cache()
                if self.rank == 0:
                    print(
                        "RingDilatedAttentionHybridOptimizedV2: Pattern cache initialized"
                    )
            except Exception as e:
                if self.rank == 0:
                    print(f"Failed to initialize pattern cache: {e}")
                self._pattern_cache = None

        # Local caches
        self._dilation_pattern_cache = {}
        self._causal_mask_cache = {}
        self._precomputed_patterns = {}

        # Pre-allocate buffers for ring communication
        self._kv_receive_buffer = None

        # Dropout
        self.dropout_p = dropout

        # Pre-compute patterns if requested
        if self.precompute_patterns:
            self._precompute_all_patterns()

    def _precompute_all_patterns(self):
        """Pre-compute all dilation patterns."""
        for seg_idx, (seg_len, dilation_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            if dilation_rate > 1:
                for offset in range(dilation_rate):
                    key = (seg_len, dilation_rate, offset)
                    if key not in self._precomputed_patterns:
                        pattern = self._create_dilation_pattern(
                            seg_len, dilation_rate, offset
                        )
                        self._precomputed_patterns[key] = pattern

    def _create_dilation_pattern(
        self, seg_len: int, dilation_rate: int, offset: int
    ) -> Tensor:
        """Create a dilation pattern efficiently."""
        # Create pattern indices
        indices = torch.arange(
            offset, seg_len, dilation_rate, device=self.device, dtype=torch.long
        )

        # Ensure we have the right number of elements
        expected_len = (seg_len + dilation_rate - 1 - offset) // dilation_rate
        if indices.shape[0] > expected_len:
            indices = indices[:expected_len]
        elif indices.shape[0] < expected_len:
            # Pad with valid indices
            pad_size = expected_len - indices.shape[0]
            pad_indices = (
                torch.arange(pad_size, device=self.device, dtype=torch.long)
                % indices.shape[0]
            )
            indices = torch.cat([indices, indices[pad_indices]])

        return indices

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with fixed multi-GPU support.
        """
        b, n, h, d = q.shape

        # Single device fallback (for testing)
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Multi-GPU ring attention
        return self._ring_forward_with_dilated_segments(q, k, v, is_causal)

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward (simplified for testing)."""
        # Just use the original hybrid implementation logic
        from .ring_dilated_attention_hybrid import RingDilatedAttentionHybrid

        temp_model = RingDilatedAttentionHybrid(
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            dropout=self.dropout,
            ring_size=1,
            device=self.device,
            dtype=self.dtype,
        )
        return temp_model(q, k, v, is_causal)

    def _ring_forward_with_dilated_segments(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """
        Ring attention with fixed chunk handling.
        """
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
            if self._memory_pool is not None:
                try:
                    self._kv_receive_buffer = self._memory_pool.allocate(
                        kv_local.shape, dtype=kv_local.dtype, device=kv_local.device
                    )
                except:
                    self._kv_receive_buffer = torch.empty_like(kv_local)
            else:
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
            chunk_output, chunk_lse = self._compute_dilated_chunk_attention_fixed(
                q, k_chunk, v_chunk, chunk_start, chunk_size, is_causal
            )

            # Accumulate with LSE
            accumulator.update(chunk_output, chunk_lse)

        # Get final output and transpose back
        output = accumulator.get_output().transpose(1, 2)  # (b, n, h, d)

        # Clean up memory pool if needed
        if self._memory_pool is not None and hasattr(
            self._memory_pool, "maybe_cleanup"
        ):
            self._memory_pool.maybe_cleanup()

        return output

    def _compute_dilated_chunk_attention_fixed(
        self,
        q: Tensor,  # (b, n, h, d) - full Q
        k_chunk: Tensor,  # (b, chunk_size, h, d) - chunk of K
        v_chunk: Tensor,  # (b, chunk_size, h, d) - chunk of V
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention with fixed chunk handling.
        """
        b, n, h, d = q.shape

        # Transpose to attention format
        q_t = q.transpose(1, 2)  # (b, h, n, d)
        k_chunk_t = k_chunk.transpose(1, 2)  # (b, h, chunk_size, d)
        v_chunk_t = v_chunk.transpose(1, 2)  # (b, h, chunk_size, d)

        # Calculate head groups
        heads_per_group = self._calculate_head_groups(h)

        # Pre-allocate output
        if self._memory_pool is not None:
            try:
                output = self._memory_pool.allocate(
                    (b, h, n, d), dtype=q.dtype, device=q.device
                )
                lse = self._memory_pool.allocate(
                    (b, h, n), dtype=q.dtype, device=q.device
                )
                output.zero_()
                lse.fill_(float("-inf"))
            except:
                output = torch.zeros(b, h, n, d, device=q.device, dtype=q.dtype)
                lse = torch.full(
                    (b, h, n), float("-inf"), device=q.device, dtype=q.dtype
                )
        else:
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

            # Process this head group with fixed segment handling
            group_output, group_lse = self._process_head_group_segments_fixed(
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

    def _process_head_group_segments_fixed(
        self,
        q_group: Tensor,  # (b, gh, n, d)
        k_chunk_group: Tensor,  # (b, gh, chunk_size, d)
        v_chunk_group: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset_idx: int,
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process head group segments with fixed chunk handling.
        """
        b, gh, n, d = q_group.shape

        # Allocate output
        if self._memory_pool is not None:
            try:
                output = self._memory_pool.allocate(
                    (b, gh, n, d), dtype=q_group.dtype, device=q_group.device
                )
                lse = self._memory_pool.allocate(
                    (b, gh, n), dtype=q_group.dtype, device=q_group.device
                )
                output.zero_()
                lse.fill_(float("-inf"))
            except:
                output = torch.zeros(
                    b, gh, n, d, device=q_group.device, dtype=q_group.dtype
                )
                lse = torch.full(
                    (b, gh, n),
                    float("-inf"),
                    device=q_group.device,
                    dtype=q_group.dtype,
                )
        else:
            output = torch.zeros(
                b, gh, n, d, device=q_group.device, dtype=q_group.dtype
            )
            lse = torch.full(
                (b, gh, n), float("-inf"), device=q_group.device, dtype=q_group.dtype
            )

        # Process segments
        num_segments = (n + segment_len - 1) // segment_len

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_len
            seg_end = min(seg_start + segment_len, n)
            actual_seg_len = seg_end - seg_start

            # Get query segment
            q_seg = q_group[:, :, seg_start:seg_end]

            if dilation_rate > 1:
                # Apply dilation to query segment
                offset = offset_idx % dilation_rate
                q_pattern = self._get_segment_dilation_pattern(
                    actual_seg_len, dilation_rate, offset
                )

                # Process all chunks that overlap with this segment
                # For each position in the dilated query, find which chunk contains the corresponding key
                seg_output = torch.zeros(
                    b,
                    gh,
                    q_pattern.shape[0],
                    d,
                    device=q_group.device,
                    dtype=q_group.dtype,
                )
                seg_lse = torch.full(
                    (b, gh, q_pattern.shape[0]),
                    float("-inf"),
                    device=q_group.device,
                    dtype=q_group.dtype,
                )

                # Compute which global positions the dilated query attends to
                global_q_positions = seg_start + q_pattern

                # Check if any of these positions fall within the current chunk
                chunk_mask = (global_q_positions >= chunk_start) & (
                    global_q_positions < chunk_start + chunk_size
                )

                if chunk_mask.any():
                    # Get the dilated query
                    q_seg_dilated = q_seg.index_select(2, q_pattern)

                    # For K and V, we need ALL positions from the chunk
                    # because different query positions might attend to different keys
                    k_chunk_seg = k_chunk_group
                    v_chunk_seg = v_chunk_group

                    # Compute attention
                    attn_output, attn_lse = self._compute_attention_with_causality(
                        q_seg_dilated,
                        k_chunk_seg,
                        v_chunk_seg,
                        global_q_positions,
                        chunk_start,
                        is_causal,
                    )

                    # Update output for this segment
                    seg_output = attn_output
                    seg_lse = attn_lse

                # Write back dilated results
                for i, idx in enumerate(q_pattern):
                    if seg_start + idx < n:
                        output[:, :, seg_start + idx] = seg_output[:, :, i]
                        lse[:, :, seg_start + idx] = seg_lse[:, :, i]
            else:
                # No dilation - standard attention
                # Determine overlap between segment and chunk
                overlap_start = max(seg_start, chunk_start)
                overlap_end = min(seg_end, chunk_start + chunk_size)

                if overlap_start < overlap_end:
                    # Get overlapping K,V from chunk
                    k_overlap = k_chunk_group[
                        :, :, overlap_start - chunk_start : overlap_end - chunk_start
                    ]
                    v_overlap = v_chunk_group[
                        :, :, overlap_start - chunk_start : overlap_end - chunk_start
                    ]

                    # Compute attention
                    seg_output, seg_lse = self._compute_segment_attention_simple(
                        q_seg, k_overlap, v_overlap, seg_start, overlap_start, is_causal
                    )

                    output[:, :, seg_start:seg_end] = seg_output
                    lse[:, :, seg_start:seg_end] = seg_lse

        return output, lse

    def _compute_attention_with_causality(
        self,
        q: Tensor,  # (b, gh, q_len, d)
        k: Tensor,  # (b, gh, k_len, d)
        v: Tensor,  # (b, gh, k_len, d)
        q_positions: Tensor,  # Global positions of Q
        k_start: int,  # Start position of K chunk
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention with proper causal masking for global positions."""
        scale = 1.0 / math.sqrt(q.shape[-1])

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask if needed
        if is_causal:
            _ = q.shape[2]
            k_len = k.shape[2]

            # Create mask based on global positions
            q_pos_expanded = q_positions.unsqueeze(1)  # (q_len, 1)
            k_positions = torch.arange(k_start, k_start + k_len, device=k.device)
            k_pos_expanded = k_positions.unsqueeze(0)  # (1, k_len)

            # Mask where q_pos < k_pos (can't attend to future)
            causal_mask = q_pos_expanded >= k_pos_expanded  # (q_len, k_len)
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Compute attention
        # Manual computation since we have pre-computed scores
        attn_weights = torch.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)
        output = torch.matmul(attn_weights, v)
        lse = scores.logsumexp(dim=-1)

        return output, lse

    def _compute_segment_attention_simple(
        self,
        q_seg: Tensor,
        k_seg: Tensor,
        v_seg: Tensor,
        seg_start: int,
        k_start: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Simple attention computation for non-dilated segments."""
        scale = 1.0 / math.sqrt(q_seg.shape[-1])

        # Apply causal mask if needed
        mask = None
        if is_causal and seg_start <= k_start:
            # Need causal mask
            q_len = q_seg.shape[2]
            k_len = k_seg.shape[2]
            mask = torch.ones(q_len, k_len, device=q_seg.device, dtype=torch.bool)

            for i in range(q_len):
                for j in range(k_len):
                    if seg_start + i < k_start + j:
                        mask[i, j] = False

        # Compute attention scores
        scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale

        # Apply mask if needed
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Manual computation
        attn_weights = torch.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)
        output = torch.matmul(attn_weights, v_seg)
        lse = scores.logsumexp(dim=-1)

        return output, lse

    def _get_segment_dilation_pattern(
        self, seg_len: int, dilation_rate: int, offset: int
    ) -> Tensor:
        """Get dilation pattern with efficient caching."""
        cache_key = (seg_len, dilation_rate, offset)

        # Check pre-computed patterns first
        if cache_key in self._precomputed_patterns:
            return self._precomputed_patterns[cache_key]

        # Check local cache
        if cache_key in self._dilation_pattern_cache:
            return self._dilation_pattern_cache[cache_key]

        # Create pattern
        pattern = self._create_dilation_pattern(seg_len, dilation_rate, offset)
        self._dilation_pattern_cache[cache_key] = pattern

        return pattern

    def _calculate_head_groups(self, num_heads: int) -> List[int]:
        """Calculate how many heads belong to each segment/dilation configuration."""
        num_groups = len(self.segment_lengths)
        base_heads = num_heads // num_groups
        extra_heads = num_heads % num_groups

        heads_per_group = [base_heads] * num_groups
        for i in range(extra_heads):
            heads_per_group[i] += 1

        return heads_per_group
