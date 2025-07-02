"""
Ring Dilated Attention Hybrid - True ring attention with production features.

This implementation combines:
- V3's true ring communication (O(n/p) memory scaling)
- V2's dilation support and optimizations (FIXED to use proper segment-wise dilation)
- V3's LSE accumulation for numerical stability
- V2's memory pool, caching, and Flash Attention support

IMPORTANT: This version fixes the dilated attention computation to properly
segment sequences first, then apply dilation within segments.
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
    create_causal_mask,
)

# Import LSE utilities from V3
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

# Import memory pool from V2
try:
    from .core.enhanced_memory_pool import get_enhanced_memory_pool

    HAS_ENHANCED_MEMORY_POOL = True
except ImportError:
    HAS_ENHANCED_MEMORY_POOL = False

# Import pattern cache from V2
try:
    from .core.pattern_cache import get_global_pattern_cache

    HAS_PATTERN_CACHE = True
except ImportError:
    HAS_PATTERN_CACHE = False

# Import Flash Attention utilities from V2
try:
    from .utils.flash_attention_utils import (
        flash_attention_forward,
        get_flash_attention_support,
    )

    HAS_FLASH_UTILS = True
except ImportError:
    HAS_FLASH_UTILS = False


class RingDilatedAttentionHybrid(nn.Module):
    """
    Hybrid Ring Dilated Attention combining the best of V2 and V3.

    Features:
    - True ring communication with O(n/p) memory scaling (V3)
    - Full dilation support in multi-GPU mode (V2) - FIXED to use segment-wise dilation
    - LSE accumulation for numerical stability (V3)
    - Memory pool and pattern caching (V2)
    - Flash Attention integration (V2)
    - Hardware-aware execution paths (V2)
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # V2 optimization features
        enable_memory_pool: bool = True,
        enable_profiling: bool = False,
        lightweight_pool: bool = True,
        use_pattern_cache: bool = True,
        memory_pool_threshold_mb: float = 16.0,
        use_flash_attention: bool = True,
        # V3 features (bucketing disabled due to issues)
        use_bucketed: bool = False,
    ):
        """Initialize Hybrid Ring Dilated Attention."""
        super().__init__()

        assert len(segment_lengths) == len(dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout if dropout is not None else 0.0

        # Device and dtype setup
        self.device = device or torch.cuda.current_device()
        self.dtype = dtype

        # Ring setup
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.ring_size = ring_size or dist.get_world_size()
        else:
            self.rank = 0
            self.ring_size = 1

        # V2 optimization features
        self.enable_memory_pool = enable_memory_pool and HAS_ENHANCED_MEMORY_POOL
        self.enable_profiling = enable_profiling
        self.lightweight_pool = lightweight_pool
        self.use_pattern_cache = use_pattern_cache and HAS_PATTERN_CACHE
        self.memory_pool_threshold_mb = memory_pool_threshold_mb
        self.use_flash_attention = use_flash_attention and HAS_FLASH_UTILS

        # Initialize memory pool if enabled
        self._memory_pool = None
        if self.enable_memory_pool:
            self._memory_pool = get_enhanced_memory_pool(
                enable_profiling=self.enable_profiling,
            )

        # Initialize pattern cache
        self._pattern_cache = None
        if self.use_pattern_cache:
            self._pattern_cache = get_global_pattern_cache()

        # Local caches (always used)
        self._dilation_pattern_cache = {}
        self._causal_mask_cache = {}

        # Flash Attention setup
        self._can_use_flash = False
        self._skip_flash_attempt = False
        self._use_direct_sdpa = False
        self.flash_backend = None

        if self.use_flash_attention and HAS_FLASH_UTILS:
            # Get Flash Attention support info
            support_info = get_flash_attention_support(self.device)
            self._can_use_flash = support_info.get("has_flash_attn", False)
            self.flash_backend = support_info.get("recommended_backend")

            # Determine if we should skip Flash attempts
            if hasattr(torch.cuda, "get_device_capability"):
                major, minor = torch.cuda.get_device_capability(self.device)
                compute_capability = major + minor / 10

                # For Pascal GPUs, prefer direct SDPA
                if compute_capability <= 6.1:
                    self._skip_flash_attempt = True
                    self._use_direct_sdpa = True
                    if self.rank == 0:
                        print(
                            f"RingDilatedAttentionHybrid: Using SDPA path for CC {compute_capability}"
                        )

        # Smart dtype selection based on GPU
        if self.dtype is None:
            from .utils.gpu_utils import get_optimal_dtype

            self.dtype = get_optimal_dtype(
                self.device, prefer_fp16=True, warn_pascal=False
            )

        # Pre-allocate buffers for ring communication
        self._kv_receive_buffer = None

        # Dropout
        self.dropout_p = dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass using true ring attention with V2's features.

        Key: Uses ring passing (V3) instead of all_gather (V2) for true O(n/p) scaling.
        Fixed: Now properly applies dilated attention within segments.
        """
        b, n, h, d = q.shape

        # Single device fallback
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Use proper dilated attention with segmentation
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

        This is the key fix: We segment sequences and apply dilation within
        segments during attention computation, not globally beforehand.
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

        # Stack for ring passing (NO dilation applied yet!)
        kv_local = torch.stack((k_local, v_local))

        # Pre-allocate receive buffer if needed
        if (
            self._kv_receive_buffer is None
            or self._kv_receive_buffer.shape != kv_local.shape
        ):
            self._kv_receive_buffer = torch.empty_like(kv_local)

        # Initialize LSE accumulator (V3's numerical stability)
        accumulator = StableRingAccumulator(
            output_shape=(b, h, n, d),  # Note: heads before seq for LSE
            device=q.device,
            dtype=q.dtype,
        )

        # TRUE RING ATTENTION: Pass K,V chunks around the ring
        # This is the key difference from V2 which uses all_gather
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

            # Accumulate with LSE
            accumulator.update(chunk_output, chunk_lse)

        # Get final output and transpose back
        output = accumulator.get_output().transpose(1, 2)  # (b, n, h, d)

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
        if self.enable_memory_pool and self._memory_pool is not None:
            output = self._memory_pool.allocate(
                (b, h, n, d), dtype=q.dtype, device=q.device
            )
            lse = self._memory_pool.allocate((b, h, n), dtype=q.dtype, device=q.device)
            lse.fill_(float("-inf"))
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

            # Process this head group with proper segmentation
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
                offset = (
                    offset_idx % dilation_rate
                )  # Fixed: consistent offset across segments

                # Get dilation pattern for segment
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

        # Check local cache first
        if cache_key in self._dilation_pattern_cache:
            return self._dilation_pattern_cache[cache_key]

        # Check global cache if enabled
        if self.use_pattern_cache and self._pattern_cache is not None:
            full_key = f"segment_dilation_{cache_key}"
            if full_key in self._pattern_cache:
                pattern = self._pattern_cache[full_key]
                if pattern.device != self.device:
                    pattern = pattern.to(self.device)
                return pattern

        # Create pattern for segment
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

        # Cache locally
        self._dilation_pattern_cache[cache_key] = pattern

        # Store in global cache if enabled
        if self.use_pattern_cache and self._pattern_cache is not None:
            full_key = f"segment_dilation_{cache_key}"
            self._pattern_cache[full_key] = (
                pattern.cpu() if pattern.is_cuda else pattern
            )

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
        # Try Flash Attention if available
        if self._can_use_flash and not self._skip_flash_attempt:
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

                # Compute LSE separately for accumulation
                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) / math.sqrt(
                    q_seg.shape[-1]
                )
                if is_causal:
                    # Apply causal mask properly for segment
                    q_len, k_len = q_seg.shape[2], k_seg.shape[2]
                    for i in range(q_len):
                        for j in range(k_len):
                            if seg_start + i < overlap_start + j:
                                scores[:, :, i, j] = float("-inf")

                lse = scores.logsumexp(dim=-1)
                return output.transpose(1, 2), lse

            except Exception:
                pass  # Fall through to standard

        # Use optimized attention computation with backend fallbacks
        if HAS_OPTIMIZED_LSE:
            return compute_attention_with_lse_optimized(
                q_seg,
                k_seg,
                v_seg,
                scale=1.0 / math.sqrt(q_seg.shape[-1]),
                mask=self._get_segment_causal_mask(
                    q_seg.shape[2], k_seg.shape[2], seg_start, overlap_start, is_causal
                ),
                dropout=self.dropout,
                training=self.training,
                is_causal=is_causal
                and seg_start == 0,  # Only first segment needs causal
            )
        else:
            # Fallback to standard computation
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

        cache_key = (q_len, k_len, seg_start, overlap_start)

        if cache_key not in self._causal_mask_cache:
            mask = torch.ones(q_len, k_len, device=self.device, dtype=torch.bool)

            for i in range(q_len):
                for j in range(k_len):
                    q_pos = seg_start + i
                    k_pos = overlap_start + j
                    if q_pos < k_pos:
                        mask[i, j] = False

            self._causal_mask_cache[cache_key] = mask

            # Limit cache size
            if len(self._causal_mask_cache) > 100:
                keys_to_remove = list(self._causal_mask_cache.keys())[:50]
                for key in keys_to_remove:
                    del self._causal_mask_cache[key]

        return self._causal_mask_cache[cache_key]

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
        """Process segments on single device with proper dilation."""
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

            # Apply dilation within segment
            if dilation_rate > 1:
                offset = (
                    offset_idx % dilation_rate
                )  # Fixed: consistent offset across segments
                pattern = self._get_segment_dilation_pattern(
                    actual_seg_len, dilation_rate, offset
                )

                q_seg = q_seg.index_select(2, pattern)
                k_seg = k_seg.index_select(2, pattern)
                v_seg = v_seg.index_select(2, pattern)

            # Compute attention for segment with optimized backend
            if HAS_OPTIMIZED_LSE:
                seg_output, _ = compute_attention_with_lse_optimized(
                    q_seg,
                    k_seg,
                    v_seg,
                    scale=1.0 / math.sqrt(d),
                    mask=self._get_segment_causal_mask(
                        q_seg.shape[2], k_seg.shape[2], seg_start, seg_start, is_causal
                    ),
                    dropout=self.dropout,
                    training=self.training,
                    is_causal=is_causal and seg_start == 0,
                )
            else:
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

    # Keep all other methods from original implementation
    def _get_causal_mask(
        self, seq_len_q: int, seq_len_kv: int, chunk_offset: int = 0
    ) -> Tensor:
        """Get cached causal mask (from V2) - kept for compatibility."""
        cache_key = (seq_len_q, seq_len_kv, chunk_offset)

        if cache_key not in self._causal_mask_cache:
            # Create causal mask for chunk
            if chunk_offset > 0:
                # For ring chunks
                q_positions = torch.arange(seq_len_q, device=self.device)
                kv_positions = torch.arange(
                    chunk_offset, chunk_offset + seq_len_kv, device=self.device
                )
                mask = create_causal_mask(q_positions, kv_positions, self.device)
            else:
                # Standard causal mask
                mask = torch.triu(
                    torch.ones(
                        seq_len_q, seq_len_kv, device=self.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                mask = ~mask  # Invert for True = attend

            self._causal_mask_cache[cache_key] = mask

            # Limit cache size
            if len(self._causal_mask_cache) > 100:
                keys_to_remove = list(self._causal_mask_cache.keys())[:50]
                for key in keys_to_remove:
                    del self._causal_mask_cache[key]

        return self._causal_mask_cache[cache_key]

    def _get_dilation_pattern(
        self, seq_len: int, dilation_rate: int, offset: int = 0
    ) -> Tensor:
        """Get cached dilation pattern (from V2) - kept for compatibility."""
        cache_key = f"{seq_len}_{dilation_rate}_{offset}"

        # Check global pattern cache first
        if self.use_pattern_cache and self._pattern_cache is not None:
            if cache_key in self._pattern_cache:
                pattern = self._pattern_cache[cache_key]
                # Convert back to tensor if needed
                if not isinstance(pattern, torch.Tensor):
                    pattern = torch.tensor(pattern, device=self.device)
                elif pattern.device != self.device:
                    pattern = pattern.to(self.device)
                return pattern

        # Check local cache
        if cache_key not in self._dilation_pattern_cache:
            # Create dilation pattern
            if dilation_rate > 1:
                indices = torch.arange(
                    offset, seq_len, dilation_rate, device=self.device
                )
                if len(indices) < seq_len:
                    # Pad by cycling
                    repeats = (seq_len + len(indices) - 1) // len(indices)
                    extended = indices.repeat(repeats)
                    indices = extended[:seq_len]
            else:
                indices = torch.arange(seq_len, device=self.device)

            self._dilation_pattern_cache[cache_key] = indices

            # Store in global pattern cache
            if self.use_pattern_cache and self._pattern_cache is not None:
                full_key = f"dilation_{cache_key}"
                # Store on CPU to save GPU memory
                self._pattern_cache[full_key] = (
                    indices.cpu() if indices.is_cuda else indices
                )

        return self._dilation_pattern_cache[cache_key]


# Alias for compatibility
RingDilatedAttentionTrue = RingDilatedAttentionHybrid
