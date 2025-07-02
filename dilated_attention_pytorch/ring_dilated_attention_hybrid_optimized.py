"""
Ring Dilated Attention Hybrid Optimized - Properly integrated V2 optimizations.

This implementation fixes the performance issues in the original hybrid by:
- Properly initializing and using memory pools
- Implementing single-GPU fast path
- Batch processing segments when possible
- Pre-computing and caching patterns efficiently
- Using optimized memory access patterns
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

# Import Flash Attention utilities
try:
    from .utils.flash_attention_utils import (
        flash_attention_forward,
        get_flash_attention_support,
    )

    HAS_FLASH_UTILS = True
except ImportError:
    HAS_FLASH_UTILS = False

# Import optimized attention computation
try:
    from .utils.attention_utils import optimize_attention_computation

    HAS_OPTIMIZE_ATTENTION = True
except ImportError:
    HAS_OPTIMIZE_ATTENTION = False


class RingDilatedAttentionHybridOptimized(nn.Module):
    """
    Optimized Hybrid Ring Dilated Attention with proper V2 integration.

    Key optimizations:
    - Properly initialized memory pools and pattern caches
    - Single-GPU fast path that bypasses ring infrastructure
    - Batch segment processing for better GPU utilization
    - Pre-computed patterns with efficient caching
    - Optimized memory access patterns using views instead of index_select
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
        # Optimization settings
        batch_segments: bool = True,
        precompute_patterns: bool = True,
    ):
        """Initialize Optimized Hybrid Ring Dilated Attention."""
        super().__init__()

        assert len(segment_lengths) == len(dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout if dropout is not None else 0.0
        self.batch_segments = batch_segments
        self.precompute_patterns = precompute_patterns

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
        self.enable_memory_pool = enable_memory_pool
        self.enable_profiling = enable_profiling
        self.lightweight_pool = lightweight_pool
        self.use_pattern_cache = use_pattern_cache
        self.memory_pool_threshold_mb = memory_pool_threshold_mb
        self.use_flash_attention = use_flash_attention

        # Initialize memory pool FIRST
        self._memory_pool = None
        if self.enable_memory_pool and HAS_ENHANCED_MEMORY_POOL:
            try:
                self._memory_pool = get_enhanced_memory_pool(
                    enable_profiling=self.enable_profiling,
                )
                if self.rank == 0:
                    print(
                        "RingDilatedAttentionHybridOptimized: Memory pool initialized"
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
                        "RingDilatedAttentionHybridOptimized: Pattern cache initialized"
                    )
            except Exception as e:
                if self.rank == 0:
                    print(f"Failed to initialize pattern cache: {e}")
                self._pattern_cache = None

        # Local caches (always used)
        self._dilation_pattern_cache = {}
        self._causal_mask_cache = {}
        self._precomputed_patterns = {}

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

        # Smart dtype selection based on GPU
        if self.dtype is None:
            try:
                from .utils.gpu_utils import get_optimal_dtype

                self.dtype = get_optimal_dtype(
                    self.device, prefer_fp16=True, warn_pascal=False
                )
            except:
                self.dtype = (
                    torch.float16 if torch.cuda.is_available() else torch.float32
                )

        # Pre-allocate buffers for ring communication
        self._kv_receive_buffer = None

        # Dropout
        self.dropout_p = dropout

        # Pre-compute patterns if requested
        if self.precompute_patterns:
            self._precompute_all_patterns()

    def _precompute_all_patterns(self):
        """Pre-compute all dilation patterns to avoid runtime overhead."""
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
        # Use optimized pattern generation
        indices = torch.arange(offset, seg_len, dilation_rate, device=self.device)
        if indices.shape[0] < seg_len // dilation_rate:
            # Pad if necessary
            pad_size = seg_len // dilation_rate - indices.shape[0]
            indices = torch.cat([indices, indices[:pad_size]])
        return indices[: seg_len // dilation_rate]

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with optimizations.
        """
        b, n, h, d = q.shape

        # Single device fast path
        if self.ring_size == 1:
            return self._optimized_single_device_forward(q, k, v, is_causal)

        # Multi-GPU ring attention
        return self._ring_forward_with_dilated_segments(q, k, v, is_causal)

    def _optimized_single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Optimized single device forward with batch processing."""
        b, n, h, d = q.shape

        # Use memory pool for output
        if self._memory_pool is not None:
            try:
                output = self._memory_pool.allocate(
                    (b, n, h, d), dtype=q.dtype, device=q.device
                )
                output.zero_()
            except:
                output = torch.zeros(b, n, h, d, device=q.device, dtype=q.dtype)
        else:
            output = torch.zeros(b, n, h, d, device=q.device, dtype=q.dtype)

        # Calculate head groups
        heads_per_group = self._calculate_head_groups(h)
        head_start = 0

        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Process this head group efficiently
            if self.batch_segments and HAS_OPTIMIZE_ATTENTION:
                # Use optimized attention computation
                group_output = self._batch_process_segments_optimized(
                    q[:, :, head_start:head_end],
                    k[:, :, head_start:head_end],
                    v[:, :, head_start:head_end],
                    segment_len,
                    dilation_rate,
                    i,
                    is_causal,
                )
            else:
                # Fall back to segment-by-segment processing
                group_output = self._process_segments_standard(
                    q[:, :, head_start:head_end],
                    k[:, :, head_start:head_end],
                    v[:, :, head_start:head_end],
                    segment_len,
                    dilation_rate,
                    i,
                    is_causal,
                )

            output[:, :, head_start:head_end] = group_output
            head_start = head_end

        return output

    def _batch_process_segments_optimized(
        self,
        q_group: Tensor,  # (b, n, gh, d)
        k_group: Tensor,
        v_group: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset_idx: int,
        is_causal: bool,
    ) -> Tensor:
        """Process all segments in a batch using optimized attention."""
        b, n, gh, d = q_group.shape

        if dilation_rate == 1:
            # No dilation needed, process entire sequence
            q_t = q_group.transpose(1, 2)  # (b, gh, n, d)
            k_t = k_group.transpose(1, 2)
            v_t = v_group.transpose(1, 2)

            # Use optimized attention computation
            output = optimize_attention_computation(
                q_t,
                k_t,
                v_t,
                is_causal=is_causal,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

            return output.transpose(1, 2)  # (b, n, gh, d)

        # With dilation, we need segment processing
        num_segments = (n + segment_len - 1) // segment_len
        output = torch.zeros_like(q_group)

        # Process segments in parallel when possible
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_len
            seg_end = min(seg_start + segment_len, n)
            actual_seg_len = seg_end - seg_start

            # Get segments
            q_seg = q_group[:, seg_start:seg_end]
            k_seg = k_group[:, seg_start:seg_end]
            v_seg = v_group[:, seg_start:seg_end]

            # Apply dilation using pre-computed pattern
            if dilation_rate > 1:
                offset = offset_idx % dilation_rate
                pattern_key = (actual_seg_len, dilation_rate, offset)

                if pattern_key in self._precomputed_patterns:
                    pattern = self._precomputed_patterns[pattern_key]
                else:
                    pattern = self._create_dilation_pattern(
                        actual_seg_len, dilation_rate, offset
                    )

                # Use index_select for correct indexing
                q_seg = q_seg.index_select(1, pattern)
                k_seg = k_seg.index_select(1, pattern)
                v_seg = v_seg.index_select(1, pattern)

            # Compute attention
            q_seg_t = q_seg.transpose(1, 2)
            k_seg_t = k_seg.transpose(1, 2)
            v_seg_t = v_seg.transpose(1, 2)

            seg_output = optimize_attention_computation(
                q_seg_t,
                k_seg_t,
                v_seg_t,
                is_causal=is_causal and seg_idx == 0,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

            seg_output = seg_output.transpose(1, 2)

            # Write back
            if dilation_rate > 1:
                # Write back to dilated positions
                for i, idx in enumerate(pattern):
                    if seg_start + idx < n:
                        output[:, seg_start + idx] = seg_output[:, i]
            else:
                output[:, seg_start:seg_end] = seg_output

        return output

    def _process_segments_standard(
        self,
        q_group: Tensor,
        k_group: Tensor,
        v_group: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset_idx: int,
        is_causal: bool,
    ) -> Tensor:
        """Standard segment processing (fallback)."""
        b, n, gh, d = q_group.shape
        output = torch.zeros_like(q_group)

        # Transpose for attention format
        q_t = q_group.transpose(1, 2)  # (b, gh, n, d)
        k_t = k_group.transpose(1, 2)
        v_t = v_group.transpose(1, 2)

        num_segments = (n + segment_len - 1) // segment_len

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_len
            seg_end = min(seg_start + segment_len, n)
            actual_seg_len = seg_end - seg_start

            # Get segment
            q_seg = q_t[:, :, seg_start:seg_end]
            k_seg = k_t[:, :, seg_start:seg_end]
            v_seg = v_t[:, :, seg_start:seg_end]

            # Apply dilation if needed
            if dilation_rate > 1:
                offset = offset_idx % dilation_rate
                pattern = self._get_segment_dilation_pattern(
                    actual_seg_len, dilation_rate, offset
                )

                q_seg = q_seg.index_select(2, pattern)
                k_seg = k_seg.index_select(2, pattern)
                v_seg = v_seg.index_select(2, pattern)

            # Compute attention
            if HAS_OPTIMIZED_LSE:
                seg_output, _ = compute_attention_with_lse_optimized(
                    q_seg,
                    k_seg,
                    v_seg,
                    scale=1.0 / math.sqrt(d),
                    dropout=self.dropout_p,
                    training=self.training,
                    is_causal=is_causal and seg_idx == 0,
                )
            else:
                seg_output, _ = compute_attention_with_lse(
                    q_seg,
                    k_seg,
                    v_seg,
                    scale=1.0 / math.sqrt(d),
                    dropout=self.dropout_p,
                    training=self.training,
                    is_causal=is_causal and seg_idx == 0,
                )

            # Write back
            if dilation_rate > 1:
                for i, idx in enumerate(pattern):
                    output[:, seg_start + idx, :, :] = seg_output[:, :, i, :].transpose(
                        1, 2
                    )
            else:
                output[:, seg_start:seg_end, :, :] = seg_output.transpose(1, 2)

        return output

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

    def _ring_forward_with_dilated_segments(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Ring attention with optimized memory usage."""
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

        # Pre-allocate receive buffer if needed
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

        # Ring attention with optimized chunk processing
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
            chunk_output, chunk_lse = self._compute_dilated_chunk_attention_optimized(
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

    def _compute_dilated_chunk_attention_optimized(
        self,
        q: Tensor,
        k_chunk: Tensor,
        v_chunk: Tensor,
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute chunk attention with optimizations."""
        b, n, h, d = q.shape

        # Use memory pool for allocations
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

        # Process head groups with batch optimization
        heads_per_group = self._calculate_head_groups(h)
        head_start = 0

        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Process this head group
            group_output, group_lse = self._process_head_group_segments_optimized(
                q.transpose(1, 2)[:, head_start:head_end],
                k_chunk.transpose(1, 2)[:, head_start:head_end],
                v_chunk.transpose(1, 2)[:, head_start:head_end],
                segment_len,
                dilation_rate,
                i,
                chunk_start,
                chunk_size,
                is_causal,
            )

            output[:, head_start:head_end] = group_output
            lse[:, head_start:head_end] = group_lse
            head_start = head_end

        return output, lse

    def _process_head_group_segments_optimized(
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
        """Process head group segments with optimizations."""
        b, gh, n, d = q_group.shape

        # Allocate output using memory pool
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

            # Determine overlap between segment and chunk
            overlap_start = max(seg_start, chunk_start)
            overlap_end = min(seg_end, chunk_start + chunk_size)

            if overlap_start >= overlap_end:
                continue

            # Get query segment
            q_seg = q_group[:, :, seg_start:seg_end]

            # Get overlapping K,V from chunk
            k_overlap = k_chunk_group[
                :, :, overlap_start - chunk_start : overlap_end - chunk_start
            ]
            v_overlap = v_chunk_group[
                :, :, overlap_start - chunk_start : overlap_end - chunk_start
            ]

            # Apply dilation if needed
            if dilation_rate > 1:
                # Use optimized dilation pattern
                offset = offset_idx % dilation_rate
                q_pattern = self._get_segment_dilation_pattern(
                    seg_end - seg_start, dilation_rate, offset
                )

                # Map pattern to overlap
                overlap_pattern = self._map_pattern_to_overlap_optimized(
                    q_pattern, seg_start, overlap_start, overlap_end - overlap_start
                )

                if overlap_pattern is not None and overlap_pattern.numel() > 0:
                    q_seg_dilated = q_seg.index_select(2, q_pattern)
                    k_overlap_dilated = k_overlap.index_select(2, overlap_pattern)
                    v_overlap_dilated = v_overlap.index_select(2, overlap_pattern)

                    # Compute attention
                    seg_output, seg_lse = self._compute_segment_attention_optimized(
                        q_seg_dilated,
                        k_overlap_dilated,
                        v_overlap_dilated,
                        seg_start,
                        overlap_start,
                        is_causal,
                    )

                    # Write back dilated results
                    for i, idx in enumerate(q_pattern):
                        if seg_start + idx < n:
                            output[:, :, seg_start + idx] = seg_output[:, :, i]
                            lse[:, :, seg_start + idx] = seg_lse[:, :, i]
            else:
                # No dilation
                seg_output, seg_lse = self._compute_segment_attention_optimized(
                    q_seg, k_overlap, v_overlap, seg_start, overlap_start, is_causal
                )

                output[:, :, seg_start:seg_end] = seg_output
                lse[:, :, seg_start:seg_end] = seg_lse

        return output, lse

    def _map_pattern_to_overlap_optimized(
        self,
        pattern: Tensor,
        seg_start: int,
        overlap_start: int,
        overlap_len: int,
    ) -> Optional[Tensor]:
        """Optimized pattern mapping."""
        # Use vectorized operations
        global_positions = seg_start + pattern
        mask = (global_positions >= overlap_start) & (
            global_positions < overlap_start + overlap_len
        )
        valid_indices = pattern[mask] - (overlap_start - seg_start)

        return valid_indices if valid_indices.numel() > 0 else None

    def _compute_segment_attention_optimized(
        self,
        q_seg: Tensor,
        k_seg: Tensor,
        v_seg: Tensor,
        seg_start: int,
        overlap_start: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention with optimal backend."""
        scale = 1.0 / math.sqrt(q_seg.shape[-1])

        # Use optimized attention computation if available
        if HAS_OPTIMIZE_ATTENTION:
            output = optimize_attention_computation(
                q_seg,
                k_seg,
                v_seg,
                is_causal=is_causal and seg_start == 0,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

            # Compute LSE for accumulation
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale
            if is_causal and seg_start == overlap_start:
                scores = scores.masked_fill(
                    torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1),
                    float("-inf"),
                )
            lse = scores.logsumexp(dim=-1)

            return output, lse

        # Fallback to LSE computation
        if HAS_OPTIMIZED_LSE:
            return compute_attention_with_lse_optimized(
                q_seg,
                k_seg,
                v_seg,
                scale=scale,
                dropout=self.dropout_p,
                training=self.training,
                is_causal=is_causal and seg_start == 0,
            )
        else:
            return compute_attention_with_lse(
                q_seg,
                k_seg,
                v_seg,
                scale=scale,
                dropout=self.dropout_p,
                training=self.training,
                is_causal=is_causal and seg_start == 0,
            )


# Alias for backward compatibility
RingDilatedAttentionHybrid = RingDilatedAttentionHybridOptimized
