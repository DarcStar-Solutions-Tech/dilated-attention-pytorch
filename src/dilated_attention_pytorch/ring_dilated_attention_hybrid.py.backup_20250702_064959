"""
Ring Dilated Attention Hybrid - True ring attention with production features.

This implementation combines:
- V3's true ring communication (O(n/p) memory scaling)
- V2's dilation support and optimizations
- V3's LSE accumulation for numerical stability
- V2's memory pool, caching, and Flash Attention support
"""

import math
import warnings
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist

# Import ring utilities from V3
from .ring_attention_utils import (
    exists,
    default,
    all_ring_pass,
    split_by_rank,
    create_causal_mask,
    RingInfo,
)

# Import LSE utilities from V3
from .ring_attention_lse import (
    StableRingAccumulator,
    compute_attention_with_lse,
)

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
    - Full dilation support in multi-GPU mode (V2)
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

        # Core parameters
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout

        # Device and dtype configuration (V2 style with smart defaults)
        self.device = device or (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Smart dtype selection using gpu_utils (from V2)
        if dtype is not None:
            self.dtype = dtype
        else:
            # Try to use GPU utilities for optimal dtype selection
            try:
                from .utils.gpu_utils import get_optimal_dtype

                self.dtype = get_optimal_dtype(
                    self.device, prefer_fp16=True, warn_pascal=False
                )
            except ImportError:
                # Fallback to simple logic if gpu_utils not available
                self.dtype = (
                    torch.float16 if self.device.type == "cuda" else torch.float32
                )

        # Ring configuration
        if dist.is_initialized():
            self.ring_size = default(ring_size, dist.get_world_size())
            self.rank = dist.get_rank()
        else:
            self.ring_size = 1
            self.rank = 0

        # V2 optimization features
        self.use_flash_attention = use_flash_attention
        self.flash_backend = None
        self._can_use_flash = False

        # Initialize Flash Attention support (from V2)
        if self.use_flash_attention and HAS_FLASH_UTILS:
            try:
                self.flash_support = get_flash_attention_support(self.device)
                if self.flash_support["has_flash"]:
                    self.flash_backend = self.flash_support.get(
                        "best_backend", "standard"
                    )
                    self._can_use_flash = True
            except Exception as e:
                if self.rank == 0:
                    warnings.warn(f"Flash Attention initialization failed: {e}")

        # Memory pool integration (from V2)
        self.enable_memory_pool = enable_memory_pool and HAS_ENHANCED_MEMORY_POOL
        self.lightweight_pool = lightweight_pool
        self.memory_pool_threshold_mb = memory_pool_threshold_mb
        self._memory_pool = None

        if self.enable_memory_pool:
            if lightweight_pool:
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=False,
                    enable_bucketed=True,
                    enable_numa=False,
                    enable_profiling=enable_profiling,
                )
            else:
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=True,
                    enable_bucketed=True,
                    enable_numa=True,
                    enable_profiling=enable_profiling,
                )

        # Pattern caching (from V2)
        self.use_pattern_cache = use_pattern_cache and HAS_PATTERN_CACHE
        if self.use_pattern_cache:
            # Get global pattern cache - it's a plain dictionary
            self._pattern_cache = get_global_pattern_cache()
        else:
            # Fall back to local cache
            self._pattern_cache = None

        # Local caches (from V2)
        self._causal_mask_cache = {}
        self._dilation_pattern_cache = {}
        self._head_groups_cache = None

        # Pre-allocated buffers for ring communication
        self._kv_receive_buffer = None

        # Hardware-aware execution path (from V2)
        self._determine_execution_path()

    def _determine_execution_path(self):
        """Determine optimal execution path based on hardware (from V2)."""
        self._use_direct_sdpa = False
        self._skip_flash_attempt = False

        if self.device.type == "cuda":
            compute_capability = torch.cuda.get_device_capability(self.device)
            cc_major, cc_minor = compute_capability

            # For older GPUs, use SDPA directly
            if cc_major < 8:  # Pre-Ampere
                self._skip_flash_attempt = True
                self._use_direct_sdpa = True

                if self.rank == 0:
                    print(
                        f"RingDilatedAttentionHybrid: Using SDPA path for CC {cc_major}.{cc_minor}"
                    )

    def _calculate_head_groups(self, num_heads: int) -> list[int]:
        """Calculate head group distribution (from both V2 and V3)."""
        if (
            self._head_groups_cache is not None
            and sum(self._head_groups_cache) == num_heads
        ):
            return self._head_groups_cache

        num_segments = len(self.segment_lengths)
        base_heads = num_heads // num_segments
        extra_heads = num_heads % num_segments

        head_groups = [base_heads] * num_segments
        for i in range(extra_heads):
            head_groups[-(i + 1)] += 1

        self._head_groups_cache = head_groups
        return head_groups

    def _get_dilation_pattern(
        self, seq_len: int, dilation_rate: int, offset: int
    ) -> Tensor:
        """Get cached dilation pattern (from V2)."""
        cache_key = (seq_len, dilation_rate, offset)

        # Check global pattern cache first (it's a plain dictionary)
        if self.use_pattern_cache and self._pattern_cache is not None:
            full_key = f"dilation_{cache_key}"
            if full_key in self._pattern_cache:
                pattern = self._pattern_cache[full_key]
                # Move to correct device if needed
                if isinstance(pattern, torch.Tensor) and pattern.device != self.device:
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

    def _get_causal_mask(
        self, seq_len_q: int, seq_len_kv: int, chunk_offset: int = 0
    ) -> Tensor:
        """Get cached causal mask (from V2)."""
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

    def _apply_dilation_to_tensor(
        self, tensor: Tensor, is_query: bool = False
    ) -> Tensor:
        """Apply dilation patterns to tensor based on head groups (from V2)."""
        b, n, h, d = tensor.shape
        heads_per_group = self._calculate_head_groups(h)

        # Allocate output
        if self.enable_memory_pool and self._memory_pool is not None:
            output = self._memory_pool.allocate(
                tensor.shape, dtype=tensor.dtype, device=tensor.device
            )
        else:
            output = torch.empty_like(tensor)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Get dilation pattern
            offset = i % dilation_rate if dilation_rate > 1 else 0
            pattern = self._get_dilation_pattern(n, dilation_rate, offset)

            # Apply to head group
            tensor_group = tensor[:, :, head_start:head_end, :]
            output[:, :, head_start:head_end, :] = tensor_group.index_select(1, pattern)

            head_start = head_end

        return output

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
        """
        b, n, h, d = q.shape

        # Single device fallback
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Ensure sequence length is divisible by ring size
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )

        # TRUE RING ATTENTION: Split K,V across ranks (V3 approach)
        # Each GPU only stores 1/p of the K,V tensors
        k_local = split_by_rank(k, self.rank, self.ring_size)
        v_local = split_by_rank(v, self.rank, self.ring_size)

        # CRITICAL: Apply dilation patterns BEFORE ring communication (V2 feature)
        # This enables dilation > 1 in multi-GPU mode
        k_local_dilated = self._apply_dilation_to_tensor(k_local)
        v_local_dilated = self._apply_dilation_to_tensor(v_local)

        # Apply dilation to full Q (needed for all chunks)
        q_dilated = self._apply_dilation_to_tensor(q, is_query=True)

        # Stack K,V for ring passing
        kv_local = torch.stack((k_local_dilated, v_local_dilated))

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

        # Transpose Q for attention computation
        q_transposed = q_dilated.transpose(1, 2)  # (b, h, n, d)

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

            # Calculate chunk position in global sequence
            chunk_size = n // self.ring_size
            chunk_idx = ring_info.ring_rank
            chunk_start = chunk_idx * chunk_size

            # Compute attention for this chunk
            chunk_output, chunk_lse = self._compute_chunk_attention(
                q_transposed,
                k_chunk,
                v_chunk,
                chunk_start,
                chunk_size,
                is_causal,
                ring_info,
            )

            # Accumulate with LSE
            accumulator.update(chunk_output, chunk_lse)

        # Get final output and transpose back
        output = accumulator.get_output().transpose(1, 2)  # (b, n, h, d)

        return output

    def _compute_chunk_attention(
        self,
        q: Tensor,  # (b, h, n, d) - full Q
        k_chunk: Tensor,  # (b, chunk_size, h, d) - chunk of K
        v_chunk: Tensor,  # (b, chunk_size, h, d) - chunk of V
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
        ring_info: RingInfo,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention for one chunk with V2's optimizations."""
        # Transpose K,V chunks to attention format
        k_chunk = k_chunk.transpose(1, 2)  # (b, h, chunk_size, d)
        v_chunk = v_chunk.transpose(1, 2)  # (b, h, chunk_size, d)

        # Try Flash Attention first (V2 optimization)
        if self._can_use_flash and not self._skip_flash_attempt:
            try:
                # Adjust causal for chunk
                adjusted_causal = is_causal and chunk_start == 0

                # Use Flash Attention
                output = flash_attention_forward(
                    q.transpose(1, 2),  # Flash expects (b, n, h, d)
                    k_chunk.transpose(1, 2),
                    v_chunk.transpose(1, 2),
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=adjusted_causal,
                    backend=self.flash_backend,
                )

                # Compute LSE separately for accumulation
                # This is a limitation but necessary for ring accumulation
                scores = torch.matmul(q, k_chunk.transpose(-2, -1)) / math.sqrt(
                    q.shape[-1]
                )
                if is_causal:
                    mask = self._get_causal_mask(q.shape[2], chunk_size, chunk_start)
                    scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                lse = scores.logsumexp(dim=-1, keepdim=True).squeeze(-1)

                return output.transpose(1, 2), lse

            except Exception:
                pass  # Fall through to standard computation

        # Standard computation with LSE (V3 approach)
        mask = None
        if is_causal:
            mask = self._get_causal_mask(q.shape[2], chunk_size, chunk_start)

        output, lse = compute_attention_with_lse(
            q,
            k_chunk,
            v_chunk,
            scale=1.0 / math.sqrt(q.shape[-1]),
            mask=mask,
            dropout=self.dropout,
            training=self.training,
        )

        return output, lse

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward with dilation patterns."""
        # Apply dilation patterns
        q_dilated = self._apply_dilation_to_tensor(q, is_query=True)
        k_dilated = self._apply_dilation_to_tensor(k)
        v_dilated = self._apply_dilation_to_tensor(v)

        # Use Flash Attention if available
        if self._can_use_flash:
            try:
                output = flash_attention_forward(
                    q_dilated,
                    k_dilated,
                    v_dilated,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    backend=self.flash_backend,
                )
                return output
            except Exception:
                pass

        # Fall back to standard attention
        q_t = q_dilated.transpose(1, 2)
        k_t = k_dilated.transpose(1, 2)
        v_t = v_dilated.transpose(1, 2)

        # Compute attention
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(q.shape[-1])

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(q.shape[1], k.shape[1], device=q.device), diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        output = torch.matmul(attn_weights, v_t)
        return output.transpose(1, 2)


# Alias for compatibility
RingDilatedAttentionTrue = RingDilatedAttentionHybrid
