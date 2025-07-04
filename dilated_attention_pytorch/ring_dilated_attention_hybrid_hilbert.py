"""
Ring Dilated Attention Hybrid with Hilbert Ordering.

This implementation adds Hilbert curve memory ordering to the efficient
RingDilatedAttentionHybridOptimizedV2 that achieves 262K+ tokens.

Key features maintained:
- O(n/p) memory scaling with proper ring passing
- No all_gather operations
- Efficient chunk handling
- Online softmax with LSE accumulation
- Memory pool support

Hilbert enhancement:
- Apply Hilbert ordering to K,V chunks for better cache efficiency
- Maintain all memory-efficient properties of the original
"""

from typing import Optional, Tuple, List, Any
from functools import partial

import torch
from torch import Tensor

# Import the parent class we're extending
from .ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)

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


class RingDilatedAttentionHybridHilbert(RingDilatedAttentionHybridOptimizedV2):
    """
    Hilbert-enhanced Ring Dilated Attention maintaining O(n/p) memory.

    This adds Hilbert curve ordering to K,V chunks while preserving all
    the memory-efficient properties that enable 262K+ token sequences.
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
        hilbert_chunk_size: int = 4096,
        # DilatedAttention options
        use_xformers: bool = True,
        attention_op: Optional[Any] = None,
    ):
        """Initialize Hilbert Ring Dilated Attention."""
        # Initialize parent class with all required parameters
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

        # Hilbert-specific settings (parent handles everything else)
        self.use_hilbert = use_hilbert
        self.hilbert_chunk_size = hilbert_chunk_size
        self._hilbert_cache = {}

        # Create DilatedAttention instance for single-device computation
        from .dilated_attention import DilatedAttention

        self.dilated_attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
            op=attention_op if use_xformers else None,
            enable_memory_pool=enable_memory_pool,
            enable_profiling=enable_profiling,
            lightweight_pool=True,  # Use lightweight pool for performance
        )

    def _generate_hilbert_curve(self, n: int) -> torch.Tensor:
        """Generate Hilbert curve mapping for size n."""
        if n in self._hilbert_cache:
            return self._hilbert_cache[n]

        # For small sizes, use identity
        if n <= 64:
            mapping = torch.arange(n, dtype=torch.long, device=self.device)
            self._hilbert_cache[n] = mapping
            return mapping

        # Find power of 2 size
        size = 1
        while size * size < n:
            size *= 2

        def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
            x = y = 0
            s = 1
            while s < n:
                rx = 1 if (d // 2) & 1 else 0
                ry = 1 if (d ^ rx) & 1 else 0
                if ry == 0:
                    if rx == 1:
                        x, y = n - 1 - x, n - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d //= 4
                s *= 2
            return x, y

        # Generate mapping
        mapping = torch.zeros(n, dtype=torch.long, device=self.device)
        hilbert_to_linear = {}

        for i in range(min(n, size * size)):
            x, y = hilbert_d2xy(size, i)
            linear_idx = y * size + x
            if linear_idx < n:
                hilbert_to_linear[i] = linear_idx

        # Fill mapping
        hilbert_idx = 0
        for i in sorted(hilbert_to_linear.keys()):
            linear_idx = hilbert_to_linear[i]
            mapping[linear_idx] = hilbert_idx
            hilbert_idx += 1

        self._hilbert_cache[n] = mapping
        return mapping

    def _apply_hilbert_to_chunk(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Hilbert ordering to a chunk of K or V."""
        if not self.use_hilbert:
            return tensor

        b, n, h, d = tensor.shape

        # Apply Hilbert in chunks if sequence is long
        if n > self.hilbert_chunk_size:
            num_chunks = (n + self.hilbert_chunk_size - 1) // self.hilbert_chunk_size
            result = torch.empty_like(tensor)

            for i in range(num_chunks):
                start = i * self.hilbert_chunk_size
                end = min(start + self.hilbert_chunk_size, n)
                chunk_len = end - start

                # Get Hilbert mapping for this chunk
                mapping = self._generate_hilbert_curve(chunk_len)

                # Apply to chunk
                result[:, start:end] = tensor[:, start:end][:, mapping]

            return result
        else:
            # Apply Hilbert to entire sequence
            mapping = self._generate_hilbert_curve(n)
            return tensor[:, mapping]

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass with Hilbert-enhanced ring attention."""
        b, n, h, d = q.shape

        # Single device fallback
        if self.ring_size == 1:
            return self._single_device_forward(q, k, v, is_causal)

        # Multi-GPU ring attention with Hilbert
        return self._ring_forward_with_hilbert(q, k, v, is_causal)

    def _single_device_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Single device forward with Hilbert ordering."""
        # Apply Hilbert to K,V
        k_hilbert = self._apply_hilbert_to_chunk(k)
        v_hilbert = self._apply_hilbert_to_chunk(v)

        # Use the DilatedAttention module which includes:
        # - Memory pooling
        # - Pattern caching
        # - xFormers/Flash Attention support
        # - Proper dilated attention computation
        output = self.dilated_attention(
            query=q,
            key=k_hilbert,
            value=v_hilbert,
            is_causal=is_causal,
        )

        return output

    def _calculate_head_groups(self, num_heads: int) -> List[int]:
        """Calculate how many heads belong to each segment/dilation configuration."""
        num_groups = len(self.segment_lengths)
        base_heads = num_heads // num_groups
        extra_heads = num_heads % num_groups

        heads_per_group = [base_heads] * num_groups
        for i in range(extra_heads):
            heads_per_group[i] += 1

        return heads_per_group

    def _ring_forward_with_hilbert(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """
        Ring attention with Hilbert ordering applied to K,V chunks.

        Key insight: Apply Hilbert ordering BEFORE splitting into chunks
        to maintain spatial locality benefits across the entire sequence.
        """
        b, n, h, d = q.shape

        # Ensure divisibility
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )

        # Apply Hilbert ordering to K,V BEFORE splitting
        # This ensures Hilbert curve spans the full sequence
        k_hilbert = self._apply_hilbert_to_chunk(k) if self.use_hilbert else k
        v_hilbert = self._apply_hilbert_to_chunk(v) if self.use_hilbert else v

        # Split K,V across GPUs (now in Hilbert space)
        chunk_size = n // self.ring_size
        k_local = split_by_rank(k_hilbert, self.rank, self.ring_size)
        v_local = split_by_rank(v_hilbert, self.rank, self.ring_size)

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
                except Exception:
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
            # K,V are already in Hilbert space, improving cache efficiency
            chunk_output, chunk_lse = self._compute_dilated_chunk_attention_hilbert(
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

    def _compute_dilated_chunk_attention_hilbert(
        self,
        q: Tensor,
        k_chunk: Tensor,
        v_chunk: Tensor,
        chunk_start: int,
        chunk_size: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention with K,V already in Hilbert space.

        This is identical to the parent implementation since Hilbert
        ordering is already applied. The cache efficiency benefits
        come from the improved spatial locality of the reordered data.
        """
        # Use parent's implementation directly since we properly inherit
        return super()._compute_dilated_chunk_attention_fixed(
            q, k_chunk, v_chunk, chunk_start, chunk_size, is_causal
        )
