"""
Ring Dilated Attention V3 - With optimized pattern caching.

This version includes optimized pattern transfer to eliminate the performance
regression seen with CPU→GPU transfers in V2.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

try:
    from .core.enhanced_memory_pool import get_enhanced_memory_pool

    HAS_ENHANCED_MEMORY_POOL = True
except ImportError:
    HAS_ENHANCED_MEMORY_POOL = False
    warnings.warn("Enhanced memory pool not available")

try:
    from .core.optimized_pattern_cache import get_optimized_pattern_cache

    HAS_OPTIMIZED_CACHE = True
except ImportError:
    HAS_OPTIMIZED_CACHE = False
    warnings.warn("Optimized pattern cache not available")


class RingDilatedAttentionV3(nn.Module):
    """
    Ring Dilated Attention with optimized pattern caching.

    Key improvements over V2:
    - GPU-resident pattern cache for frequently accessed patterns
    - Batch pattern transfers for efficiency
    - Adaptive tier management (hot patterns stay on GPU)
    - Async transfers with prefetching
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_memory_pool: bool = False,
        enable_profiling: bool = False,
        lightweight_pool: bool = True,
        use_pattern_cache: bool = True,
        cache_on_gpu: bool = True,  # New: keep hot patterns on GPU
    ):
        super().__init__()

        assert len(segment_lengths) == len(dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Ring configuration
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.ring_size = ring_size or self.world_size

        # For single-GPU operation, we can simulate any ring size
        if self.world_size == 1 and self.ring_size > 1:
            self.mode = "simulated"
        elif self.world_size > 1:
            self.mode = "distributed"
            self.ring_size = min(self.ring_size, self.world_size)
        else:
            self.mode = "single"

        # Pre-allocated communication buffers for distributed mode
        self._kv_send_buffer = None
        self._kv_recv_buffer = None

        # Pattern caching setup
        self.use_pattern_cache = use_pattern_cache and HAS_OPTIMIZED_CACHE
        self.cache_on_gpu = cache_on_gpu

        if self.use_pattern_cache:
            # Use optimized pattern cache with GPU support
            self._pattern_cache = get_optimized_pattern_cache(
                max_gpu_patterns=20,  # Keep top 20 patterns on GPU
                max_cpu_patterns=100,
                gpu_memory_limit_mb=50.0,  # 50MB limit for patterns
                enable_async=True,
                enable_prefetch=True,
            )

            # Pre-generate and cache common patterns on initialization
            if self.cache_on_gpu:
                self._pregenerrate_patterns()
        else:
            # Fall back to local cache
            self._dilated_indices_cache = {}

        # Enhanced memory pool integration
        self.enable_memory_pool = enable_memory_pool and HAS_ENHANCED_MEMORY_POOL
        self.lightweight_pool = lightweight_pool
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

    def _pregenerrate_patterns(self):
        """Pre-generate common patterns and store them on GPU."""
        if not self.use_pattern_cache:
            return

        device = self.device if self.cache_on_gpu else torch.device("cpu")

        # Pre-generate patterns for each segment/dilation combination
        for seg_len, dil_rate in zip(self.segment_lengths, self.dilation_rates):
            for offset in range(min(dil_rate, 4)):  # Generate first 4 offsets
                cache_key = f"ring_dilated_s{seg_len}_r{dil_rate}_off{offset}"

                # Generate pattern
                if dil_rate > 1:
                    dilated_indices = torch.arange(
                        offset, seg_len, dil_rate, device=device
                    )
                    if len(dilated_indices) < seg_len:
                        repeats = (seg_len + len(dilated_indices) - 1) // len(
                            dilated_indices
                        )
                        extended = dilated_indices.repeat(repeats)
                        dilated_indices = extended[:seg_len]
                    dilated_indices = dilated_indices % seg_len
                else:
                    dilated_indices = torch.arange(0, seg_len, device=device)

                # Store in cache (on GPU if enabled)
                self._pattern_cache.put(
                    cache_key,
                    dilated_indices,
                    store_on_gpu=self.cache_on_gpu,
                )

                # Pin frequently used patterns
                if offset == 0:  # Most common offset
                    self._pattern_cache.pin_pattern(cache_key, device)

    def _apply_dilation(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dilation_rate: int,
        offset: int,
        segment_len: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply dilation pattern to the segment tensors with optimized caching."""
        device = q.device

        if self.use_pattern_cache:
            # Use optimized pattern cache
            cache_key = f"ring_dilated_s{segment_len}_r{dilation_rate}_off{offset}"

            # Prefetch next offset pattern
            next_offset = (offset + 1) % dilation_rate if dilation_rate > 1 else 0
            prefetch_key = (
                f"ring_dilated_s{segment_len}_r{dilation_rate}_off{next_offset}"
            )

            # Get pattern with prefetching
            dilated_indices = self._pattern_cache.get(
                cache_key,
                target_device=device,
                prefetch_next=prefetch_key if offset < dilation_rate - 1 else None,
            )

            if dilated_indices is None:
                # Generate pattern if not cached
                if dilation_rate > 1:
                    dilated_indices = torch.arange(
                        offset, segment_len, dilation_rate, device=device
                    )
                    if len(dilated_indices) < segment_len:
                        repeats = (segment_len + len(dilated_indices) - 1) // len(
                            dilated_indices
                        )
                        extended = dilated_indices.repeat(repeats)
                        dilated_indices = extended[:segment_len]
                    dilated_indices = dilated_indices % segment_len
                else:
                    dilated_indices = torch.arange(0, segment_len, device=device)

                # Store in cache (on GPU if frequently accessed)
                self._pattern_cache.put(
                    cache_key,
                    dilated_indices,
                    store_on_gpu=(device.type == "cuda" and self.cache_on_gpu),
                )
        else:
            # Use local cache (original implementation)
            cache_key = (segment_len, dilation_rate, offset, device)

            if cache_key not in self._dilated_indices_cache:
                indices = torch.arange(0, segment_len, device=device)
                if dilation_rate > 1:
                    dilated_indices = torch.arange(
                        offset, segment_len, dilation_rate, device=device
                    )
                    if len(dilated_indices) < segment_len:
                        repeats = (segment_len + len(dilated_indices) - 1) // len(
                            dilated_indices
                        )
                        extended = dilated_indices.repeat(repeats)
                        dilated_indices = extended[:segment_len]

                    self._dilated_indices_cache[cache_key] = (
                        dilated_indices % segment_len
                    )
                else:
                    self._dilated_indices_cache[cache_key] = indices

            dilated_indices = self._dilated_indices_cache[cache_key]

        # Apply dilated indexing to K and V
        k_dilated = k.index_select(
            2, dilated_indices
        )  # [b, num_segments, segment_len, h, d]
        v_dilated = v.index_select(2, dilated_indices)

        return q, k_dilated, v_dilated

    def _apply_dilation_batch(
        self,
        segments: list[tuple[Tensor, Tensor, Tensor, int, int, int]],
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """
        Apply dilation to multiple segments in batch for efficiency.

        Args:
            segments: List of (q, k, v, dilation_rate, offset, segment_len) tuples

        Returns:
            List of (q, k_dilated, v_dilated) tuples
        """
        if not self.use_pattern_cache:
            # Fall back to individual processing
            return [
                self._apply_dilation(q, k, v, dr, off, sl)
                for q, k, v, dr, off, sl in segments
            ]

        # Collect pattern keys
        keys = []
        for _, _, _, dilation_rate, offset, segment_len in segments:
            cache_key = f"ring_dilated_s{segment_len}_r{dilation_rate}_off{offset}"
            keys.append(cache_key)

        # Batch fetch patterns
        device = segments[0][0].device
        patterns = self._pattern_cache.get_batch(keys, target_device=device)

        # Apply patterns
        results = []
        for i, (q, k, v, dilation_rate, offset, segment_len) in enumerate(segments):
            cache_key = keys[i]
            dilated_indices = patterns.get(cache_key)

            if dilated_indices is None:
                # Generate missing pattern
                if dilation_rate > 1:
                    dilated_indices = torch.arange(
                        offset, segment_len, dilation_rate, device=device
                    )
                    if len(dilated_indices) < segment_len:
                        repeats = (segment_len + len(dilated_indices) - 1) // len(
                            dilated_indices
                        )
                        extended = dilated_indices.repeat(repeats)
                        dilated_indices = extended[:segment_len]
                    dilated_indices = dilated_indices % segment_len
                else:
                    dilated_indices = torch.arange(0, segment_len, device=device)

                # Cache for next time
                self._pattern_cache.put(
                    cache_key,
                    dilated_indices,
                    store_on_gpu=(device.type == "cuda" and self.cache_on_gpu),
                )

            # Apply dilation
            k_dilated = k.index_select(2, dilated_indices)
            v_dilated = v.index_select(2, dilated_indices)
            results.append((q, k_dilated, v_dilated))

        return results

    # Copy remaining methods from RingDilatedAttentionV2
    def _allocate_tensor(self, shape, dtype, device, strategy="auto", zero_init=True):
        """Allocate tensor using enhanced memory pool if enabled."""
        if self._memory_pool is not None:
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            bytes_per_element = (
                torch.finfo(dtype).bits // 8
                if dtype.is_floating_point
                else torch.iinfo(dtype).bits // 8
            )
            tensor_size_mb = (num_elements * bytes_per_element) / (1024 * 1024)

            if tensor_size_mb >= 1.0:
                tensor = self._memory_pool.allocate(shape, dtype, device, strategy)
                if zero_init:
                    tensor.zero_()
                return tensor

        if zero_init:
            return torch.zeros(shape, dtype=dtype, device=device)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def _deallocate_tensor(self, tensor):
        """Return tensor to memory pool if enabled."""
        if self._memory_pool is not None:
            self._memory_pool.deallocate(tensor)

    def cleanup_buffers(self):
        """Clean up allocated communication buffers."""
        if self._kv_send_buffer is not None:
            self._deallocate_tensor(self._kv_send_buffer)
            self._kv_send_buffer = None
        if self._kv_recv_buffer is not None:
            self._deallocate_tensor(self._kv_recv_buffer)
            self._kv_recv_buffer = None

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_buffers()
        except Exception:
            pass

    def _apply_dilated_attention_pattern(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool
    ) -> Tensor:
        """Apply dilated attention patterns to Q, K, V tensors."""
        b, n, h, d = query.shape
        device, dtype = query.device, query.dtype

        output = self._allocate_tensor((b, n, h, d), dtype, device, strategy="auto")

        heads_per_group = self._calculate_head_groups(h)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            effective_segment_len = min(segment_len, n)
            head_end = head_start + group_size

            offset = i % dilation_rate if dilation_rate > 1 else 0
            group_output = self._process_dilated_segment(
                query[:, :, head_start:head_end, :],
                key[:, :, head_start:head_end, :],
                value[:, :, head_start:head_end, :],
                effective_segment_len,
                dilation_rate,
                offset,
                is_causal,
            )

            output[:, :, head_start:head_end, :] = group_output
            head_start = head_end

        return output

    def _calculate_head_groups(self, num_heads: int) -> list[int]:
        """Calculate how to distribute heads across different segment lengths."""
        num_segments = len(self.segment_lengths)
        base_heads = num_heads // num_segments
        extra_heads = num_heads % num_segments

        head_groups = [base_heads] * num_segments

        for i in range(extra_heads):
            head_groups[-(i + 1)] += 1

        return head_groups

    def _process_dilated_segment(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset: int,
        is_causal: bool,
    ) -> Tensor:
        """Process one dilated segment with the specified dilation rate."""
        b, n, h, d = q.shape

        if n < segment_len:
            return self._simple_attention(q, k, v, is_causal)

        num_segments = n // segment_len

        if num_segments == 0:
            return self._simple_attention(q, k, v, is_causal)

        q_seg = q[:, : num_segments * segment_len, :, :].view(
            b, num_segments, segment_len, h, d
        )
        k_seg = k[:, : num_segments * segment_len, :, :].view(
            b, num_segments, segment_len, h, d
        )
        v_seg = v[:, : num_segments * segment_len, :, :].view(
            b, num_segments, segment_len, h, d
        )

        if dilation_rate > 1 or offset > 0:
            q_seg, k_seg, v_seg = self._apply_dilation(
                q_seg, k_seg, v_seg, dilation_rate, offset, segment_len
            )

        q_flat = q_seg.transpose(2, 3).reshape(b * num_segments, h, segment_len, d)
        k_flat = k_seg.transpose(2, 3).reshape(b * num_segments, h, segment_len, d)
        v_flat = v_seg.transpose(2, 3).reshape(b * num_segments, h, segment_len, d)

        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / math.sqrt(d)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(segment_len, segment_len, device=q.device), diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        out_flat = torch.matmul(attn_weights, v_flat)

        out_seg = out_flat.reshape(b, num_segments, h, segment_len, d).transpose(2, 3)
        output = out_seg.reshape(b, num_segments * segment_len, h, d)

        if num_segments * segment_len < n:
            remaining_len = n - num_segments * segment_len
            q_remain = q[:, -remaining_len:, :, :]
            k_remain = k[:, -remaining_len:, :, :]
            v_remain = v[:, -remaining_len:, :, :]

            scores_remain = torch.matmul(
                q_remain.transpose(1, 2), k_remain.transpose(1, 2).transpose(-2, -1)
            ) / math.sqrt(d)

            if is_causal:
                causal_mask_remain = torch.triu(
                    torch.ones(remaining_len, remaining_len, device=q.device),
                    diagonal=1,
                ).bool()
                scores_remain.masked_fill_(
                    causal_mask_remain.unsqueeze(0).unsqueeze(1), float("-inf")
                )

            attn_remain = F.softmax(scores_remain, dim=-1)
            if self.dropout > 0 and self.training:
                attn_remain = F.dropout(attn_remain, p=self.dropout)

            out_remain = torch.matmul(attn_remain, v_remain.transpose(1, 2)).transpose(
                1, 2
            )

            full_output = self._allocate_tensor(
                (b, n, h, d), q.dtype, q.device, strategy="auto"
            )
            full_output[:, : num_segments * segment_len, :, :] = output
            full_output[:, -remaining_len:, :, :] = out_remain
            output = full_output

        return output

    def _simple_attention(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Simple attention for small sequences or fallback cases."""
        scores = torch.matmul(
            q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)
        ) / math.sqrt(q.size(-1))

        if is_causal:
            seq_len = q.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device), diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        output = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2)
        return output

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with correct Ring Attention."""
        if attention_mask is not None:
            warnings.warn("Attention mask not yet supported in Ring Attention V3")

        b, n, h, d = query.shape

        assert key.shape == value.shape == query.shape

        if self.mode == "single" or self.ring_size == 1:
            return self._single_device_forward(query, key, value, is_causal)
        elif self.mode == "simulated":
            return self._simulated_ring_forward(query, key, value, is_causal)
        else:
            return self._distributed_ring_forward(query, key, value, is_causal)

    def _single_device_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Dilated attention without ring for single device."""
        return self._apply_dilated_attention_pattern(q, k, v, is_causal)

    def _simulated_ring_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Simulate ring attention on single GPU by processing chunks sequentially."""
        b, n, h, d = q.shape
        chunk_size = n // self.ring_size

        output = torch.zeros_like(q)

        for i in range(self.ring_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)

            chunk_output = self._apply_dilated_attention_pattern(
                q[:, start_idx:end_idx],
                k[:, start_idx:end_idx],
                v[:, start_idx:end_idx],
                is_causal,
            )

            output[:, start_idx:end_idx] = chunk_output

        return output

    def _distributed_ring_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """True distributed ring attention across multiple GPUs."""
        # Implementation would be same as V2
        # Omitted for brevity - copy from RingDilatedAttentionV2
        raise NotImplementedError("Distributed mode not yet implemented in V3")

    def get_memory_estimate(
        self, seq_len: int, batch_size: int = 1, num_heads: int = 8, head_dim: int = 64
    ) -> dict:
        """Estimate memory usage for given configuration."""
        element_size = 2 if self.dtype == torch.float16 else 4

        q_memory = batch_size * seq_len * num_heads * head_dim * element_size

        chunk_size = (seq_len + self.ring_size - 1) // self.ring_size
        kv_memory = 2 * batch_size * chunk_size * num_heads * head_dim * element_size

        output_memory = batch_size * seq_len * num_heads * head_dim * element_size

        comm_memory = 2 * kv_memory if self.mode == "distributed" else 0

        # Add pattern cache memory estimate
        pattern_memory = 0
        if self.use_pattern_cache:
            stats = self._pattern_cache.get_stats()
            pattern_memory = stats.get("gpu_memory_used_mb", 0) * 1024 * 1024

        total = q_memory + kv_memory + output_memory + comm_memory + pattern_memory

        return {
            "mode": self.mode,
            "ring_size": self.ring_size,
            "q_memory_gb": q_memory / (1024**3),
            "kv_memory_gb": kv_memory / (1024**3),
            "output_memory_gb": output_memory / (1024**3),
            "comm_memory_gb": comm_memory / (1024**3),
            "pattern_memory_mb": pattern_memory / (1024**2),
            "total_per_device_gb": total / (1024**3),
            "memory_reduction_factor": (2 * seq_len) / (2 * chunk_size),
        }
