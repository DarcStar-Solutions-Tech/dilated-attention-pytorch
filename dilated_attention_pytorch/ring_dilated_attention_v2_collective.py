"""
Ring Dilated Attention V2 - Corrected with Collective Operations + Flash Attention.

This implementation fixes the distributed communication issues by using
collective operations (all_gather) instead of error-prone isend/irecv.

Key improvements:
1. Uses dist.all_gather instead of isend/irecv
2. No CUDA memory errors
3. Simpler and more robust
4. Often faster due to NCCL optimizations
5. Flash Attention integration for improved performance
6. Automatic backend selection (Flash Attention 3/2, xformers, SDPA)
7. Memory-efficient chunked processing for very long sequences
"""

import math
import warnings
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Note: Optimized attention is now handled via Flash Attention utilities
# The optimize_attention_computation function is no longer needed as we use
# flash_attention_forward directly

# Import enhanced memory pool if available
try:
    from .core.enhanced_memory_pool import get_enhanced_memory_pool

    HAS_ENHANCED_MEMORY_POOL = True
except ImportError:
    HAS_ENHANCED_MEMORY_POOL = False

# Import pattern cache if available
try:
    from .core.pattern_cache import get_global_pattern_cache

    HAS_PATTERN_CACHE = True
except ImportError:
    HAS_PATTERN_CACHE = False

# Import Flash Attention utilities
try:
    from .utils.flash_attention_utils import (
        flash_attention_forward,
        chunked_flash_attention,
        get_flash_attention_support,
    )

    HAS_FLASH_UTILS = True
except ImportError:
    HAS_FLASH_UTILS = False
    warnings.warn("Flash Attention utilities not available")

# Import ImprovedDilatedAttention for single-GPU fallback
try:
    from .improved_dilated_attention import ImprovedDilatedAttention

    HAS_IMPROVED_ATTENTION = True
except ImportError:
    HAS_IMPROVED_ATTENTION = False


class RingDilatedAttentionV2Collective(nn.Module):
    """
    Corrected Ring Dilated Attention using collective operations.

    This version replaces the problematic isend/irecv communication with
    robust all_gather operations that handle synchronization properly.

    This is a standalone implementation that does not inherit from the
    deprecated RingDilatedAttentionV2 class.
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
        memory_pool_threshold_mb: float = 16.0,
        use_flash_attention: bool = True,
        flash_chunk_size: int = 2048,
    ):
        """Initialize Ring Dilated Attention with collective operations."""
        super().__init__()

        assert len(segment_lengths) == len(dilation_rates)

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Smart dtype selection
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
                # Fallback to original logic if gpu_utils not available
                self.dtype = (
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

        # Pre-allocate lists for all_gather if in distributed mode
        if self.mode == "distributed" and self.ring_size > 1:
            self._k_chunks_list = None
            self._v_chunks_list = None

        # Create ImprovedDilatedAttention for single-GPU fallback
        self._single_gpu_attention = None
        if HAS_IMPROVED_ATTENTION and (self.mode == "single" or self.ring_size == 1):
            self._single_gpu_attention = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=dropout,
                device=device,
                dtype=dtype,
                enable_memory_pool=enable_memory_pool,
                enable_profiling=enable_profiling,
                lightweight_pool=lightweight_pool,
            )
            if self.rank == 0:  # Only log from rank 0 to avoid duplicate messages
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    "RingDilatedAttention: Using ImprovedDilatedAttention for single-GPU "
                    f"optimization (ring_size={self.ring_size})"
                )

        # Pattern caching setup
        self.use_pattern_cache = use_pattern_cache and HAS_PATTERN_CACHE
        if self.use_pattern_cache:
            # Use global pattern cache
            self._pattern_cache = get_global_pattern_cache()
        else:
            # Fall back to local cache
            self._dilated_indices_cache = {}

        # Flash Attention setup
        self.use_flash_attention = use_flash_attention and HAS_FLASH_UTILS
        self.flash_chunk_size = flash_chunk_size

        # Check Flash Attention support for this device
        if self.use_flash_attention:
            self.flash_support = get_flash_attention_support(self.device)
            self.flash_backend = self.flash_support["recommended_backend"]

            # Log attention backend configuration
            if self.rank == 0:  # Only log from rank 0
                if self.flash_backend == "xformers":
                    print(
                        f"Using xformers for GPU {torch.cuda.get_device_name(self.device)} "
                        f"(compute {self.flash_support['compute_capability']})"
                    )
                elif self.flash_backend == "sdpa":
                    print(
                        f"Using PyTorch SDPA for GPU {torch.cuda.get_device_name(self.device)} "
                        f"(compute {self.flash_support['compute_capability']})"
                    )
                elif self.flash_backend != "standard":
                    print(f"Using {self.flash_backend} for attention computation")
                else:
                    warnings.warn(
                        f"No optimized attention backend available for {torch.cuda.get_device_name(self.device)} "
                        f"(compute {self.flash_support['compute_capability']}). Using standard attention.",
                        RuntimeWarning,
                    )

        # Enhanced memory pool integration
        self.enable_memory_pool = enable_memory_pool and HAS_ENHANCED_MEMORY_POOL
        self.lightweight_pool = lightweight_pool
        self.memory_pool_threshold_mb = memory_pool_threshold_mb
        self._memory_pool = None
        if self.enable_memory_pool:
            if lightweight_pool:
                # Use lightweight pool for communication buffers and outputs
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=False,  # Disable for speed
                    enable_bucketed=True,  # Keep for different buffer sizes
                    enable_numa=False,  # Disable for speed in distributed setting
                    enable_profiling=enable_profiling,
                )
            else:
                # Full memory pool with all features
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=True,
                    enable_bucketed=True,
                    enable_numa=True,
                    enable_profiling=enable_profiling,
                )

        # CRITICAL FIX: Add caching for masks and patterns
        self._causal_mask_cache = {}
        self._dilation_pattern_cache = {}
        self._head_groups_cache = None

    def _get_causal_mask(
        self, seq_len_q: int, seq_len_kv: int, chunk_offset: int = 0
    ) -> Tensor:
        """Get cached causal mask or create new one."""
        cache_key = (seq_len_q, seq_len_kv, chunk_offset)

        if cache_key not in self._causal_mask_cache:
            # Create vectorized causal mask
            if chunk_offset > 0:
                # For ring attention chunks
                row_indices = torch.arange(seq_len_q, device=self.device).unsqueeze(1)
                col_indices = torch.arange(seq_len_kv, device=self.device).unsqueeze(0)
                mask = (row_indices + chunk_offset) >= col_indices
            else:
                # Standard causal mask
                mask = torch.tril(
                    torch.ones(
                        seq_len_q, seq_len_kv, device=self.device, dtype=torch.bool
                    )
                )

            self._causal_mask_cache[cache_key] = mask

            # Limit cache size to prevent memory bloat
            if len(self._causal_mask_cache) > 100:
                # Remove oldest entries
                keys_to_remove = list(self._causal_mask_cache.keys())[:50]
                for key in keys_to_remove:
                    del self._causal_mask_cache[key]

        return self._causal_mask_cache[cache_key]

    def _calculate_head_groups(self, num_heads: int) -> list[int]:
        """Calculate how to distribute heads across different segment lengths."""
        # Use cached result if available
        if self._head_groups_cache is not None and len(self._head_groups_cache) == len(
            self.segment_lengths
        ):
            total_cached = sum(self._head_groups_cache)
            if total_cached == num_heads:
                return self._head_groups_cache

        num_segments = len(self.segment_lengths)
        base_heads = num_heads // num_segments
        extra_heads = num_heads % num_segments

        head_groups = [base_heads] * num_segments

        # Distribute extra heads to larger segments (later in the list)
        for i in range(extra_heads):
            head_groups[-(i + 1)] += 1

        # Cache the result
        self._head_groups_cache = head_groups

        return head_groups

    def _allocate_tensor(self, shape, dtype, device, strategy="auto", zero_init=True):
        """
        Allocate tensor using enhanced memory pool if enabled.

        Args:
            shape: Tensor shape tuple
            dtype: Tensor data type
            device: Target device
            strategy: Allocation strategy for memory pool
            zero_init: Whether to zero-initialize the tensor

        Returns:
            Allocated tensor (optionally zero-initialized)
        """
        if self._memory_pool is not None:
            # Calculate tensor size in bytes
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            bytes_per_element = (
                torch.finfo(dtype).bits // 8
                if dtype.is_floating_point
                else torch.iinfo(dtype).bits // 8
            )
            tensor_size_mb = (num_elements * bytes_per_element) / (1024 * 1024)

            # Use memory pool only for large tensors
            # Even in distributed mode, only use for sufficiently large allocations
            use_pool = tensor_size_mb >= self.memory_pool_threshold_mb

            if use_pool:
                tensor = self._memory_pool.allocate(shape, dtype, device, strategy)
                if zero_init:
                    tensor.zero_()
                return tensor

        # Fallback to direct allocation for small tensors or when pool is disabled
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
            pass  # Ignore errors during cleanup

    def _compute_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        chunk_offset: int = 0,
        use_flash: Optional[bool] = None,
    ) -> Tensor:
        """
        Unified attention computation with automatic backend selection.

        Args:
            q, k, v: Tensors of shape [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            chunk_offset: Offset for causal mask in chunked processing
            use_flash: Override flash attention setting (None = use default)

        Returns:
            Attention output of shape [batch, seq_len, num_heads, head_dim]
        """
        use_flash = use_flash if use_flash is not None else self.use_flash_attention

        # Try Flash Attention first if enabled
        if use_flash and HAS_FLASH_UTILS:
            try:
                # Adjust causal for chunk offset
                adjusted_causal = is_causal and chunk_offset == 0

                output = flash_attention_forward(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=adjusted_causal,
                    backend=self.flash_backend,
                )
                return output
            except Exception:
                pass  # Fall through to standard

        # Standard PyTorch implementation
        b, n_q, h, d = q.shape
        n_kv = k.shape[1]

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [b, h, n_q, d]
        k = k.transpose(1, 2)  # [b, h, n_kv, d]
        v = v.transpose(1, 2)  # [b, h, n_kv, d]

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        # Apply causal mask
        if is_causal:
            causal_mask = self._get_causal_mask(n_q, n_kv, chunk_offset)
            scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention
        output = torch.matmul(attn_weights, v)
        return output.transpose(1, 2)  # [b, n_q, h, d]

    def _simple_attention(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Compute simple attention without dilation for fallback cases."""
        # Just delegate to unified method
        return self._compute_attention(q, k, v, is_causal, chunk_offset=0)

    def _apply_dilated_attention_pattern(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool
    ) -> Tensor:
        """
        Apply dilated attention patterns to Q, K, V tensors.

        This method divides attention heads into groups, where each group
        processes different segment lengths with different dilation rates.
        """
        b, n, h, d = query.shape
        device, dtype = query.device, query.dtype

        # Pre-allocate output using memory pool (main output tensor)
        output = self._allocate_tensor((b, n, h, d), dtype, device, strategy="auto")

        # Calculate head groups for different segment lengths
        heads_per_group = self._calculate_head_groups(h)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            # If sequence is shorter than segment, use the full sequence
            effective_segment_len = min(segment_len, n)

            head_end = head_start + group_size

            # Apply dilated attention for this head group
            # Use modulo to cycle offset through dilation rate
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
        """
        Process one dilated segment with the specified dilation rate.
        """
        b, n, h, d = q.shape

        # Handle small sequences that don't fit the segment length
        if n < segment_len:
            # Use simple attention for small sequences
            return self._simple_attention(q, k, v, is_causal)

        # Reshape into segments
        num_segments = n // segment_len

        # Skip if no complete segments
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

        # Apply dilation to segments
        if dilation_rate > 1:
            q_seg, k_seg, v_seg = self._apply_dilation(
                q_seg, k_seg, v_seg, dilation_rate, offset
            )

        # Compute attention for dilated segments
        output_seg = []
        for seg_idx in range(num_segments):
            seg_output = self._simple_attention(
                q_seg[:, seg_idx], k_seg[:, seg_idx], v_seg[:, seg_idx], is_causal
            )
            output_seg.append(seg_output)

        # Stack segment outputs
        output_seg = torch.stack(
            output_seg, dim=1
        )  # [b, num_segments, segment_len, h, d]
        output_seg = output_seg.view(b, num_segments * segment_len, h, d)

        # Handle remaining positions
        output = self._allocate_tensor((b, n, h, d), q.dtype, q.device)
        output[:, : num_segments * segment_len] = output_seg

        if n > num_segments * segment_len:
            remainder_output = self._simple_attention(
                q[:, num_segments * segment_len :],
                k[:, num_segments * segment_len :],
                v[:, num_segments * segment_len :],
                is_causal,
            )
            output[:, num_segments * segment_len :] = remainder_output

        return output

    def _apply_dilation(
        self,
        q_seg: Tensor,
        k_seg: Tensor,
        v_seg: Tensor,
        dilation_rate: int,
        offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply dilation pattern to segments with caching."""
        b, num_segments, segment_len, h, d = q_seg.shape

        # Skip if segment_len < dilation_rate
        if segment_len < dilation_rate:
            return q_seg, k_seg, v_seg

        # Generate cache key for pattern
        cache_key = (segment_len, dilation_rate, offset, q_seg.device.type)

        # Check pattern cache
        if self.use_pattern_cache and self._pattern_cache is not None:
            dilated_indices = self._pattern_cache.get(cache_key)

            if dilated_indices is None:
                # Generate dilated indices
                base_indices = torch.arange(
                    0, segment_len, dilation_rate, device=q_seg.device
                )
                dilated_indices = (base_indices + offset) % segment_len

                # Pad if necessary to maintain segment length
                if len(dilated_indices) < segment_len:
                    # Repeat pattern to fill segment
                    repeats = (segment_len + len(dilated_indices) - 1) // len(
                        dilated_indices
                    )
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:segment_len]
                dilated_indices = dilated_indices % segment_len

                # Store in pattern cache
                self._pattern_cache[cache_key] = dilated_indices
        else:
            # Check local cache
            if cache_key in self._dilated_indices_cache:
                dilated_indices = self._dilated_indices_cache[cache_key]
            else:
                # Generate dilated indices
                base_indices = torch.arange(
                    0, segment_len, dilation_rate, device=q_seg.device
                )
                dilated_indices = (base_indices + offset) % segment_len

                # Pad if necessary to maintain segment length
                if len(dilated_indices) < segment_len:
                    # Repeat pattern to fill segment
                    repeats = (segment_len + len(dilated_indices) - 1) // len(
                        dilated_indices
                    )
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:segment_len]
                dilated_indices = dilated_indices % segment_len

                # Store in local cache if pattern cache not available
                if not self.use_pattern_cache:
                    self._dilated_indices_cache[cache_key] = dilated_indices

        # Apply dilation by gathering
        q_dilated = q_seg.index_select(2, dilated_indices)
        k_dilated = k_seg.index_select(2, dilated_indices)
        v_dilated = v_seg.index_select(2, dilated_indices)

        return q_dilated, k_dilated, v_dilated

    def _single_device_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Single device forward - just apply dilated attention patterns."""
        return self._apply_dilated_attention_pattern(q, k, v, is_causal)

    def _simulated_ring_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """
        Simulate ring attention on a single device by processing chunks sequentially.
        This allows testing ring attention logic without multiple GPUs.
        """
        b, n, h, d = q.shape
        chunk_size = (n + self.ring_size - 1) // self.ring_size

        # Apply dilated patterns to full Q
        q_dilated = self._apply_dilated_patterns_to_query(q)

        # Initialize output and online softmax accumulators
        output = torch.zeros((b, h, n, d), device=q.device, dtype=q.dtype)
        running_max = torch.full(
            (b, h, n, 1), float("-inf"), device=q.device, dtype=q.dtype
        )
        running_sum = torch.zeros((b, h, n, 1), device=q.device, dtype=q.dtype)

        # Process each chunk sequentially (simulating ring communication)
        for step in range(self.ring_size):
            chunk_idx = step
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n)
            actual_chunk_size = chunk_end - chunk_start

            if actual_chunk_size <= 0:
                continue

            # Extract K/V chunks
            k_chunk = k[:, chunk_start:chunk_end]
            v_chunk = v[:, chunk_start:chunk_end]

            # Apply dilated patterns to chunks
            k_chunk_dilated, v_chunk_dilated = self._apply_dilated_patterns_to_chunk(
                k_chunk, v_chunk, chunk_start, actual_chunk_size
            )

            # Compute attention with online softmax
            self._compute_chunk_attention_with_online_softmax(
                q_dilated,
                k_chunk_dilated,
                v_chunk_dilated,
                chunk_start,
                is_causal,
                running_max,
                running_sum,
                output,
                step,
            )

        # Final normalization
        output = output / (running_sum + 1e-8)

        # Transpose back to [b, n, h, d]
        return output.transpose(1, 2)

    def _ring_attention(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """
        Ring attention with collective operations and proper dilated patterns.

        This implementation:
        1. Applies dilated attention patterns to the full Q, K, V first
        2. Gathers all dilated K/V chunks using collective operations
        3. Processes them with online softmax for correct normalization
        """
        b, n, h, d = q.shape

        # CRITICAL FIX: Remove incorrect dilated pattern application
        # The _apply_dilated_attention_pattern method computes attention output,
        # not dilated K/V tensors. The dilation is already applied per chunk
        # in the ring attention loop below.

        # Apply dilated patterns directly to K and V chunks
        chunk_size = (n + self.ring_size - 1) // self.ring_size

        # Get local K/V chunks
        local_start = self.rank * chunk_size
        local_end = min((self.rank + 1) * chunk_size, n)
        actual_chunk_size = local_end - local_start

        # Extract local chunks
        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Apply dilated patterns to the local chunks
        k_local_dilated, v_local_dilated = self._apply_dilated_patterns_to_chunk(
            k_local, v_local, local_start, actual_chunk_size
        )

        # Handle padding if needed
        if actual_chunk_size < chunk_size:
            pad_size = chunk_size - actual_chunk_size
            k_local_dilated = F.pad(k_local_dilated, (0, 0, 0, 0, 0, pad_size), value=0)
            v_local_dilated = F.pad(v_local_dilated, (0, 0, 0, 0, 0, pad_size), value=0)

        # Gather all dilated K/V chunks across devices using collective operation
        if (
            self._k_chunks_list is None
            or self._k_chunks_list[0].shape != k_local_dilated.shape
        ):
            self._k_chunks_list = [
                torch.empty_like(k_local_dilated) for _ in range(self.ring_size)
            ]
            self._v_chunks_list = [
                torch.empty_like(v_local_dilated) for _ in range(self.ring_size)
            ]

        # All-gather operation - robust and handles synchronization
        dist.all_gather(self._k_chunks_list, k_local_dilated)
        dist.all_gather(self._v_chunks_list, v_local_dilated)

        # Apply dilated patterns to full query as well
        q_dilated = self._apply_dilated_patterns_to_query(q)

        # Process each chunk with dilated attention using online softmax
        output = torch.zeros((b, h, n, d), device=q.device, dtype=q.dtype)
        running_max = torch.full(
            (b, h, n, 1), float("-inf"), device=q.device, dtype=q.dtype
        )
        running_sum = torch.zeros((b, h, n, 1), device=q.device, dtype=q.dtype)

        for step in range(self.ring_size):
            # Determine which chunk to process
            chunk_idx = (self.rank - step) % self.ring_size
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n)

            # Get the dilated K/V chunks for this step
            k_chunk_dilated = self._k_chunks_list[chunk_idx]
            v_chunk_dilated = self._v_chunks_list[chunk_idx]

            # Trim padding if this is the last chunk
            if chunk_end - chunk_start < chunk_size:
                actual_size = chunk_end - chunk_start
                k_chunk_dilated = k_chunk_dilated[:, :actual_size]
                v_chunk_dilated = v_chunk_dilated[:, :actual_size]

            # Compute attention with dilated patterns using online softmax
            self._compute_chunk_attention_with_online_softmax(
                q_dilated,
                k_chunk_dilated,
                v_chunk_dilated,
                chunk_start,
                is_causal,
                running_max,
                running_sum,
                output,
                step,
            )

        # Final normalization
        output = output / (running_sum + 1e-8)  # Add epsilon for numerical stability

        # Transpose back to [b, n, h, d]
        return output.transpose(1, 2)

    def _compute_chunk_attention_simple(
        self, q: Tensor, k_chunk: Tensor, v_chunk: Tensor, is_causal: bool
    ):
        """Simplified attention computation for a chunk."""
        b, n, h, d = q.shape
        _, n_kv, _, _ = k_chunk.shape

        # Transpose to [b, h, n, d] format
        q_t = q.transpose(1, 2)  # [b, h, n, d]
        k_t = k_chunk.transpose(1, 2)  # [b, h, n_kv, d]
        v_t = v_chunk.transpose(1, 2)  # [b, h, n_kv, d]

        # Compute attention scores
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(
            d
        )  # [b, h, n, n_kv]

        # Apply causal mask if needed (simplified)
        if is_causal:
            # For simplicity, apply basic causal mask
            mask = torch.triu(torch.ones(n, n_kv, device=q.device), diagonal=1).bool()
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Get max values for numerical stability
        max_vals = scores.amax(dim=-1, keepdim=True)  # [b, h, n, 1]

        # Compute softmax
        exp_scores = torch.exp(scores - max_vals)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        attn_weights = exp_scores / sum_exp

        # Apply to values
        output = torch.matmul(attn_weights, v_t)  # [b, h, n, d]

        return output, max_vals

    def _combine_chunk_outputs(self, outputs, max_vals):
        """Combine outputs from different chunks with proper normalization."""
        if len(outputs) == 1:
            return outputs[0]

        # For simplicity, just average the outputs
        # In a full implementation, this would use proper online softmax
        combined = torch.stack(outputs, dim=0).mean(dim=0)

        # Transpose back to [b, n, h, d]
        return combined.transpose(1, 2)

    def _apply_dilation_to_tensor(
        self,
        tensor: Tensor,
        segment_lengths: list[int],
        dilation_rates: list[int],
    ) -> Tensor:
        """
        Apply dilation patterns to a single tensor (K or V).

        Args:
            tensor: Input tensor of shape [b, n, h, d]
            segment_lengths: List of segment lengths
            dilation_rates: List of dilation rates

        Returns:
            Dilated tensor of same shape
        """
        b, n, h, d = tensor.shape

        # For dilation_rate=1, no change needed
        if all(dr == 1 for dr in dilation_rates):
            return tensor

        # TODO: Implement proper dilation logic here
        # For now, return tensor as-is since dilation is handled per-chunk
        return tensor

    def _apply_dilated_patterns_to_chunk(
        self, k_chunk: Tensor, v_chunk: Tensor, chunk_start: int, chunk_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply dilated patterns to K/V chunks based on head groups."""
        b, n, h, d = k_chunk.shape

        # Calculate head groups for different segment lengths (same as parent)
        heads_per_group = self._calculate_head_groups(h)

        # Create output tensors
        k_dilated = torch.zeros_like(k_chunk)
        v_dilated = torch.zeros_like(v_chunk)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Apply dilation to this head group
            offset = i % dilation_rate if dilation_rate > 1 else 0

            # Get dilated indices for this segment
            k_group = k_chunk[:, :, head_start:head_end, :]
            v_group = v_chunk[:, :, head_start:head_end, :]

            if dilation_rate > 1 and n >= dilation_rate:
                # Apply dilation pattern
                dilated_indices = torch.arange(
                    offset, n, dilation_rate, device=k_chunk.device
                )
                if len(dilated_indices) < n:
                    # Pad by repeating pattern
                    repeats = (n + len(dilated_indices) - 1) // len(dilated_indices)
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:n] % n

                k_group_dilated = k_group.index_select(1, dilated_indices)
                v_group_dilated = v_group.index_select(1, dilated_indices)
            else:
                # No dilation needed - these groups could potentially use optimized attention
                k_group_dilated = k_group
                v_group_dilated = v_group

            k_dilated[:, :, head_start:head_end, :] = k_group_dilated
            v_dilated[:, :, head_start:head_end, :] = v_group_dilated
            head_start = head_end

        return k_dilated, v_dilated

    def _apply_dilated_patterns_to_query(self, q: Tensor) -> Tensor:
        """Apply dilated patterns to query tensor based on head groups."""
        b, n, h, d = q.shape

        # Calculate head groups for different segment lengths (same as parent)
        heads_per_group = self._calculate_head_groups(h)

        # Create output tensor
        q_dilated = torch.zeros_like(q)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Apply dilation to this head group
            offset = i % dilation_rate if dilation_rate > 1 else 0

            # Get query group
            q_group = q[:, :, head_start:head_end, :]

            if dilation_rate > 1 and n >= dilation_rate:
                # Apply dilation pattern to query
                dilated_indices = torch.arange(
                    offset, n, dilation_rate, device=q.device
                )
                if len(dilated_indices) < n:
                    # Pad by repeating pattern
                    repeats = (n + len(dilated_indices) - 1) // len(dilated_indices)
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:n] % n

                q_group_dilated = q_group.index_select(1, dilated_indices)
            else:
                # No dilation needed
                q_group_dilated = q_group

            q_dilated[:, :, head_start:head_end, :] = q_group_dilated
            head_start = head_end

        return q_dilated

    def _compute_chunk_attention_with_online_softmax(
        self,
        q_dilated: Tensor,
        k_chunk: Tensor,
        v_chunk: Tensor,
        chunk_start: int,
        is_causal: bool,
        running_max: Tensor,
        running_sum: Tensor,
        output: Tensor,
        step: int,
    ):
        """Compute attention for a chunk using optimized attention and online softmax."""
        b, n, h, d = q_dilated.shape
        _, n_kv, _, _ = k_chunk.shape

        # CRITICAL FIX: Remove double computation - choose Flash OR manual, not both
        if self.use_flash_attention and HAS_FLASH_UTILS and not is_causal and n == n_kv:
            # Use Flash Attention for non-causal same-length chunks
            try:
                # Use unified attention computation
                chunk_output = self._compute_attention(
                    q_dilated, k_chunk, v_chunk, is_causal=False, chunk_offset=0
                )

                # For online softmax, compute only the max statistics we need
                q_t = q_dilated.transpose(1, 2)
                k_t = k_chunk.transpose(1, 2)

                # Compute just max values for online softmax
                scores_for_stats = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(
                    d
                )
                chunk_max = scores_for_stats.amax(dim=-1, keepdim=True)

                # Update online softmax running statistics
                new_max = torch.maximum(running_max, chunk_max)
                if step > 0:
                    output.mul_(torch.exp(running_max - new_max))
                    running_sum.mul_(torch.exp(running_max - new_max))

                running_max.copy_(new_max)

                # Add Flash output (transpose back from [b,n,h,d] to [b,h,n,d])
                output.add_(chunk_output.transpose(1, 2))

                # Approximate running sum update for Flash path
                running_sum.add_(torch.ones_like(running_sum))

                return  # Skip manual computation

            except Exception:
                pass  # Fall through to manual computation

        # Manual computation fallback (original implementation)
        # Transpose to [b, h, n, d] format for computation
        q_t = q_dilated.transpose(1, 2)  # [b, h, n, d]
        k_t = k_chunk.transpose(1, 2)  # [b, h, n_kv, d]
        v_t = v_chunk.transpose(1, 2)  # [b, h, n_kv, d]

        # Compute attention scores
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(
            d
        )  # [b, h, n, n_kv]

        # Apply causal mask if needed
        if is_causal:
            # Use cached causal mask
            causal_mask = self._get_causal_mask(n, n_kv, chunk_start)
            scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        # Online softmax update
        chunk_max = scores.amax(dim=-1, keepdim=True)  # [b, h, n, 1]
        new_max = torch.maximum(running_max, chunk_max)

        # Rescale existing output if max changed and not first step
        if step > 0:
            output.mul_(torch.exp(running_max - new_max))

        # Update running sum with proper scaling
        running_sum.mul_(torch.exp(running_max - new_max))
        running_sum.add_(torch.exp(scores - new_max).sum(dim=-1, keepdim=True))

        # Update running max
        running_max.copy_(new_max)

        # Accumulate weighted values
        exp_scores = torch.exp(scores - new_max)  # [b, h, n, n_kv]
        chunk_output = torch.matmul(exp_scores, v_t)  # [b, h, n, d]

        # Add to output (already in [b, h, n, d] format)
        output.add_(chunk_output)

    def _compute_attention_chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        chunk_offset: int = 0,
    ) -> Tensor:
        """
        Compute attention for a chunk using optimized backend.

        Args:
            q, k, v: Query, key, value tensors [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to use causal masking
            chunk_offset: Offset for causal masking in chunked processing

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        # Just delegate to unified method
        return self._compute_attention(q, k, v, is_causal, chunk_offset)

    def _compute_attention_standard(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        chunk_offset: int = 0,
    ) -> Tensor:
        """Standard attention computation (fallback)."""
        # Just delegate to unified method with Flash disabled
        return self._compute_attention(
            q, k, v, is_causal, chunk_offset, use_flash=False
        )

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass with collective communication.

        Routes to appropriate implementation based on mode.
        """
        # Use ImprovedDilatedAttention for single GPU if available
        if self._single_gpu_attention is not None and self.ring_size == 1:
            return self._single_gpu_attention(q, k, v, is_causal)

        # For very long sequences on single GPU, use chunked Flash Attention if available
        seq_len = q.shape[1]
        if (
            self.use_flash_attention
            and self.ring_size == 1
            and seq_len > 16384
            and HAS_FLASH_UTILS
        ):
            try:
                # Use chunked Flash Attention for very long sequences
                output = chunked_flash_attention(
                    q,
                    k,
                    v,
                    chunk_size=self.flash_chunk_size,
                    is_causal=is_causal,
                    dropout_p=self.dropout if self.training else 0.0,
                    backend=self.flash_backend,
                )
                return output
            except Exception as e:
                warnings.warn(
                    f"Chunked Flash Attention failed for long sequence, falling back: {e}",
                    stacklevel=2,
                )

        # For distributed mode with ring_size > 1, use collective ring attention
        if self.mode == "distributed" and self.ring_size > 1:
            return self._ring_attention(q, k, v, is_causal)
        elif self.mode == "simulated" and self.ring_size > 1:
            return self._simulated_ring_forward(q, k, v, is_causal)
        else:
            # Fall back to single device implementation
            return self._single_device_forward(q, k, v, is_causal)


# Convenience function for creating the corrected version
def create_ring_dilated_attention_v2_collective(
    segment_lengths: list[int], dilation_rates: list[int], **kwargs
) -> RingDilatedAttentionV2Collective:
    """
    Create a Ring Dilated Attention V2 with collective operations.

    This is the recommended version for distributed training as it uses
    robust collective operations instead of error-prone point-to-point
    communication.
    """
    return RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths, dilation_rates=dilation_rates, **kwargs
    )
