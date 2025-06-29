"""
Improved Dilated Attention implementation using the refactored core architecture.

This module provides an optimized dilated attention mechanism with enhanced
memory efficiency and performance optimizations.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

# Handle torch.nn.attention availability
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    HAS_SDPA_KERNEL = True
except ImportError:
    HAS_SDPA_KERNEL = False

    # Fallback for older PyTorch versions
    class SDPBackend:
        FLASH_ATTENTION = "flash_attention"
        EFFICIENT_ATTENTION = "efficient_attention"
        MATH = "math"

    def sdpa_kernel(_backends):
        """Dummy context manager for older PyTorch."""
        import contextlib

        return contextlib.nullcontext()


from .core import BaseDilatedAttention, DilatedAttentionConfig, get_global_pattern_cache
from .core.enhanced_memory_pool import get_enhanced_memory_pool


class ImprovedDilatedAttention(BaseDilatedAttention):
    """
    Improved implementation of dilated attention with optimizations.

    This implementation includes:
    - Pre-computed head distributions for efficiency
    - Cached dilation indices to avoid recomputation
    - Direct tensor views instead of einops for memory efficiency
    - Optimized SDPA kernel selection
    - In-place operations to reduce memory allocation
    - Enhanced memory pool integration for reduced allocation overhead

    Args:
        segment_lengths: List of segment lengths for each attention group
        dilation_rates: List of dilation rates corresponding to each segment
        dropout: Dropout probability (default: 0.0)
        use_tf32: Whether to enable TF32 for matmul ops (default: True)
        enable_memory_pool: Enable enhanced memory pool for tensor allocation (default: True)
        enable_profiling: Enable memory profiling when using memory pool (default: False)
        lightweight_pool: Use lightweight pool config for better performance (default: True)

    Example:
        >>> attention = ImprovedDilatedAttention(
        ...     segment_lengths=[2048, 4096, 8192],
        ...     dilation_rates=[1, 2, 4],
        ...     dropout=0.1,
        ...     enable_memory_pool=True,
        ...     enable_profiling=True
        ... )
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dropout: float = 0.0,
        use_tf32: bool = True,
        enable_memory_pool: bool = False,  # Disabled by default due to overhead
        enable_profiling: bool = False,
        lightweight_pool: bool = True,
        **kwargs,
    ):
        # Create configuration
        config = DilatedAttentionConfig(
            segment_lengths=list(segment_lengths),
            dilation_rates=list(dilation_rates),
            dropout=dropout,
            use_tf32=use_tf32,
            **kwargs,
        )

        # Initialize base class
        super().__init__(config)

        # Use global pattern cache instead of local cache
        self._pattern_cache = get_global_pattern_cache()
        self._sdpa_backend = self._select_sdpa_backend()

        # Enhanced memory pool integration
        self.enable_memory_pool = enable_memory_pool
        self.lightweight_pool = lightweight_pool
        self._memory_pool = None
        if enable_memory_pool:
            if lightweight_pool:
                # Use simpler memory pool configuration for better performance
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=False,  # Disable for speed
                    enable_bucketed=True,  # Keep for common sizes
                    enable_numa=False,  # Disable for speed
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

    def _select_sdpa_backend(self):
        """Select optimal SDPA backend based on hardware."""
        if not HAS_SDPA_KERNEL:
            return None

        # Prioritize Flash Attention if available
        backends = []

        # Check Flash Attention availability
        if self._use_flash_attn:
            backends.append(SDPBackend.FLASH_ATTENTION)

        # Add other backends
        backends.extend([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

        return backends

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

            # Only use memory pool for tensors larger than 1MB
            # Small tensors have too much overhead
            if tensor_size_mb >= 1.0:
                tensor = self._memory_pool.allocate(shape, dtype, device, strategy)
                if zero_init:
                    tensor.zero_()
                return tensor

        # Fallback to direct allocation for small tensors or when pool is disabled
        if zero_init:
            return torch.zeros(shape, dtype=dtype, device=device)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for improved dilated attention.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            attention_mask: Optional attention mask (not supported in segments)

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        # Validate inputs using base class method
        self._validate_forward_inputs(query, key, value, attention_mask)

        # Extract dimensions
        b, n, h, d = query.shape
        device, dtype = query.device, query.dtype

        # Pre-allocate output using enhanced memory pool (main output tensor)
        out = self._allocate_tensor((b, n, h, d), dtype, device, strategy="auto")

        # Get head groups from base class cache
        group_sizes, head_ranges = self._get_head_groups(h)

        # Process all segments with optimized memory access patterns
        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths, strict=False)
        ):
            if g == 0 or n < s:  # Skip empty groups or too-small sequences
                continue

            hmin, hmax = head_ranges[i]
            offset = i % r

            # Use direct tensor views for memory efficiency
            # Shape: [b, n, h, d] -> [b, n//s, s, g, d]
            q_slice = query[:, :, hmin:hmax, :].view(b, n // s, s, g, d)
            k_slice = key[:, :, hmin:hmax, :].view(b, n // s, s, g, d)
            v_slice = value[:, :, hmin:hmax, :].view(b, n // s, s, g, d)

            # Apply dilation with cached indices
            if r > 1 or offset:
                # Use pattern cache for dilated indices
                cache_key = f"dilated_indices_s{s}_r{r}_off{offset}"
                idx = self._pattern_cache.get(cache_key, target_device=device)

                if idx is None:
                    # Create dilated indices on CPU and cache
                    idx = torch.arange(offset, s, r, device=torch.device("cpu"))
                    self._pattern_cache.put(cache_key, idx, move_to_cpu=False)
                    idx = idx.to(device)

                # Use advanced indexing for dilated sampling
                q_slice = q_slice[:, :, idx, :, :]
                k_slice = k_slice[:, :, idx, :, :]
                v_slice = v_slice[:, :, idx, :, :]

            # Reshape for attention
            bn = b * (n // s)
            dilated_s = q_slice.size(2)
            q_flat = q_slice.contiguous().view(bn, dilated_s, g, d)
            k_flat = k_slice.contiguous().view(bn, dilated_s, g, d)
            v_flat = v_slice.contiguous().view(bn, dilated_s, g, d)

            # Apply optimized attention computation
            if HAS_SDPA_KERNEL and self._sdpa_backend:
                with sdpa_kernel(self._sdpa_backend):
                    x = F.scaled_dot_product_attention(
                        q_flat,
                        k_flat,
                        v_flat,
                        attn_mask=None,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal,
                        scale=None,
                    )
            else:
                # Fallback to standard SDPA
                x = F.scaled_dot_product_attention(
                    q_flat,
                    k_flat,
                    v_flat,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )

            # Reshape back and add to output
            x_reshaped = x.view(b, n // s, dilated_s, g, d)

            # Scatter back to original positions
            if r > 1 or offset:
                # Check if we should use memory pool for this temporary tensor
                temp_shape = (b, n // s, s, g, d)
                temp_elements = b * (n // s) * s * g * d
                temp_bytes = temp_elements * (torch.finfo(dtype).bits // 8)
                temp_size_mb = temp_bytes / (1024 * 1024)

                if self._memory_pool is not None and temp_size_mb >= 1.0:
                    # Use memory pool for large temporary tensors
                    temp_output = self._allocate_tensor(
                        temp_shape,
                        dtype,
                        device,
                        strategy="bucketed",
                        zero_init=False,
                    )
                    temp_output.fill_(0.0)
                    temp_output[:, :, idx, :, :] = x_reshaped
                    out[:, :, hmin:hmax, :].add_(temp_output.reshape(b, n, g, d))
                    # Return to pool
                    self._memory_pool.deallocate(temp_output)
                else:
                    # Direct allocation for small tensors
                    temp_output = torch.zeros(temp_shape, dtype=dtype, device=device)
                    temp_output[:, :, idx, :, :] = x_reshaped
                    out[:, :, hmin:hmax, :].add_(temp_output.reshape(b, n, g, d))
            else:
                out[:, :, hmin:hmax, :].add_(x_reshaped.reshape(b, n, g, d))

        # Normalize by number of groups (in-place for efficiency)
        # NOTE: Normalization must happen before dropout for mathematical correctness
        out.div_(self.num_groups)

        # Apply dropout if configured (from base class)
        out = self._apply_dropout(out)

        return out

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        repr_str = super().extra_repr()
        cache_stats = self._pattern_cache.get_stats()
        repr_str += f", pattern_cache_size={cache_stats['size']}"
        if HAS_SDPA_KERNEL:
            repr_str += ", sdpa_optimized=True"
        return repr_str


# Backward compatibility function
def create_improved_dilated_attention(
    segment_lengths: Sequence[int], dilation_rates: Sequence[int], **kwargs
) -> ImprovedDilatedAttention:
    """
    Create an improved dilated attention module (backward compatibility).

    Args:
        segment_lengths: List of segment lengths
        dilation_rates: List of dilation rates
        **kwargs: Additional arguments

    Returns:
        ImprovedDilatedAttention module
    """
    return ImprovedDilatedAttention(segment_lengths, dilation_rates, **kwargs)


# Optional: Enable torch.compile for further optimization
# To enable:
# ImprovedDilatedAttention = torch.compile(ImprovedDilatedAttention, fullgraph=True)
