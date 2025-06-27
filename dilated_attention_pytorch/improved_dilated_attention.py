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


from .core import BaseDilatedAttention, DilatedAttentionConfig


class ImprovedDilatedAttention(BaseDilatedAttention):
    """
    Improved implementation of dilated attention with optimizations.

    This implementation includes:
    - Pre-computed head distributions for efficiency
    - Cached dilation indices to avoid recomputation
    - Direct tensor views instead of einops for memory efficiency
    - Optimized SDPA kernel selection
    - In-place operations to reduce memory allocation

    Args:
        segment_lengths: List of segment lengths for each attention group
        dilation_rates: List of dilation rates corresponding to each segment
        dropout: Dropout probability (default: 0.0)
        use_tf32: Whether to enable TF32 for matmul ops (default: True)

    Example:
        >>> attention = ImprovedDilatedAttention(
        ...     segment_lengths=[2048, 4096, 8192],
        ...     dilation_rates=[1, 2, 4],
        ...     dropout=0.1
        ... )
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dropout: float = 0.0,
        use_tf32: bool = True,
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

        # Additional caches for improved performance
        self._cached_indices = {}  # Cache for dilation indices
        self._sdpa_backend = self._select_sdpa_backend()

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

        # Pre-allocate output with optimal memory pattern
        out = torch.zeros(b, n, h, d, device=device, dtype=dtype)

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
                # Get or create cached indices
                cache_key = (s, r, offset, device)
                if cache_key not in self._cached_indices:
                    self._cached_indices[cache_key] = torch.arange(
                        offset, s, r, device=device
                    )
                idx = self._cached_indices[cache_key]

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
                # Create temporary tensor for scattering
                temp_output = torch.zeros(
                    b, n // s, s, g, d, device=device, dtype=dtype
                )
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
        repr_str += f", cached_indices={len(self._cached_indices)}"
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
