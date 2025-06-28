"""
Improved Dilated Attention V2 with attention-specific buffer management.

This is a demonstration of how to integrate the AttentionBufferManager
for optimized memory allocation in attention mechanisms.
"""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .improved_dilated_attention import ImprovedDilatedAttention
from .core.attention_buffer_manager import (
    BufferType,
    create_attention_buffer_manager,
)


class ImprovedDilatedAttentionV2(ImprovedDilatedAttention):
    """
    Improved dilated attention with attention-specific buffer management.

    This implementation demonstrates the use of AttentionBufferManager
    for optimized allocation of Q, K, V, output, and temporary buffers.
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dropout: float = 0.0,
        use_tf32: bool = True,
        enable_buffer_manager: bool = True,
        enable_buffer_reuse: bool = True,
        enable_preallocation: bool = False,
        **kwargs,
    ):
        """
        Initialize improved dilated attention V2.

        Args:
            segment_lengths: List of segment lengths
            dilation_rates: List of dilation rates
            dropout: Dropout probability
            use_tf32: Enable TF32 for matmul
            enable_buffer_manager: Use attention buffer manager
            enable_buffer_reuse: Enable buffer reuse across iterations
            enable_preallocation: Pre-allocate common buffer sizes
            **kwargs: Additional arguments
        """
        # Disable the old memory pool in favor of buffer manager
        kwargs["enable_memory_pool"] = False
        super().__init__(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_tf32=use_tf32,
            **kwargs,
        )

        # Initialize buffer manager
        self.enable_buffer_manager = enable_buffer_manager
        self.buffer_manager = None

        if enable_buffer_manager:
            self.buffer_manager = create_attention_buffer_manager(
                enable_reuse=enable_buffer_reuse,
                enable_preallocation=enable_preallocation,
                enable_profiling=kwargs.get("enable_profiling", False),
                lightweight=kwargs.get("lightweight_pool", True),
            )

    def _allocate_buffer(
        self,
        buffer_type: BufferType,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        zero_init: Optional[bool] = None,
    ) -> Tensor:
        """
        Allocate a buffer using the buffer manager.

        Args:
            buffer_type: Type of buffer to allocate
            shape: Buffer shape
            dtype: Data type
            device: Target device
            zero_init: Whether to zero-initialize

        Returns:
            Allocated buffer
        """
        if self.buffer_manager is not None:
            return self.buffer_manager.allocate(
                buffer_type, shape, dtype, device, zero_init
            )
        else:
            # Fallback to direct allocation
            if zero_init:
                return torch.zeros(shape, dtype=dtype, device=device)
            else:
                return torch.empty(shape, dtype=dtype, device=device)

    def _deallocate_buffer(self, buffer: Tensor, buffer_type: BufferType) -> None:
        """Return buffer to manager for potential reuse."""
        if self.buffer_manager is not None:
            self.buffer_manager.deallocate(buffer, buffer_type)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass with attention-specific buffer management.

        This implementation demonstrates optimal buffer allocation
        for different components of the attention mechanism.
        """
        # Validate inputs
        self._validate_forward_inputs(query, key, value, attention_mask)

        b, n, h, d = query.shape
        device, dtype = query.device, query.dtype

        # Pre-allocate buffers if enabled
        if self.buffer_manager and hasattr(self.buffer_manager, "preallocate_buffers"):
            self.buffer_manager.preallocate_buffers(b, n, h, d, device)

        # Allocate output buffer using buffer manager
        out = self._allocate_buffer(
            BufferType.OUTPUT, (b, n, h, d), dtype, device, zero_init=True
        )

        # Get head groups
        group_sizes, head_ranges = self._get_head_groups(h)

        # Process each segment
        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths, strict=False)
        ):
            if g == 0 or n < s:
                continue

            hmin, hmax = head_ranges[i]
            offset = i % r

            # Process segment with optimized buffer allocation
            self._process_segment(
                query[:, :, hmin:hmax, :],
                key[:, :, hmin:hmax, :],
                value[:, :, hmin:hmax, :],
                out[:, :, hmin:hmax, :],
                s,
                r,
                offset,
                g,
                n,
                is_causal,
            )

        # Normalize output
        out.div_(self.num_groups)

        # Apply dropout
        out = self._apply_dropout(out)

        # Note: We don't deallocate the output buffer here as it's returned
        # The caller is responsible for managing the output buffer

        return out

    def _process_segment(
        self,
        q_slice: Tensor,
        k_slice: Tensor,
        v_slice: Tensor,
        out_slice: Tensor,
        segment_len: int,
        dilation_rate: int,
        offset: int,
        num_heads: int,
        seq_len: int,
        is_causal: bool,
    ) -> None:
        """Process a single segment with optimized buffer allocation."""
        b = q_slice.size(0)
        device, dtype = q_slice.device, q_slice.dtype

        # Reshape to segments
        q_seg = q_slice.view(b, seq_len // segment_len, segment_len, num_heads, -1)
        k_seg = k_slice.view(b, seq_len // segment_len, segment_len, num_heads, -1)
        v_seg = v_slice.view(b, seq_len // segment_len, segment_len, num_heads, -1)

        # Apply dilation if needed
        if dilation_rate > 1 or offset:
            # Get dilated indices
            cache_key = (segment_len, dilation_rate, offset, device)
            if cache_key not in self._cached_indices:
                self._cached_indices[cache_key] = torch.arange(
                    offset, segment_len, dilation_rate, device=device
                )
            idx = self._cached_indices[cache_key]

            # Apply dilation
            q_seg = q_seg[:, :, idx, :, :]
            k_seg = k_seg[:, :, idx, :, :]
            v_seg = v_seg[:, :, idx, :, :]

        # Flatten for attention computation
        bn = b * (seq_len // segment_len)
        dilated_s = q_seg.size(2)
        q_flat = q_seg.contiguous().view(bn, dilated_s, num_heads, -1)
        k_flat = k_seg.contiguous().view(bn, dilated_s, num_heads, -1)
        v_flat = v_seg.contiguous().view(bn, dilated_s, num_heads, -1)

        # Compute attention using SDPA
        x = F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back
        x_reshaped = x.view(b, seq_len // segment_len, dilated_s, num_heads, -1)

        # Handle dilation scatter
        if dilation_rate > 1 or offset:
            # Allocate temporary buffer for scatter operation
            temp_shape = (b, seq_len // segment_len, segment_len, num_heads, x.size(-1))
            temp_output = self._allocate_buffer(
                BufferType.TEMP, temp_shape, dtype, device, zero_init=True
            )

            # Scatter dilated results
            temp_output[:, :, idx, :, :] = x_reshaped

            # Add to output
            out_slice.add_(temp_output.reshape(b, seq_len, num_heads, -1))

            # Return temporary buffer to pool
            self._deallocate_buffer(temp_output, BufferType.TEMP)
        else:
            # Direct add without dilation
            out_slice.add_(x_reshaped.reshape(b, seq_len, num_heads, -1))

    def get_buffer_stats(self) -> dict:
        """Get buffer manager statistics."""
        if self.buffer_manager:
            return self.buffer_manager.get_stats()
        return {}

    def cleanup_buffers(self):
        """Clean up buffer manager caches."""
        if self.buffer_manager:
            self.buffer_manager.clear_cache()
        # Call parent cleanup if it exists
        if hasattr(super(), "cleanup_buffers"):
            super().cleanup_buffers()
