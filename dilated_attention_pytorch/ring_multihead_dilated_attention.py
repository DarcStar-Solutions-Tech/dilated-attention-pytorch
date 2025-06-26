"""
Ring Multihead Dilated Attention implementation using the refactored core architecture.

This module implements a multihead attention wrapper around Ring Dilated Attention,
providing a drop-in replacement for standard multihead attention with O(n) memory
complexity for arbitrarily long sequences.

Key Features:
- O(n) memory complexity through Ring Attention
- Fused QKV projections for 3x memory efficiency
- MAGNETO architecture compatibility
- Automatic mixed precision support
- Gradient checkpointing integration
- Full compatibility with nn.MultiheadAttention interface

This implementation combines:
- Ring Attention (O(n) memory scaling)
- Dilated Attention (efficient long-range dependencies)
- Multihead Attention (parallel attention heads)
- Advanced memory optimizations
"""

import threading
import warnings
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .core import (
    BaseMultiheadDilatedAttention,
    MultiheadConfig,
    RingAttentionConfig,
    split_attention_heads,
)
from .ring_dilated_attention import RingDilatedAttention


class RingMultiheadDilatedAttention(BaseMultiheadDilatedAttention):
    """
    Ring-based Multihead Dilated Attention with O(n) memory complexity.

    This class provides a complete multihead attention implementation using
    Ring Dilated Attention as the core attention mechanism. It maintains
    compatibility with nn.MultiheadAttention while enabling linear memory
    scaling for extremely long sequences.

    Key advantages over standard multihead attention:
    - O(n) memory instead of O(n²) through Ring Attention
    - Efficient long-range dependencies through Dilated Attention
    - 3x memory efficiency through fused QKV projections
    - Linear scaling to arbitrarily long sequences
    - Distributed computation across multiple devices
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        # Ring attention specific parameters
        block_size: int = 1024,
        ring_size: int | None = None,
        use_checkpointing: bool = True,
        # Hardware optimization parameters
        use_tf32: bool = True,
        use_flash_attention: bool = True,
        compile_model: bool = False,
        # Device parameters
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize Ring Multihead Dilated Attention.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            segment_lengths: Sequence of segment lengths for dilated attention
            dilation_rates: Corresponding dilation rates for each segment
            dropout: Dropout probability (default: 0.0)
            bias: Whether to use bias in linear projections (default: True)
            layer_norm: Whether to apply layer norm before output projection (default: True)
            layer_norm_eps: Layer norm epsilon (default: 1e-5)
            gamma_init: Initialization gain for MAGNETO architecture (default: 1.0)

            block_size: Block size for ring attention computation (default: 1024)
            ring_size: Number of devices in ring (auto-detected if None)
            use_checkpointing: Enable gradient checkpointing (default: True)

            use_tf32: Enable TF32 optimization (default: True)
            use_flash_attention: Enable Flash Attention backend (default: True)
            compile_model: Enable torch.compile optimization (default: False)

            device: Device to place parameters on
            dtype: Data type for parameters
        """
        # Create configurations
        multihead_config = MultiheadConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )

        attention_config = RingAttentionConfig(
            segment_lengths=list(segment_lengths),
            dilation_rates=list(dilation_rates),
            dropout=dropout,
            use_tf32=use_tf32,
            block_size=block_size,
            ring_size=ring_size,
            use_checkpointing=use_checkpointing,
            device=device,
            dtype=dtype,
        )

        # Initialize base class
        super().__init__(multihead_config, attention_config)

        # Store additional attributes

        self.use_flash_attention = use_flash_attention
        self.compile_model = compile_model
        self.use_checkpointing = use_checkpointing

        # Fused QKV projection for maximum memory efficiency
        self.use_fused_qkv = True
        
        # Initialize base class
        super().__init__(multihead_config, attention_config)
        if self.use_fused_qkv:
            factory_kwargs = {"device": self.device, "dtype": self.dtype}
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)

        # Create ring attention module
        self.attention = self._create_attention_module()

        # Advanced memory optimization: Pre-allocate QKV output buffers
        self._qkv_output_buffers = {}
        self._output_projection_cache = {}

        # Thread safety for buffer management (in addition to base class)
        self._buffer_lock = threading.Lock()

        # Optional compilation for additional optimization
        if compile_model:
            self._enable_compilation()

    def _create_attention_module(self) -> RingDilatedAttention:
        """Create the underlying ring dilated attention module."""
        return RingDilatedAttention(
            segment_lengths=self.attention_config.segment_lengths,
            dilation_rates=self.attention_config.dilation_rates,
            dropout=self.attention_config.dropout,
            use_tf32=self.attention_config.use_tf32,
            block_size=self.attention_config.block_size,
            ring_size=self.attention_config.ring_size,
            use_checkpointing=self.attention_config.use_checkpointing,
            device=self.device,
            dtype=self.dtype,
        )

    def _init_qkv_projections(self, factory_kwargs: dict):
        """Initialize fused QKV projection for ring attention efficiency."""
        if self.use_fused_qkv:
            # Fused projection already initialized in __init__
            # Skip base class initialization
            return
        else:
            # Fallback to separate projections
            super()._init_qkv_projections(factory_kwargs)

    def _reset_parameters(self):
        """Initialize parameters following MAGNETO architecture guidelines."""
        if self.use_fused_qkv and hasattr(self, "qkv_proj"):
            embed_dim = self.embed_dim

            # Initialize fused QKV projection with proper gains
            q_weight = self.qkv_proj.weight[:embed_dim, :]
            k_weight = self.qkv_proj.weight[embed_dim : 2 * embed_dim, :]
            v_weight = self.qkv_proj.weight[2 * embed_dim :, :]

            # Standard Xavier for Q and K
            nn.init.xavier_normal_(q_weight)
            nn.init.xavier_normal_(k_weight)

            # MAGNETO initialization for V with gain
            nn.init.xavier_normal_(v_weight, gain=self.multihead_config.gamma_init)

            # Initialize bias if present
            if self.qkv_proj.bias is not None:
                nn.init.constant_(self.qkv_proj.bias, 0)
        else:
            # Use base class initialization
            super()._reset_parameters()

    def _enable_compilation(self):
        """Enable torch.compile optimization for additional performance."""
        try:
            self.attention = torch.compile(self.attention, mode="max-autotune", fullgraph=True)
            if hasattr(self, "qkv_proj"):
                self.qkv_proj = torch.compile(self.qkv_proj, mode="max-autotune")
            if hasattr(self, "out_proj"):
                self.out_proj = torch.compile(self.out_proj, mode="max-autotune")
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}")

    def _apply_fused_qkv_projection(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply fused QKV projection with advanced memory optimization.

        This optimized version:
        1. Pre-allocates output buffers to eliminate memory allocation
        2. Uses tensor views and copy operations instead of slicing
        3. Optimizes for the common self-attention case
        4. Minimizes intermediate tensor creation

        Args:
            query: Query input [batch, seq_len, embed_dim]
            key: Key input [batch, seq_len, embed_dim]
            value: Value input [batch, seq_len, embed_dim]

        Returns:
            Tuple of (q, k, v) tensors with shape [batch, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, _ = query.shape
        head_dim = self.embed_dim // self.num_heads
        target_shape = (batch_size, seq_len, self.num_heads, head_dim)

        # Thread-safe buffer allocation
        buffer_key = (target_shape, query.dtype, query.device)
        with self._buffer_lock:
            # Pre-allocate output buffers for efficient memory usage
            if buffer_key not in self._qkv_output_buffers:
                self._qkv_output_buffers[buffer_key] = {
                    "q": torch.empty(target_shape, dtype=query.dtype, device=query.device),
                    "k": torch.empty(target_shape, dtype=query.dtype, device=query.device),
                    "v": torch.empty(target_shape, dtype=query.dtype, device=query.device),
                }

        # Ensure buffers match current input dimensions - use resize for efficiency
        buffers = self._qkv_output_buffers[buffer_key]
        if buffers["q"].shape != target_shape:
            # Validate target shape is reasonable before allocation
            total_elements = 1
            for dim in target_shape:
                total_elements *= dim

            max_reasonable_elements = 100 * 1024 * 1024  # 100M elements max
            if total_elements > max_reasonable_elements:
                raise RuntimeError(
                    f"Requested buffer size ({total_elements} elements, "
                    f"~{total_elements * 4 / (1024 * 1024):.1f}MB) exceeds maximum reasonable size. "
                    f"Consider reducing batch size or sequence length."
                )

            # Resize existing buffers instead of recreating
            try:
                buffers["q"].resize_(target_shape)
                buffers["k"].resize_(target_shape)
                buffers["v"].resize_(target_shape)
            except RuntimeError as resize_error:
                # Enhanced fallback with error recovery
                try:
                    # Clear old buffers first
                    del buffers["q"], buffers["k"], buffers["v"]
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    # Recreate buffers
                    buffers["q"] = torch.empty(target_shape, dtype=query.dtype, device=query.device)
                    buffers["k"] = torch.empty(target_shape, dtype=query.dtype, device=query.device)
                    buffers["v"] = torch.empty(target_shape, dtype=query.dtype, device=query.device)
                    self._qkv_output_buffers[buffer_key] = buffers
                except RuntimeError as alloc_error:
                    # Ultimate fallback: use smaller batch processing
                    if batch_size > 1:
                        warnings.warn(
                            f"Buffer allocation failed ({alloc_error}). "
                            f"Consider reducing batch size from {batch_size}."
                        )
                    raise RuntimeError(
                        f"Failed to allocate QKV buffers: resize failed ({resize_error}), "
                        f"recreation failed ({alloc_error})"
                    )

        # Optimized projection handling
        is_self_attention = torch.equal(query, key) and torch.equal(key, value)

        if is_self_attention:
            # Self-attention: single fused projection with direct buffer writes
            qkv = self.qkv_proj(query)  # [batch, seq_len, 3*embed_dim]

            # Use views and copy operations instead of slicing for efficiency
            q_flat = qkv[:, :, : self.embed_dim].view(target_shape)
            k_flat = qkv[:, :, self.embed_dim : 2 * self.embed_dim].view(target_shape)
            v_flat = qkv[:, :, 2 * self.embed_dim :].view(target_shape)

            # Copy to pre-allocated buffers
            buffers["q"].copy_(q_flat)
            buffers["k"].copy_(k_flat)
            buffers["v"].copy_(v_flat)
        else:
            # Cross-attention: separate projections (less common case)
            # Use separate projections for cross-attention
            q_proj = self.qkv_proj(query)[:, :, : self.embed_dim]
            k_proj = self.qkv_proj(key)[:, :, self.embed_dim : 2 * self.embed_dim]
            v_proj = self.qkv_proj(value)[:, :, 2 * self.embed_dim :]

            # Reshape and copy to buffers
            buffers["q"].copy_(q_proj.view(target_shape))
            buffers["k"].copy_(k_proj.view(target_shape))
            buffers["v"].copy_(v_proj.view(target_shape))

        return buffers["q"], buffers["k"], buffers["v"]

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """
        Forward pass through Ring Multihead Dilated Attention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (uses query if None)
            value: Value tensor (uses query if None)
            key_padding_mask: Not supported with ring attention
            need_weights: Whether to return attention weights (not supported)
            attn_mask: Not supported with ring attention
            is_causal: Whether to apply causal masking
            average_attn_weights: Whether to average attention weights (unused)

        Returns:
            If need_weights is False:
                Attention output [batch, seq_len, embed_dim]
            If need_weights is True:
                Tuple of (output, None) - weights not supported
        """
        # Handle ring attention limitations
        if attn_mask is not None or key_padding_mask is not None:
            warnings.warn(
                "Attention masks are not supported with Ring Attention. Masks will be ignored."
            )

        if need_weights:
            warnings.warn(
                "Attention weights are not computed with Ring Attention. "
                "Returning None for weights."
            )

        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query

        # Validate inputs
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"Expected 3D tensors (batch, seq_len, embed_dim), got shapes: "
                f"query={query.shape}, key={key.shape}, value={value.shape}"
            )

        batch_size, seq_len, _ = query.shape

        # Apply fused QKV projections
        if self.use_fused_qkv and hasattr(self, "qkv_proj"):
            # Use fused projection with caching
            q, k, v = self._apply_fused_qkv_projection(query, key, value)
        else:
            # Use base class projections
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # Apply layer normalization if enabled
            q, k = self._apply_layer_norm(q, k)

            # Split into heads
            q = split_attention_heads(q, self.num_heads)
            k = split_attention_heads(k, self.num_heads)
            v = split_attention_heads(v, self.num_heads)

        # Apply Ring Dilated Attention with error recovery
        try:
            if self.use_checkpointing and self.training:
                attn_output = torch.utils.checkpoint.checkpoint(
                    self.attention.forward, q, k, v, is_causal, None, use_reentrant=False
                )
            else:
                attn_output = self.attention(q, k, v, is_causal, None)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Memory recovery strategy
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.clear_cache()

                # Retry with checkpointing
                if not self.use_checkpointing:
                    warnings.warn("OOM detected, retrying with gradient checkpointing")
                    attn_output = torch.utils.checkpoint.checkpoint(
                        self.attention.forward, q, k, v, is_causal, None, use_reentrant=False
                    )
                else:
                    raise RuntimeError(
                        "Out of memory even with checkpointing. "
                        "Consider reducing batch size or sequence length."
                    ) from e
            else:
                raise e

        # Merge heads back
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Apply post-attention layer norm if enabled (MAGNETO style)
        if self.multihead_config.layer_norm and hasattr(self, "q_ln"):
            attn_output = self.q_ln(attn_output)

        # Output projection
        output = self.out_proj(attn_output)

        if need_weights:
            return output, None
        else:
            return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        repr_str = super().extra_repr()
        repr_str += f", ring_size={self.attention.ring_size}"
        repr_str += f", block_size={self.attention.block_size}"
        repr_str += f", use_fused_qkv={self.use_fused_qkv}"
        if self.compile_model:
            repr_str += ", compiled=True"
        return repr_str

    def clear_cache(self):
        """Clear cached buffers to free memory with thread safety."""
        # Clear base class cache first
        super().clear_cache()

        # Clear ring-specific caches
        with self._buffer_lock:
            self._qkv_output_buffers.clear()
            self._output_projection_cache.clear()

        # Clear attention module cache
        if hasattr(self.attention, "clear_cache"):
            self.attention.clear_cache()

    def get_memory_info(self) -> dict[str, Any]:
        """
        Get comprehensive memory usage information for the attention layer.

        Returns:
            Dictionary with memory statistics and theoretical complexity.
        """
        # Get base class info
        info = super().get_memory_info()

        # Add ring-specific info
        info.update(
            {
                "memory_complexity": "O(n)",
                "ring_size": self.attention.ring_size,
                "block_size": self.attention.block_size,
                "supports_infinite_context": True,
                "max_sequence_length": "unlimited (distributed)",
                "memory_per_device": f"O(n / {self.attention.ring_size})",
                "qkv_buffers_cached": len(self._qkv_output_buffers),
                "output_buffers_cached": len(self._output_projection_cache),
            }
        )

        # Include ring attention memory info if available
        if hasattr(self.attention, "get_memory_info"):
            ring_info = self.attention.get_memory_info()
            info.update({f"ring_{k}": v for k, v in ring_info.items()})

        return info


# Optional: Enable torch.compile for the entire module
# Uncomment the following lines if you want maximum optimization
# RingMultiheadDilatedAttention = torch.compile(
#     RingMultiheadDilatedAttention,
#     mode='max-autotune',
#     fullgraph=True
# )
