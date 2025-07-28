"""
Hilbert Attention Mixin for easy integration of HilbertAttentionCore.

This mixin provides a simple way to add Hilbert curve optimization to any
attention implementation.
"""

import torch
from typing import Optional, Dict

from ..kernels.hilbert_attention_core import (
    HilbertAttentionCore,
    create_hilbert_mapping,
)


class HilbertAttentionMixin:
    """
    Mixin to add Hilbert curve optimization to attention mechanisms.

    This can be used in two ways:
    1. Just for Hilbert ordering (using existing attention computation)
    2. Full HilbertAttentionCore integration (Triton kernels + custom backward)
    """

    def setup_hilbert_attention(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        use_hilbert_core: bool = False,
        use_custom_backward: bool = True,
    ):
        """
        Initialize Hilbert attention components.

        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            segment_size: Size of attention segments
            dilation_rate: Dilation rate for attention
            dropout: Dropout probability
            use_hilbert_core: If True, use full HilbertAttentionCore
            use_custom_backward: If True, use optimized backward pass
        """
        self._hilbert_cache: Dict[int, torch.Tensor] = {}
        self._inverse_hilbert_cache: Dict[int, torch.Tensor] = {}
        self.use_hilbert_core = use_hilbert_core

        if use_hilbert_core:
            # Full integration with Triton kernels
            self.hilbert_attention = HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                dropout=dropout,
                use_custom_backward=use_custom_backward,
            )

    def get_hilbert_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert indices for a given sequence length."""
        if seq_len not in self._hilbert_cache:
            indices = create_hilbert_mapping(seq_len).to(device)
            self._hilbert_cache[seq_len] = indices

            # Also cache inverse mapping
            inverse = torch.zeros_like(indices)
            inverse[indices] = torch.arange(seq_len, device=device, dtype=indices.dtype)
            self._inverse_hilbert_cache[seq_len] = inverse

        return self._hilbert_cache[seq_len]

    def get_inverse_hilbert_indices(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Get cached inverse Hilbert indices."""
        if seq_len not in self._inverse_hilbert_cache:
            # This will create both forward and inverse mappings
            self.get_hilbert_indices(seq_len, device)
        return self._inverse_hilbert_cache[seq_len]

    def apply_hilbert_ordering(
        self,
        tensor: torch.Tensor,
        inverse: bool = False,
        dim: int = 1,  # Usually sequence dimension
    ) -> torch.Tensor:
        """
        Apply Hilbert ordering to a tensor.

        Args:
            tensor: Input tensor
            inverse: If True, apply inverse Hilbert ordering
            dim: Dimension to apply ordering along (default: 1 for sequence)

        Returns:
            Reordered tensor
        """
        seq_len = tensor.shape[dim]
        device = tensor.device

        if inverse:
            indices = self.get_inverse_hilbert_indices(seq_len, device)
        else:
            indices = self.get_hilbert_indices(seq_len, device)

        # Apply ordering along specified dimension
        return torch.index_select(tensor, dim, indices)

    def compute_hilbert_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        use_hilbert_ordering: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute attention using Hilbert optimization.

        This method can be used as a drop-in replacement for standard attention.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim] or [batch, seq_len, hidden_dim]
            k: Key tensor
            v: Value tensor
            use_hilbert_ordering: Whether to apply Hilbert ordering
            **kwargs: Additional arguments for attention computation

        Returns:
            Attention output with same shape as input
        """
        if not hasattr(self, "hilbert_attention") or not self.use_hilbert_core:
            # Just apply Hilbert ordering, use existing attention
            if use_hilbert_ordering:
                q_ordered = self.apply_hilbert_ordering(q, inverse=False)
                k_ordered = self.apply_hilbert_ordering(k, inverse=False)
                v_ordered = self.apply_hilbert_ordering(v, inverse=False)

                # Call the parent class attention method
                # This assumes the parent class has a method like _compute_attention
                if hasattr(self, "_compute_attention"):
                    output = self._compute_attention(
                        q_ordered, k_ordered, v_ordered, **kwargs
                    )
                else:
                    # Fallback to scaled dot-product attention
                    output = self._scaled_dot_product_attention(
                        q_ordered, k_ordered, v_ordered, **kwargs
                    )

                # Apply inverse ordering to output
                return self.apply_hilbert_ordering(output, inverse=True)
            else:
                # No Hilbert ordering requested
                if hasattr(self, "_compute_attention"):
                    return self._compute_attention(q, k, v, **kwargs)
                else:
                    return self._scaled_dot_product_attention(q, k, v, **kwargs)
        else:
            # Use full HilbertAttentionCore
            # Need to reshape if input is 4D
            if q.dim() == 4:
                batch, seq_len, num_heads, head_dim = q.shape
                hidden_dim = num_heads * head_dim

                # Combine heads dimension
                q_3d = q.reshape(batch, seq_len, hidden_dim)

                # Use query as input (HilbertAttentionCore handles QKV internally)
                output = self.hilbert_attention(q_3d, use_hilbert=use_hilbert_ordering)

                # Reshape back to 4D
                return output.reshape(batch, seq_len, num_heads, head_dim)
            else:
                # Already 3D, use directly
                return self.hilbert_attention(q, use_hilbert=use_hilbert_ordering)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Fallback scaled dot-product attention implementation."""
        # Get dimensions
        if q.dim() == 4:
            batch, seq_len, num_heads, head_dim = q.shape
            # Reshape for batch matrix multiply
            q = q.transpose(1, 2).reshape(batch * num_heads, seq_len, head_dim)
            k = k.transpose(1, 2).reshape(batch * num_heads, seq_len, head_dim)
            v = v.transpose(1, 2).reshape(batch * num_heads, seq_len, head_dim)
            reshape_output = True
        else:
            reshape_output = False

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        if dropout_p > 0 and self.training:
            attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

        # Apply attention to values
        output = torch.bmm(attn_weights, v)

        # Reshape back if needed
        if reshape_output:
            output = output.reshape(batch, num_heads, seq_len, head_dim)
            output = output.transpose(1, 2)

        return output


# Example usage in existing class:
"""
class MyRingAttention(nn.Module, HilbertAttentionMixin):
    def __init__(self, dim, heads, ...):
        super().__init__()
        self.dim = dim
        self.heads = heads
        
        # Setup Hilbert attention
        self.setup_hilbert_attention(
            hidden_dim=dim,
            num_heads=heads,
            use_hilbert_core=True,  # Use full Triton implementation
        )
        
    def forward(self, q, k, v):
        # Use Hilbert-optimized attention
        return self.compute_hilbert_attention(q, k, v)
"""
