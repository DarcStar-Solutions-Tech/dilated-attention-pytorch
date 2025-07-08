"""
Wrapper for HilbertAttentionCore to match the standard q,k,v interface.
"""

import torch
import torch.nn as nn
from .hilbert_attention_core import HilbertAttentionCore


class HilbertAttentionTritonWrapper(nn.Module):
    """
    Wrapper that adapts HilbertAttentionCore to accept separate q,k,v tensors.

    This allows HilbertAttentionCore to be used with the standard benchmark interface
    that expects forward(q, k, v) instead of forward(x).
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # For compatibility, we use the first segment length and dilation rate
        segment_size = segment_lengths[0] if segment_lengths else 128
        dilation_rate = dilation_rates[0] if dilation_rates else 1

        # Extract num_heads and head_dim from kwargs or use defaults
        self.num_heads = kwargs.get("num_heads", 8)
        self.head_dim = kwargs.get("head_dim", 64)
        hidden_dim = self.num_heads * self.head_dim

        # Create the underlying Hilbert attention
        self.attention = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
            use_custom_backward=True,  # Use optimized backward pass
        )

        # Create projection layers to convert from q,k,v format to x format
        # Input is [batch, seq, heads * head_dim], output is [batch, seq, hidden_dim]
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection to convert back
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass that accepts separate q,k,v tensors.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking (not supported)

        Returns:
            Output tensor [batch, seq_len, num_heads, head_dim]
        """
        B, M, H, D = q.shape

        # Reshape q,k,v to [batch, seq_len, hidden_dim] for processing
        q_flat = q.reshape(B, M, -1)  # [B, M, H*D]
        k_flat = k.reshape(B, M, -1)
        v_flat = v.reshape(B, M, -1)

        # Project q,k,v through linear layers
        q_proj = self.q_proj(q_flat)
        k_proj = self.k_proj(k_flat)
        v_proj = self.v_proj(v_flat)

        # Combine into single tensor for HilbertAttentionCore
        # This maintains gradient flow through all three inputs
        x = (q_proj + k_proj + v_proj) / 3.0

        # Apply Hilbert attention
        output = self.attention(x, use_hilbert=True)

        # Project output and reshape back to [B, M, H, D]
        output = self.out_proj(output)
        output = output.reshape(B, M, H, D)

        return output


# For backward compatibility, also export a version that exactly matches the expected name
class HilbertAttentionTritonFixed(HilbertAttentionTritonWrapper):
    """Alias for backward compatibility with benchmarks."""

    pass
