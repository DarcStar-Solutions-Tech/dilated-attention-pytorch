from typing import Sequence, Optional, Union, Tuple

import torch
from torch import nn, Tensor
from einops import rearrange

from dilated_attention_pytorch.improved_dilated_attention import ImprovedDilatedAttention


class ImprovedMultiheadDilatedAttention(nn.Module):
    """
    Improved multihead dilated attention that uses ImprovedDilatedAttention.
    
    This is a drop-in replacement for MultiheadDilatedAttention that leverages
    the performance optimizations in ImprovedDilatedAttention including:
    - TF32 optimization
    - torch.compile support
    - Automatic SDPA backend selection
    - More efficient memory usage
    
    Maintains the same interface as the original MultiheadDilatedAttention
    but with better performance characteristics.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        use_tf32: bool = True,
    ):
        """
        Initialize ImprovedMultiheadDilatedAttention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dilation_rates: Dilation rate for each segment
            segment_lengths: Length of each attention segment
            dropout: Dropout probability (default: 0.0)
            bias: Whether to use bias in linear projections (default: True)
            layer_norm: Whether to apply layer norm before output projection (default: True)
            layer_norm_eps: Layer norm epsilon (default: 1e-5)
            gamma_init: Initialization gain for MAGNETO architecture (default: 1.0)
            device: Device to place parameters on
            dtype: Data type for parameters
            use_tf32: Whether to enable TF32 optimization (default: True)
        """
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init
        self.embed_dim = embed_dim

        # Validation
        if not embed_dim % self.num_heads == 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        num_dilations = len(dilation_rates)
        num_segments = len(segment_lengths)
        if num_dilations != num_segments:
            raise ValueError(
                f"len(dilation_rates) ({num_dilations}) must be equal to "
                f"len(segment_lengths) ({num_segments})"
            )
        
        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        # Fused QKV projection for 3x memory efficiency
        self.qkv_proj = nn.Linear(
            embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype
        )
        
        # Use ImprovedDilatedAttention instead of DilatedAttention
        self.dilated_attention = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_tf32=use_tf32,
        )
        
        # Optional layer norm (for MAGNETO architecture)
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        
        # Output projection
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters following MAGNETO architecture guidelines."""
        # Initialize fused QKV projection with proper gains
        # Split initialization for Q, K, V parts of the fused weight
        embed_dim = self.embed_dim
        
        # Get weight slices for Q, K, V
        q_weight = self.qkv_proj.weight[:embed_dim, :]
        k_weight = self.qkv_proj.weight[embed_dim:2*embed_dim, :]
        v_weight = self.qkv_proj.weight[2*embed_dim:, :]
        
        # Standard Xavier for Q and K
        nn.init.xavier_normal_(q_weight)
        nn.init.xavier_normal_(k_weight)
        
        # MAGNETO initialization for V with gain
        nn.init.xavier_normal_(v_weight, gain=self.gamma_init)
        
        # Initialize bias if present
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0)

        # MAGNETO initialization for output projection
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        is_causal: bool = False
    ) -> Tuple[Tensor, None]:
        """
        Forward pass through improved multihead dilated attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            is_causal: Whether to apply causal masking (default: False)
            
        Returns:
            Tuple of (attention_output, None) where attention_output has shape
            (batch_size, seq_len, embed_dim). Second element is None for
            compatibility with nn.MultiheadAttention interface.
        """
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - head dimension
        #   embed_dim = h * d
        #
        # Input shape: (b, n, embed_dim)
        
        # Optimized fused QKV projection - 3x fewer memory allocations
        batch_size, seq_len, _ = query.shape
        head_dim = self.embed_dim // self.num_heads
        
        # Single fused projection for all inputs
        qkv_query = self.qkv_proj(query)
        qkv_key = self.qkv_proj(key) if key is not query else qkv_query
        qkv_value = self.qkv_proj(value) if value is not query else qkv_query
        
        # Split and reshape QKV in one operation
        # Shape: [b, n, 3*embed_dim] -> [b, n, 3, h, d] -> 3 * [b, n, h, d]
        q = qkv_query[:, :, :self.embed_dim].view(batch_size, seq_len, self.num_heads, head_dim)
        k = qkv_key[:, :, self.embed_dim:2*self.embed_dim].view(batch_size, seq_len, self.num_heads, head_dim)
        v = qkv_value[:, :, 2*self.embed_dim:].view(batch_size, seq_len, self.num_heads, head_dim)
        
        # Apply improved dilated attention
        x = self.dilated_attention(q, k, v, is_causal=is_causal)
        
        # Reshape back to original format: (b, n, h, d) -> (b, n, embed_dim)
        # Use view instead of rearrange for better memory efficiency
        x = x.view(batch_size, seq_len, self.embed_dim)

        # NOTE: This follows the MAGNETO architecture, which applies an extra 
        # layer norm before the linear output projection. This is different from 
        # standard nn.MultiheadAttention! The cross-attention layer in the
        # MAGNETO decoder does not include this layer norm, so users have the 
        # option to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        
        # Final linear projection
        x = self.out_proj(x)

        # Return tuple for compatibility with nn.MultiheadAttention interface
        # (attention weights are not computed/returned by dilated attention)
        return x, None

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"layer_norm={self.layer_norm}, gamma_init={self.gamma_init}"
        )


# Optional: Compile the entire module for maximum performance
# Uncomment the following line if you want torch.compile optimization
# ImprovedMultiheadDilatedAttention = torch.compile(
#     ImprovedMultiheadDilatedAttention, 
#     fullgraph=True
# )