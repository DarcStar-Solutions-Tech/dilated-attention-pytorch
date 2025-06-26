"""
Refactored Transformer layers using the new core architecture.

This module provides transformer encoder and decoder layers that use
dilated attention, with support for the factory pattern and configuration system.
"""

from typing import Callable, Optional, Sequence, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn

from .core import (
    create_multihead_dilated_attention,
    DilatedAttentionConfig,
    MultiheadConfig,
)


@dataclass
class TransformerLayerConfig:
    """Configuration for transformer layers."""
    
    d_model: int
    nhead: int
    segment_lengths: Sequence[int]
    dilation_rates: Sequence[int]
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: Union[str, Callable[[Tensor], Tensor]] = F.relu
    layer_norm_eps: float = 1e-5
    gamma_init: float = 1.0
    attention_type: str = "auto"  # For factory pattern
    device: Optional[Union[torch.device, str]] = None
    dtype: Optional[torch.dtype] = None
    
    def __post_init__(self):
        """Validate and process configuration."""
        # Convert string activation to function
        if isinstance(self.activation, str):
            self.activation = _get_activation_fn(self.activation)
        
        # Convert device if string
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


class DilatedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with dilated attention.
    
    This refactored version uses the factory pattern and configuration system
    for cleaner initialization and better flexibility.
    
    Features:
    - Sub-LayerNorm like in MAGNETO
    - Factory pattern for attention selection
    - Configuration-based initialization
    """
    
    def __init__(
        self,
        config: Optional[TransformerLayerConfig] = None,
        **kwargs
    ):
        """
        Initialize encoder layer.
        
        Args:
            config: TransformerLayerConfig object
            **kwargs: Individual parameters (for backward compatibility)
        """
        super().__init__()
        
        # Handle both config and kwargs initialization
        if config is None:
            config = TransformerLayerConfig(**kwargs)
        
        self.config = config
        
        # Store commonly used values
        self.d_model = config.d_model
        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout)
        
        # Create attention configurations
        attention_config = DilatedAttentionConfig(
            segment_lengths=list(config.segment_lengths),
            dilation_rates=list(config.dilation_rates),
            dropout=config.dropout,
            device=config.device,
            dtype=config.dtype,
        )
        
        multihead_config = MultiheadConfig(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            layer_norm=True,
            layer_norm_eps=config.layer_norm_eps,
            gamma_init=config.gamma_init,
            device=config.device,
            dtype=config.dtype,
        )
        
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            config.d_model, 
            eps=config.layer_norm_eps, 
            device=config.device, 
            dtype=config.dtype
        )
        
        # Use factory to create attention
        self.self_attn = create_multihead_dilated_attention(
            config.attention_type,
            multihead_config=multihead_config,
            attention_config=attention_config,
        )
        
        # Feedforward block
        self.norm2 = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.linear1 = nn.Linear(
            config.d_model,
            config.dim_feedforward,
            device=config.device,
            dtype=config.dtype
        )
        self.norm3 = nn.LayerNorm(
            config.dim_feedforward,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.linear2 = nn.Linear(
            config.dim_feedforward,
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters following MAGNETO strategy."""
        nn.init.xavier_normal_(self.linear1.weight, gain=self.config.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.config.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)
    
    def _self_attention_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        """Self-attention sub-layer with residual connection."""
        x = self.norm1(x)
        x = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)
        return x
    
    def _feedforward_block(self, x: Tensor) -> Tensor:
        """Feedforward sub-layer with residual connection."""
        x = self.norm2(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
    def forward(self, src: Tensor, is_causal: bool = False) -> Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            src: Input tensor of shape (batch, seq_len, d_model)
            is_causal: Whether to use causal masking
            
        Returns:
            Output tensor of same shape as input
        """
        x = src
        x = x + self._self_attention_block(x, is_causal=is_causal)
        x = x + self._feedforward_block(x)
        return x


class DilatedTransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with dilated attention.
    
    This refactored version uses the factory pattern and configuration system
    for cleaner initialization and better flexibility.
    
    Features:
    - Sub-LayerNorm like in MAGNETO
    - Factory pattern for attention selection
    - Separate self-attention and cross-attention
    """
    
    def __init__(
        self,
        config: Optional[TransformerLayerConfig] = None,
        **kwargs
    ):
        """
        Initialize decoder layer.
        
        Args:
            config: TransformerLayerConfig object
            **kwargs: Individual parameters (for backward compatibility)
        """
        super().__init__()
        
        # Handle both config and kwargs initialization
        if config is None:
            config = TransformerLayerConfig(**kwargs)
        
        self.config = config
        
        # Store commonly used values
        self.d_model = config.d_model
        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout)
        
        # Create attention configurations
        attention_config = DilatedAttentionConfig(
            segment_lengths=list(config.segment_lengths),
            dilation_rates=list(config.dilation_rates),
            dropout=config.dropout,
            device=config.device,
            dtype=config.dtype,
        )
        
        # Self-attention config (no layer norm)
        self_attn_multihead_config = MultiheadConfig(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            layer_norm=False,
            gamma_init=config.gamma_init,
            device=config.device,
            dtype=config.dtype,
        )
        
        # Cross-attention config (with layer norm)
        cross_attn_multihead_config = MultiheadConfig(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            layer_norm=True,
            layer_norm_eps=config.layer_norm_eps,
            gamma_init=config.gamma_init,
            device=config.device,
            dtype=config.dtype,
        )
        
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.self_attn = create_multihead_dilated_attention(
            config.attention_type,
            multihead_config=self_attn_multihead_config,
            attention_config=attention_config,
        )
        
        # Cross-attention block
        self.norm2 = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.multihead_attn = create_multihead_dilated_attention(
            config.attention_type,
            multihead_config=cross_attn_multihead_config,
            attention_config=attention_config,
        )
        
        # Feedforward block
        self.norm3 = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.linear1 = nn.Linear(
            config.d_model,
            config.dim_feedforward,
            device=config.device,
            dtype=config.dtype
        )
        self.norm4 = nn.LayerNorm(
            config.dim_feedforward,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.linear2 = nn.Linear(
            config.dim_feedforward,
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters following MAGNETO strategy."""
        nn.init.xavier_normal_(self.linear1.weight, gain=self.config.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.config.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)
    
    def _self_attention_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        """Self-attention sub-layer with residual connection."""
        x = self.norm1(x)
        x = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)
        return x
    
    def _multihead_attention_block(
        self, x: Tensor, memory: Tensor, is_causal: bool = False
    ) -> Tensor:
        """Cross-attention sub-layer with residual connection."""
        x = self.norm2(x)
        x = self.multihead_attn(x, memory, memory, is_causal=is_causal)
        x = self.dropout(x)
        return x
    
    def _feedforward_block(self, x: Tensor) -> Tensor:
        """Feedforward sub-layer with residual connection."""
        x = self.norm3(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            tgt: Target tensor of shape (batch, seq_len, d_model)
            memory: Memory tensor from encoder
            is_causal: Whether to use causal masking for self-attention
            memory_is_causal: Whether to use causal masking for cross-attention
            
        Returns:
            Output tensor of same shape as tgt
        """
        x = tgt
        x = x + self._self_attention_block(x, is_causal=is_causal)
        x = x + self._multihead_attention_block(x, memory, is_causal=memory_is_causal)
        x = x + self._feedforward_block(x)
        return x


# Convenience functions for creating layers
def create_encoder_layer(
    attention_type: str = "auto",
    d_model: int = 512,
    nhead: int = 8,
    segment_lengths: Sequence[int] = (2048, 4096, 8192),
    dilation_rates: Sequence[int] = (1, 2, 4),
    **kwargs
) -> DilatedTransformerEncoderLayer:
    """
    Create a transformer encoder layer with specified attention type.
    
    Args:
        attention_type: Type of attention ("auto", "standard", "improved", "ring")
        d_model: Model dimension
        nhead: Number of attention heads
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for each segment
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured encoder layer
    """
    config = TransformerLayerConfig(
        d_model=d_model,
        nhead=nhead,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        attention_type=attention_type,
        **kwargs
    )
    return DilatedTransformerEncoderLayer(config)


def create_decoder_layer(
    attention_type: str = "auto",
    d_model: int = 512,
    nhead: int = 8,
    segment_lengths: Sequence[int] = (2048, 4096, 8192),
    dilation_rates: Sequence[int] = (1, 2, 4),
    **kwargs
) -> DilatedTransformerDecoderLayer:
    """
    Create a transformer decoder layer with specified attention type.
    
    Args:
        attention_type: Type of attention ("auto", "standard", "improved", "ring")
        d_model: Model dimension
        nhead: Number of attention heads
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for each segment
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured decoder layer
    """
    config = TransformerLayerConfig(
        d_model=d_model,
        nhead=nhead,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        attention_type=attention_type,
        **kwargs
    )
    return DilatedTransformerDecoderLayer(config)