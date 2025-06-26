"""
Attention utilities for dilated attention implementations.

This module provides common utilities for attention computation, pattern generation,
and optimization strategies used across all dilated attention implementations.
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any
import math
import warnings

from ..core.constants import HAS_FLASH_ATTN, HAS_XFORMERS, HAS_SDPA


def compute_attention_scores(
    q: Tensor,
    k: Tensor, 
    scale: Optional[float] = None,
    attention_mask: Optional[Tensor] = None,
    is_causal: bool = False
) -> Tensor:
    """
    Compute scaled dot-product attention scores.
    
    Args:
        q: Query tensor [..., seq_len, head_dim]
        k: Key tensor [..., seq_len, head_dim]
        scale: Scaling factor (default: 1/sqrt(head_dim))
        attention_mask: Optional attention mask
        is_causal: Whether to apply causal masking
        
    Returns:
        Attention scores [..., seq_len, seq_len]
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask if requested
    if is_causal:
        seq_len = q.shape[-2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply attention mask if provided
    if attention_mask is not None:
        scores = scores + attention_mask
    
    return scores


def apply_dilated_attention_pattern(
    scores: Tensor,
    segment_length: int,
    dilation_rate: int,
    group_idx: int,
    total_groups: int
) -> Tensor:
    """
    Apply dilated attention pattern to attention scores.
    
    Args:
        scores: Attention scores [..., seq_len, seq_len]
        segment_length: Length of attention segment
        dilation_rate: Dilation rate for this group
        group_idx: Index of current group
        total_groups: Total number of groups
        
    Returns:
        Masked attention scores
    """
    seq_len = scores.shape[-1]
    device = scores.device
    
    # Create dilated pattern mask
    mask = create_dilated_mask(
        seq_len, segment_length, dilation_rate, device
    )
    
    # Apply mask
    scores = scores.masked_fill(~mask, float('-inf'))
    
    return scores


def create_dilated_mask(
    seq_len: int,
    segment_length: int,
    dilation_rate: int,
    device: torch.device
) -> Tensor:
    """
    Create a dilated attention mask.
    
    Args:
        seq_len: Sequence length
        segment_length: Length of attention segment
        dilation_rate: Dilation rate
        device: Device to create mask on
        
    Returns:
        Boolean mask [seq_len, seq_len]
    """
    # Initialize mask as all False
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
    # For each query position
    for i in range(seq_len):
        # Calculate the segment this position belongs to
        segment_idx = i // segment_length
        
        # Calculate the range of keys to attend to
        start = max(0, i - segment_length // 2 * dilation_rate)
        end = min(seq_len, i + segment_length // 2 * dilation_rate + 1)
        
        # Apply dilation
        if dilation_rate > 1:
            # Select positions with dilation
            positions = torch.arange(start, end, dilation_rate, device=device)
            positions = positions[positions < seq_len]
            mask[i, positions] = True
        else:
            # No dilation - continuous range
            mask[i, start:end] = True
    
    return mask


def create_block_diagonal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    overlap: int = 0
) -> Tensor:
    """
    Create a block-diagonal attention mask.
    
    Args:
        seq_len: Sequence length
        block_size: Size of attention blocks
        device: Device to create mask on
        overlap: Number of positions to overlap between blocks
        
    Returns:
        Boolean mask [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        
        # Add overlap with previous block
        if block_idx > 0 and overlap > 0:
            prev_start = max(0, start - overlap)
            mask[start:end, prev_start:start] = True
            mask[prev_start:start, start:end] = True
        
        # Main block
        mask[start:end, start:end] = True
        
        # Add overlap with next block
        if block_idx < num_blocks - 1 and overlap > 0:
            next_end = min(seq_len, end + overlap)
            mask[start:end, end:next_end] = True
            mask[end:next_end, start:end] = True
    
    return mask


def optimize_attention_computation(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0
) -> Tensor:
    """
    Optimize attention computation using available backends.
    
    Automatically selects the best available backend:
    1. Flash Attention (if available)
    2. PyTorch SDPA (if available)
    3. xFormers (if available)
    4. Standard PyTorch implementation
    
    Args:
        q, k, v: Query, key, value tensors [..., seq_len, num_heads, head_dim]
        is_causal: Whether to use causal masking
        attention_mask: Optional attention mask
        dropout_p: Dropout probability
        
    Returns:
        Attention output [..., seq_len, num_heads, head_dim]
    """
    # Try Flash Attention first
    if HAS_FLASH_ATTN and q.is_cuda:
        try:
            from flash_attn import flash_attn_func
            
            # Flash attention expects [batch, seq_len, num_heads, head_dim]
            # Reshape if needed
            original_shape = q.shape
            if q.dim() > 4:
                # Flatten batch dimensions
                q = q.reshape(-1, *q.shape[-3:])
                k = k.reshape(-1, *k.shape[-3:])
                v = v.reshape(-1, *v.shape[-3:])
            
            output = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                causal=is_causal,
                # Note: Flash Attention doesn't support arbitrary masks
            )
            
            # Reshape back
            if len(original_shape) > 4:
                output = output.reshape(*original_shape[:-3], *output.shape[-3:])
            
            return output
            
        except Exception as e:
            warnings.warn(f"Flash Attention failed, falling back: {e}")
    
    # Try PyTorch SDPA
    if HAS_SDPA:
        try:
            # SDPA expects [..., seq_len, head_dim] with num_heads as separate dim
            # Rearrange from [..., seq_len, num_heads, head_dim] 
            # to [..., num_heads, seq_len, head_dim]
            q_sdpa = q.transpose(-3, -2)
            k_sdpa = k.transpose(-3, -2)
            v_sdpa = v.transpose(-3, -2)
            
            output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attention_mask,
                dropout_p=dropout_p if q.training else 0.0,
                is_causal=is_causal
            )
            
            # Transpose back
            output = output.transpose(-3, -2)
            return output
            
        except Exception as e:
            warnings.warn(f"PyTorch SDPA failed, falling back: {e}")
    
    # Try xFormers
    if HAS_XFORMERS and q.is_cuda:
        try:
            import xformers.ops as xops
            
            # xFormers expects [batch * seq_len, num_heads, head_dim]
            batch_size = q.shape[0]
            seq_len = q.shape[-3]
            num_heads = q.shape[-2]
            head_dim = q.shape[-1]
            
            q_xf = q.reshape(batch_size * seq_len, num_heads, head_dim)
            k_xf = k.reshape(batch_size * seq_len, num_heads, head_dim)
            v_xf = v.reshape(batch_size * seq_len, num_heads, head_dim)
            
            output = xops.memory_efficient_attention(
                q_xf, k_xf, v_xf,
                attn_bias=attention_mask,
                p=dropout_p
            )
            
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
            return output
            
        except Exception as e:
            warnings.warn(f"xFormers failed, falling back: {e}")
    
    # Fallback to standard implementation
    return standard_attention(q, k, v, is_causal, attention_mask, dropout_p)


def standard_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0
) -> Tensor:
    """
    Standard scaled dot-product attention implementation.
    
    Args:
        q, k, v: Query, key, value tensors [..., seq_len, num_heads, head_dim]
        is_causal: Whether to use causal masking
        attention_mask: Optional attention mask
        dropout_p: Dropout probability
        
    Returns:
        Attention output [..., seq_len, num_heads, head_dim]
    """
    # Handle multi-head attention by reshaping
    # [..., seq_len, num_heads, head_dim] -> [..., num_heads, seq_len, head_dim]
    q = q.transpose(-3, -2)
    k = k.transpose(-3, -2)
    v = v.transpose(-3, -2)
    
    # Get dimensions
    *batch_dims, num_heads, seq_len, head_dim = q.shape
    
    # Compute attention scores for each head
    # We need to handle this head by head or reshape
    # Reshape to [...*num_heads, seq_len, head_dim]
    q_reshaped = q.reshape(-1, seq_len, head_dim)
    k_reshaped = k.reshape(-1, seq_len, head_dim)
    v_reshaped = v.reshape(-1, seq_len, head_dim)
    
    # Compute attention scores
    scores = compute_attention_scores(q_reshaped, k_reshaped, None, attention_mask, is_causal)
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    # Apply attention to values
    # attn_weights: [...*num_heads, seq_len, seq_len]
    # v_reshaped: [...*num_heads, seq_len, head_dim]
    output = torch.matmul(attn_weights, v_reshaped)
    
    # Reshape back to [..., num_heads, seq_len, head_dim]
    output = output.reshape(*batch_dims, num_heads, seq_len, head_dim)
    
    # Transpose back to [..., seq_len, num_heads, head_dim]
    output = output.transpose(-3, -2)
    
    return output


def compute_alibi_bias(
    seq_len: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Compute ALiBi (Attention with Linear Biases) positional bias.
    
    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        device: Device to create bias on
        dtype: Data type
        
    Returns:
        ALiBi bias tensor [num_heads, seq_len, seq_len]
    """
    # Compute slopes for each head
    slopes = torch.tensor(
        [2 ** (-8 * i / num_heads) for i in range(num_heads)],
        device=device,
        dtype=dtype
    )
    
    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
    
    # Compute bias
    alibi = slopes.unsqueeze(-1).unsqueeze(-1) * relative_positions.unsqueeze(0)
    
    return alibi


def compute_rotary_embeddings(
    seq_len: int,
    dim: int,
    device: torch.device,
    base: int = 10000,
    dtype: torch.dtype = torch.float32
) -> Tuple[Tensor, Tensor]:
    """
    Compute rotary position embeddings (RoPE).
    
    Args:
        seq_len: Sequence length
        dim: Embedding dimension (must be even)
        device: Device to create embeddings on
        base: Base for frequency computation
        dtype: Data type
        
    Returns:
        Tuple of (cos, sin) embeddings [seq_len, dim]
    """
    assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
    
    # Compute frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    
    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    
    # Compute angles
    angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
    
    # Create cos and sin embeddings
    cos_emb = torch.cos(angles).repeat(1, 2)
    sin_emb = torch.sin(angles).repeat(1, 2)
    
    return cos_emb, sin_emb


def apply_rotary_embeddings(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        q: Query tensor [..., seq_len, num_heads, head_dim]
        k: Key tensor [..., seq_len, num_heads, head_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Apply rotary embeddings
    # q/k shape: [..., seq_len, num_heads, head_dim]
    # cos/sin shape: [seq_len, head_dim]
    
    # Split head_dim into pairs
    q_pairs = q.view(*q.shape[:-1], -1, 2)  # [..., seq_len, num_heads, head_dim//2, 2]
    k_pairs = k.view(*k.shape[:-1], -1, 2)  # [..., seq_len, num_heads, head_dim//2, 2]
    
    # Reshape cos/sin to match
    cos_reshaped = cos.view(cos.shape[0], -1, 2)[..., 0]  # [seq_len, head_dim//2]
    sin_reshaped = sin.view(sin.shape[0], -1, 2)[..., 0]  # [seq_len, head_dim//2]
    
    # Expand cos/sin for broadcasting
    cos_expanded = cos_reshaped.unsqueeze(-2)  # [seq_len, 1, head_dim//2]
    sin_expanded = sin_reshaped.unsqueeze(-2)  # [seq_len, 1, head_dim//2]
    
    # Apply rotation formula
    q_rot_pairs = torch.stack([
        q_pairs[..., 0] * cos_expanded - q_pairs[..., 1] * sin_expanded,
        q_pairs[..., 0] * sin_expanded + q_pairs[..., 1] * cos_expanded
    ], dim=-1)
    
    k_rot_pairs = torch.stack([
        k_pairs[..., 0] * cos_expanded - k_pairs[..., 1] * sin_expanded,
        k_pairs[..., 0] * sin_expanded + k_pairs[..., 1] * cos_expanded
    ], dim=-1)
    
    # Reshape back to original shape
    q_rot = q_rot_pairs.view(*q.shape)
    k_rot = k_rot_pairs.view(*k.shape)
    
    return q_rot, k_rot


def merge_attention_heads(
    tensor: Tensor,
    num_heads: int,
    head_dim: int
) -> Tensor:
    """
    Merge attention heads back into hidden dimension.
    
    Args:
        tensor: Input tensor [..., seq_len, num_heads, head_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        
    Returns:
        Merged tensor [..., seq_len, hidden_dim]
    """
    # Merge heads
    return tensor.reshape(*tensor.shape[:-2], num_heads * head_dim)


def create_4d_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Create a 4D causal mask for attention.
    
    Args:
        seq_len: Sequence length
        device: Device to place mask on
        dtype: Data type for mask
        
    Returns:
        4D causal mask [1, 1, seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)


def apply_head_specific_masks(
    attention_scores: Tensor,
    head_masks: Optional[List[Tensor]] = None
) -> Tensor:
    """
    Apply head-specific masks to attention scores.
    
    Args:
        attention_scores: Attention scores [..., num_heads, seq_len, seq_len]
        head_masks: List of masks per head
        
    Returns:
        Masked attention scores
    """
    if head_masks is None:
        return attention_scores
        
    for h, mask in enumerate(head_masks):
        if mask is not None:
            attention_scores[..., h, :, :] = attention_scores[..., h, :, :].masked_fill(
                mask.bool(), float('-inf')
            )
    
    return attention_scores


def compute_position_embeddings(
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Compute sinusoidal position embeddings.
    
    Args:
        seq_len: Sequence length
        embed_dim: Embedding dimension
        device: Device to place embeddings on
        dtype: Data type
        
    Returns:
        Position embeddings [seq_len, embed_dim]
    """
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) *
        (-math.log(10000.0) / embed_dim)
    )
    
    pos_emb = torch.zeros(seq_len, embed_dim, device=device, dtype=dtype)
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    
    return pos_emb


def get_attention_backend_info() -> Dict[str, Any]:
    """
    Get information about available attention backends.
    
    Returns:
        Dictionary with backend availability info
    """
    info = {
        'has_flash_attention': HAS_FLASH_ATTN,
        'has_xformers': HAS_XFORMERS,
        'has_sdpa': HAS_SDPA,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name()
        info['gpu_capability'] = torch.cuda.get_device_capability()
        
    return info


def split_attention_heads(
    tensor: Tensor,
    num_heads: int
) -> Tensor:
    """
    Split hidden dimension into attention heads.
    
    Args:
        tensor: Input tensor [..., seq_len, hidden_dim]
        num_heads: Number of attention heads
        
    Returns:
        Split tensor [..., seq_len, num_heads, head_dim]
    """
    hidden_dim = tensor.shape[-1]
    head_dim = hidden_dim // num_heads
    
    return tensor.reshape(*tensor.shape[:-1], num_heads, head_dim)