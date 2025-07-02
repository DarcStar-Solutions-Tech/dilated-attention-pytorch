"""
Optimized Log-Sum-Exp utilities for Ring Attention with backend fallbacks.

This module extends the basic LSE utilities to use optimized attention
backends (Flash Attention, SDPA, xFormers) while maintaining LSE tracking.
"""

import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor

from .utils.attention_utils import optimize_attention_computation
from .core.constants import HAS_FLASH_ATTN, HAS_SDPA, HAS_XFORMERS


def compute_attention_with_lse_optimized(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    training: bool = False,
    is_causal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Compute attention with LSE tracking using optimized backends.

    This function tries to use optimized attention implementations while
    still computing LSE values for ring accumulation stability.

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        scale: Scaling factor (usually 1/sqrt(dim))
        mask: Optional attention mask
        dropout: Dropout probability
        training: Whether in training mode
        is_causal: Whether to use causal masking

    Returns:
        (output, lse) tuple where:
        - output: Attention output (batch, heads, seq_q, dim)
        - lse: Log-sum-exp values (batch, heads, seq_q)
    """
    # First, compute scores for LSE
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    # Apply causal mask if needed
    if is_causal and mask is None:
        seq_len_q, seq_len_k = q.shape[2], k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Compute LSE for accumulation
    lse = scores.logsumexp(dim=-1)

    # Now compute attention output using optimized backend
    # Reshape for optimize_attention_computation which expects
    # [..., seq_len, num_heads, head_dim]
    b, h, sq, d = q.shape
    _, _, sk, _ = k.shape

    # Transpose to expected format
    q_opt = q.transpose(1, 2)  # (b, seq_q, heads, dim)
    k_opt = k.transpose(1, 2)  # (b, seq_k, heads, dim)
    v_opt = v.transpose(1, 2)  # (b, seq_k, heads, dim)

    # Create attention mask in the right format if needed
    attn_mask = None
    if mask is not None and not is_causal:
        # Expand mask to match expected shape
        if mask.dim() == 3:  # (heads, seq_q, seq_k)
            attn_mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
        elif mask.dim() == 4:  # (batch, heads, seq_q, seq_k)
            attn_mask = mask
        # Convert boolean mask to additive mask
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.where(
                attn_mask,
                torch.zeros_like(attn_mask, dtype=q.dtype),
                torch.full_like(attn_mask, float("-inf"), dtype=q.dtype),
            )

    try:
        # Use optimized computation
        output = optimize_attention_computation(
            q_opt,
            k_opt,
            v_opt,
            is_causal=is_causal,
            attention_mask=attn_mask,
            dropout_p=dropout if training else 0.0,
        )

        # Transpose back to (batch, heads, seq, dim)
        output = output.transpose(1, 2)

    except Exception as e:
        warnings.warn(
            f"Optimized attention failed, falling back to standard: {e}", stacklevel=2
        )

        # Fallback to standard computation
        # Compute attention weights from scores
        if mask is not None and mask.dtype == torch.bool:
            scores_safe = scores.masked_fill(~mask, float("-inf"))
        else:
            scores_safe = scores

        attn_weights = torch.softmax(scores_safe, dim=-1)

        # Apply dropout if needed
        if training and dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout)

        # Compute output
        output = torch.matmul(attn_weights, v)

    return output, lse


def get_attention_backend_info() -> dict:
    """Get information about available attention backends."""
    return {
        "flash_attn": HAS_FLASH_ATTN,
        "sdpa": HAS_SDPA,
        "xformers": HAS_XFORMERS,
        "backends_available": sum([HAS_FLASH_ATTN, HAS_SDPA, HAS_XFORMERS]),
    }
