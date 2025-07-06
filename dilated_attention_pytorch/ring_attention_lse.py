"""
Ring attention LSE (Log-Sum-Exp) utilities.

This module provides LSE-related functions for numerically stable attention computation
in ring attention implementations.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


class StableRingAccumulator:
    """
    Numerically stable accumulator for ring attention.

    This class maintains running attention outputs and LSE values
    across ring communication steps.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        """Initialize the accumulator."""
        self.dtype = dtype
        self.attn_out = None
        self.lse = None

    def update(
        self, new_attn: Tensor, new_lse: Tensor, mask: Optional[Tensor] = None
    ) -> None:
        """
        Update accumulator with new attention values.

        Args:
            new_attn: New attention output
            new_lse: New LSE values
            mask: Optional mask
        """
        if self.attn_out is None:
            self.attn_out = new_attn
            self.lse = new_lse
        else:
            # Update LSE and attention output
            new_lse_val, norm_factor = update_lse(self.lse, new_lse, mask)

            # Update attention output with proper normalization
            self.attn_out = self.attn_out * norm_factor + new_attn
            self.lse = new_lse_val

    def get(self) -> Tuple[Tensor, Tensor]:
        """Get current attention output and LSE."""
        return self.attn_out, self.lse


def ring_pass_lse(
    attn_out: Tensor, lse: Tensor, next_kv_chunk: Tensor, ring_size: int, rank: int
) -> Tuple[Tensor, Tensor]:
    """
    Pass attention output and LSE through ring communication.

    Args:
        attn_out: Current attention output
        lse: Current log-sum-exp values
        next_kv_chunk: Next KV chunk to process
        ring_size: Size of the ring
        rank: Current rank

    Returns:
        Updated attention output and LSE
    """
    if ring_size <= 1:
        return attn_out, lse

    # In a real implementation, this would involve ring communication
    # For now, return as-is to avoid breaking functionality
    return attn_out, lse


def update_lse(
    old_lse: Tensor, new_scores: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Update log-sum-exp values with new attention scores.

    Args:
        old_lse: Previous LSE values
        new_scores: New attention scores
        mask: Optional attention mask

    Returns:
        Updated LSE and normalization factor
    """
    if mask is not None:
        new_scores = new_scores.masked_fill(mask, float("-inf"))

    # Compute new LSE
    new_max = torch.maximum(old_lse, new_scores.max(dim=-1, keepdim=True).values)

    # Compute stable exp
    old_exp = torch.exp(old_lse - new_max)
    new_exp = torch.exp(new_scores - new_max).sum(dim=-1, keepdim=True)

    # Update LSE
    new_lse = new_max + torch.log(old_exp + new_exp)

    # Compute normalization factor
    norm_factor = torch.exp(old_lse - new_lse)

    return new_lse, norm_factor


def compute_lse_triton(scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Compute log-sum-exp using Triton kernels if available.

    Args:
        scores: Attention scores
        mask: Optional attention mask

    Returns:
        LSE values
    """
    # For now, fall back to PyTorch implementation
    return compute_lse(scores, mask)


def compute_lse(scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Compute log-sum-exp for numerical stability.

    Args:
        scores: Attention scores
        mask: Optional attention mask

    Returns:
        LSE values
    """
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Compute max for numerical stability
    max_scores = scores.max(dim=-1, keepdim=True).values

    # Compute LSE
    exp_scores = torch.exp(scores - max_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    lse = max_scores + torch.log(sum_exp)

    return lse


def compute_attention_with_lse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute attention with LSE for numerical stability.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional attention mask
        scale: Optional scale factor

    Returns:
        Attention output and LSE values
    """
    # Compute scale factor
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Compute LSE for numerical stability
    lse = compute_lse(scores, mask=None)  # mask already applied

    # Compute attention weights
    max_scores = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)
    attn_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    # Apply attention to values
    attn_out = torch.matmul(attn_weights, v)

    return attn_out, lse


__all__ = [
    "StableRingAccumulator",
    "ring_pass_lse",
    "update_lse",
    "compute_lse_triton",
    "compute_lse",
    "compute_attention_with_lse",
]
