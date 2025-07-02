"""
Log-Sum-Exp utilities for numerically stable Ring Attention.

Based on patterns from lucidrains/ring-attention-pytorch, these utilities
ensure numerical stability when accumulating attention across ring passes.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def logsumexp_accum(
    output: Tensor,
    lse: Tensor,
    new_output: Tensor,
    new_lse: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Numerically stable accumulation using log-sum-exp trick.

    This is essential for ring attention to avoid numerical instability
    when combining outputs from different chunks.

    Args:
        output: Current accumulated output
        lse: Current log-sum-exp values
        new_output: New output to accumulate
        new_lse: New log-sum-exp values

    Returns:
        Updated (output, lse) tuple
    """
    # Handle -inf LSE values gracefully
    # Replace -inf with a very negative number to avoid NaN
    lse_safe = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, -1e10))
    new_lse_safe = torch.where(
        torch.isfinite(new_lse), new_lse, torch.full_like(new_lse, -1e10)
    )

    # Find the maximum LSE for numerical stability
    max_lse = torch.maximum(lse_safe, new_lse_safe)

    # Compute stable exponentials
    stable_lse1 = (lse_safe - max_lse).exp()
    stable_lse2 = (new_lse_safe - max_lse).exp()

    # Update output with proper weighting
    output = output * stable_lse1.unsqueeze(-1) + new_output * stable_lse2.unsqueeze(-1)

    # Update LSE
    lse_sum = stable_lse1 + stable_lse2
    # Avoid log(0) by adding small epsilon
    lse = max_lse + (lse_sum + 1e-10).log()

    # Normalize output
    output = output / (lse_sum.unsqueeze(-1) + 1e-10)

    # Restore -inf where both inputs were -inf
    both_inf = ~torch.isfinite(lse) & ~torch.isfinite(new_lse)
    if both_inf.any():
        lse = torch.where(both_inf, torch.full_like(lse, float("-inf")), lse)
        output = torch.where(both_inf.unsqueeze(-1), torch.zeros_like(output), output)

    return output, lse


def compute_attention_with_lse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    training: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Compute attention with log-sum-exp tracking for numerical stability.

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        scale: Scaling factor (usually 1/sqrt(dim))
        mask: Optional attention mask
        dropout: Dropout probability
        training: Whether in training mode

    Returns:
        (output, lse) tuple where:
        - output: Attention output (batch, heads, seq_q, dim)
        - lse: Log-sum-exp values (batch, heads, seq_q)
    """
    # Compute scores
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Compute log-sum-exp for numerical stability
    max_scores = scores.amax(dim=-1, keepdim=True)
    stable_scores = scores - max_scores
    exp_scores = stable_scores.exp()

    # Sum of exponentials
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)

    # Log-sum-exp
    lse = max_scores.squeeze(-1) + sum_exp.log().squeeze(-1)

    # Compute attention weights
    attn_weights = exp_scores / sum_exp

    # Apply dropout if needed
    if training and dropout > 0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout)

    # Compute output
    output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    return output, lse


def softclamp(tensor: Tensor, value: float) -> Tensor:
    """
    Soft clamping function to prevent extreme values.

    Used in the original ring-attention-pytorch to prevent
    attention scores from becoming too large.

    Args:
        tensor: Input tensor
        value: Clamping threshold

    Returns:
        Soft-clamped tensor
    """
    return (tensor / value).tanh() * value


class StableRingAccumulator:
    """
    Accumulator for ring attention using log-sum-exp for stability.

    This class manages the accumulation of attention outputs across
    ring passes while maintaining numerical stability.
    """

    def __init__(
        self, output_shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype
    ):
        """
        Initialize the accumulator.

        Args:
            output_shape: Shape of the output tensor (batch, heads, seq, dim)
            device: Device to create tensors on
            dtype: Data type for tensors
        """
        self.output = torch.zeros(output_shape, device=device, dtype=dtype)
        self.lse = torch.full(
            output_shape[:-1], float("-inf"), device=device, dtype=dtype
        )
        self.initialized = False

    def update(self, new_output: Tensor, new_lse: Tensor):
        """
        Update accumulator with new chunk results.

        Args:
            new_output: Output from current chunk
            new_lse: Log-sum-exp from current chunk
        """
        if not self.initialized:
            # First update - just copy
            self.output = new_output
            self.lse = new_lse
            self.initialized = True
        else:
            # Subsequent updates - use stable accumulation
            self.output, self.lse = logsumexp_accum(
                self.output, self.lse, new_output, new_lse
            )

    def get_output(self) -> Tensor:
        """Get the final normalized output."""
        return self.output


def create_ring_flash_attn_func(
    use_flash_attn: bool = True,
    use_triton: bool = False,
) -> callable:
    """
    Create an attention function that can use Flash Attention if available.

    Args:
        use_flash_attn: Whether to attempt using Flash Attention
        use_triton: Whether to use Triton kernels (requires triton)

    Returns:
        Attention function that returns (output, lse)
    """
    if use_flash_attn:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func

            def flash_attn_with_lse(q, k, v, causal=False, window_size=(-1, -1)):
                """Flash attention wrapper that returns LSE."""
                # Flash attention expects (batch, seq, heads, dim)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                # Call flash attention
                output, softmax_lse = flash_attn_func(
                    q,
                    k,
                    v,
                    causal=causal,
                    window_size=window_size,
                    return_attn_probs=False,
                    return_softmax_lse=True,
                )

                # Transpose back to (batch, heads, seq, dim)
                output = output.transpose(1, 2)

                return output, softmax_lse

            return flash_attn_with_lse

        except ImportError:
            pass

    # Fallback to standard attention with LSE
    return compute_attention_with_lse
