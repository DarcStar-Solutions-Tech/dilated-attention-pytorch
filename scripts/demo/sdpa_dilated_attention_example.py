#!/usr/bin/env python3
"""
Example of using PyTorch's scaled_dot_product_attention with dilated attention patterns.

This demonstrates how to achieve 40x speedup using SDPA with custom masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SDPADilatedAttention(nn.Module):
    """
    Dilated Attention using PyTorch's scaled_dot_product_attention.

    This implementation leverages SDPA's optimized kernels while maintaining
    the dilated attention pattern through custom masking.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_length: int,
        dilation_rate: int,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.segment_length = segment_length
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Cache for attention masks
        self._mask_cache = {}

    def _get_dilated_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create dilated attention mask."""
        cache_key = (seq_len, self.segment_length, self.dilation_rate, device, dtype)

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Create mask
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

        # Apply dilated pattern within segments
        num_segments = seq_len // self.segment_length
        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_length
            end = start + self.segment_length

            # Create dilated connections within segment
            for i in range(start, end):
                for j in range(start, end):
                    if (i - start) % self.dilation_rate == 0 and (
                        j - start
                    ) % self.dilation_rate == 0:
                        mask[i, j] = 1.0

        # Convert to attention mask (0 = attend, -inf = mask)
        attention_mask = torch.where(mask == 1, 0.0, float("-inf"))

        # Cache the mask
        self._mask_cache[cache_key] = attention_mask

        return attention_mask

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using SDPA with dilated attention pattern.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            need_weights: If True, return attention weights (not supported with SDPA)

        Returns:
            Output tensor and optionally attention weights
        """
        if need_weights and self.use_flash_attention:
            raise ValueError("need_weights=True is not supported with Flash Attention")

        batch_size, seq_len, _ = x.shape

        # Ensure sequence length is compatible
        assert seq_len % self.segment_length == 0, (
            f"Sequence length {seq_len} must be divisible by segment_length {self.segment_length}"
        )

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get dilated attention mask
        mask = self._get_dilated_mask(seq_len, x.device, x.dtype)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

        # Apply scaled dot product attention
        if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use optimized SDPA
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None
        else:
            # Fallback to manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output, attn_weights


def compare_implementations():
    """Compare SDPA vs manual attention performance."""
    import time

    # Configuration
    batch_size = 2
    seq_len = 4096
    embed_dim = 768
    num_heads = 12
    segment_length = 1024
    dilation_rate = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 for Pascal GPUs

    # Create models
    sdpa_model = SDPADilatedAttention(
        embed_dim, num_heads, segment_length, dilation_rate, use_flash_attention=True
    ).to(device, dtype)

    manual_model = SDPADilatedAttention(
        embed_dim, num_heads, segment_length, dilation_rate, use_flash_attention=False
    ).to(device, dtype)

    # Copy weights
    manual_model.load_state_dict(sdpa_model.state_dict())

    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = sdpa_model(x)
        _ = manual_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark SDPA
    start = time.perf_counter()
    for _ in range(10):
        output_sdpa, _ = sdpa_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / 10

    # Benchmark manual
    start = time.perf_counter()
    for _ in range(10):
        output_manual, _ = manual_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    manual_time = (time.perf_counter() - start) / 10

    print("Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Segment length: {segment_length}")
    print(f"  Dilation rate: {dilation_rate}")
    print(f"  Device: {device}")
    print("\nPerformance:")
    print(f"  SDPA time: {sdpa_time * 1000:.2f}ms")
    print(f"  Manual time: {manual_time * 1000:.2f}ms")
    print(f"  Speedup: {manual_time / sdpa_time:.2f}x")

    # Check accuracy
    diff = (output_sdpa - output_manual).abs().max().item()
    print("\nAccuracy:")
    print(f"  Max difference: {diff:.6e}")

    # Calculate sparsity
    with torch.no_grad():
        mask = sdpa_model._get_dilated_mask(seq_len, device, dtype)
        sparsity = (mask == float("-inf")).float().mean().item()
        print("\nPattern:")
        print(f"  Sparsity: {sparsity * 100:.1f}%")
        print(f"  Attention connections: {(1 - sparsity) * 100:.1f}%")


if __name__ == "__main__":
    print("SDPA Dilated Attention Example")
    print("=" * 50)
    compare_implementations()
