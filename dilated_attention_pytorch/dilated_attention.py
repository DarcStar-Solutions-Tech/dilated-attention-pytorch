from typing import Optional, Sequence

import torch
import xformers.ops as xops
from einops import rearrange
from torch import Tensor, nn


class DilatedAttention(nn.Module):
    """Implement dilated, scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        op: Optional[xops.AttentionOp] = None,
    ):
        super().__init__()
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError(
                "segment_lengths and dilation_rates must have the same length"
            )
        
        # Validate segment lengths and dilation rates
        if not segment_lengths:
            raise ValueError("segment_lengths cannot be empty")
        
        for i, (seg_len, dil_rate) in enumerate(zip(segment_lengths, dilation_rates)):
            if seg_len <= 0:
                raise ValueError(f"segment_lengths[{i}] must be positive, got {seg_len}")
            if dil_rate <= 0:
                raise ValueError(f"dilation_rates[{i}] must be positive, got {dil_rate}")
        
        # Validate dropout
        if not 0.0 <= attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be between 0 and 1, got {attention_dropout}")

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.op = op

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> Tensor:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #   s - segment length
        #   r - dilation rate
        #   g - group size (i.e. number of heads per segment length)
        #
        # Input shape of query, key, value: (b, n, h, d)
        # Validate input shapes
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError(
                f"Expected 4D tensors (batch, seq_len, heads, dim), got shapes: "
                f"query={query.shape}, key={key.shape}, value={value.shape}"
            )
        
        b, n, h, d = query.shape
        if key.shape != (b, n, h, d) or value.shape != (b, n, h, d):
            raise ValueError(
                f"query, key, and value must have the same shape, got: "
                f"query={query.shape}, key={key.shape}, value={value.shape}"
            )
        
        # Validate sequence length is compatible with largest segment length
        max_segment = max(self.segment_lengths)
        if n % max_segment != 0:
            raise ValueError(
                f"Sequence length ({n}) must be divisible by the largest segment length ({max_segment})"
            )
        
        out = torch.zeros_like(query)

        # *** NOTE ***
        # The original paper does not describe how to handle the case where
        #   h % len(self.segment_lengths) != 0
        #
        # In my first implementation, I naively assumed (and asserted) that
        # 'h % len(self.segment_lengths) == 0', so that I could evenly distribute
        # the heads between the different segment lengths. However, it was not
        # possible to reproduce the LongNet hyperparameters with that restriction:
        #   h=12, segment_lengths=[2048, 4096, 8192, 16384, 32768]
        #   h % len(segment_lengths) == 2
        #
        # For that reason, I have removed the assertion, and instead grouped the heads
        # into (potentially) unequally sized groups.  If not perfectly divisible, then
        # the first few groups will have an extraattention head.
        num_groups = len(self.dilation_rates)
        group_sizes = [h // num_groups] * num_groups
        for i in range(h % num_groups):
            group_sizes[i] += 1

        # Calculate correct head ranges for unequal group sizes
        head_ranges = []
        cumsum = 0
        for g in group_sizes:
            head_ranges.append((cumsum, cumsum + g))
            cumsum += g

        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths)
        ):
            # Split the input sequences into segments of length 'self.segment_length'
            q = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v = rearrange(value, "b (n s) h d -> b n s h d", s=s)
            # Apply dilation and segment offset
            offset = i % r
            hmin, hmax = head_ranges[i]
            q = q[:, :, offset::r, hmin:hmax, :]
            k = k[:, :, offset::r, hmin:hmax, :]
            v = v[:, :, offset::r, hmin:hmax, :]
            # Fold all 'n' segments into the batch dimension
            q = rearrange(q, "b n s h d -> (b n) s h d")
            k = rearrange(k, "b n s h d -> (b n) s h d")
            v = rearrange(v, "b n s h d -> (b n) s h d")

            # Apply memory efficient attention
            # NOTE: If flash attention is correctly installed, then this will also
            # automatically use the flash attention implementation.
            attn_bias = xops.LowerTriangularMask() if is_causal else None
            x = xops.memory_efficient_attention(
                query=q, key=k, value=v, op=self.op, attn_bias=attn_bias
            )
            # Unfold 'n' segments back out of the batch dimension.
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)

            # Gather the attention outputs from each dilation rate / segment length.
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, offset::r, hmin:hmax, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d", s=s)

        # Normalize across all attention outputs by dividing by the number of
        # attention groups.  See: https://arxiv.org/pdf/2307.02486.pdf, Eq. 10
        return out / num_groups


