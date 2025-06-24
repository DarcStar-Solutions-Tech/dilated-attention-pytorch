from typing import Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

# Handle torch.nn.attention availability
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_SDPA_KERNEL = True
except ImportError:
    HAS_SDPA_KERNEL = False
    # Fallback for older PyTorch versions
    class SDPBackend:
        FLASH_ATTENTION = "flash_attention"
        EFFICIENT_ATTENTION = "efficient_attention"
        MATH = "math"

class ImprovedDilatedAttention(nn.Module):
    def __init__(self, segment_lengths, dilation_rates,
                 dropout=0.0, use_tf32=True):
        super().__init__()
        assert len(segment_lengths) == len(dilation_rates)
        self.seg = segment_lengths
        self.dil = dilation_rates
        self.drop = dropout
        self.num_groups = len(self.seg)
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            
        # Pre-compute head distribution to avoid runtime calculation
        self._head_groups = None
        self._cached_indices = {}

    def _get_head_groups(self, h):
        """Pre-compute and cache head group distribution."""
        if self._head_groups is None or self._head_groups[0] != h:
            gs = [h // self.num_groups] * self.num_groups
            for i in range(h % self.num_groups): gs[i] += 1
            
            # Pre-compute head ranges for faster indexing
            head_ranges = []
            hmin = 0
            for g in gs:
                hmax = hmin + g
                head_ranges.append((hmin, hmax))
                hmin = hmax
            
            self._head_groups = (h, gs, head_ranges)
        
        return self._head_groups[1], self._head_groups[2]

    def forward(self, q, k, v, is_causal=False):
        b, n, h, d = q.shape
        device, dtype = q.device, q.dtype
        
        # Pre-allocate output with optimal memory pattern
        out = torch.empty_like(q)
        out.zero_()

        # Get pre-computed head distribution
        gs, head_ranges = self._get_head_groups(h)

        # Process all segments with optimized memory access patterns
        for i, ((g, (hmin, hmax)), r, s) in enumerate(zip(zip(gs, head_ranges), self.dil, self.seg)):
            if n < s: continue  # skip large segments if too small

            offset = i % r
            
            # Use direct tensor views instead of rearrange for memory efficiency
            # Shape: [b, n, h, d] -> [b, n//s, s, h, d]
            q_slice = q[:, :, hmin:hmax, :].view(b, n//s, s, g, d)
            k_slice = k[:, :, hmin:hmax, :].view(b, n//s, s, g, d)
            v_slice = v[:, :, hmin:hmax, :].view(b, n//s, s, g, d)

            # Apply dilation with optimized indexing
            if r > 1 or offset:
                # Cache indices for reuse
                cache_key = (s, r, offset)
                if cache_key not in self._cached_indices:
                    self._cached_indices[cache_key] = torch.arange(offset, s, r, device=device)
                idx = self._cached_indices[cache_key]
                
                # Use advanced indexing for dilated sampling
                q_slice = q_slice[:, :, idx, :, :]
                k_slice = k_slice[:, :, idx, :, :]
                v_slice = v_slice[:, :, idx, :, :]

            # Reshape for attention: [b, n//s, dilated_s, g, d] -> [b*n//s, dilated_s, g, d]
            bn = b * (n // s)
            dilated_s = q_slice.size(2)
            q_flat = q_slice.view(bn, dilated_s, g, d)
            k_flat = k_slice.view(bn, dilated_s, g, d)
            v_flat = v_slice.view(bn, dilated_s, g, d)

            # Optimized attention computation
            if HAS_SDPA_KERNEL:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    x = F.scaled_dot_product_attention(
                        q_flat, k_flat, v_flat,
                        attn_mask=None,
                        dropout_p=self.drop if self.training else 0.0,
                        is_causal=is_causal,
                        scale=None,
                        enable_gqa=False
                    )
            else:
                # Fallback for older PyTorch versions
                x = F.scaled_dot_product_attention(
                    q_flat, k_flat, v_flat,
                    attn_mask=None,
                    dropout_p=self.drop if self.training else 0.0,
                    is_causal=is_causal,
                    scale=None,
                )

            # Reshape back and add to output in-place
            # [b*n//s, dilated_s, g, d] -> [b, n//s, dilated_s, g, d] -> [b, n, g, d]
            x_reshaped = x.view(b, n//s, dilated_s, g, d)
            
            # Scatter back to original positions
            if r > 1 or offset:
                # Create temporary tensor for scattering
                temp_output = torch.zeros(b, n//s, s, g, d, device=device, dtype=dtype)
                temp_output[:, :, idx, :, :] = x_reshaped
                out[:, :, hmin:hmax, :].add_(temp_output.view(b, n, g, d))
            else:
                out[:, :, hmin:hmax, :].add_(x_reshaped.view(b, n, g, d))

        # Use in-place division to avoid creating a new tensor
        out.div_(self.num_groups)
        return out

# Optionally compile for further fusion:
# Note: Disabled by default due to compatibility issues. 
# To enable: ImprovedDilatedAttention = torch.compile(ImprovedDilatedAttention, fullgraph=True)

