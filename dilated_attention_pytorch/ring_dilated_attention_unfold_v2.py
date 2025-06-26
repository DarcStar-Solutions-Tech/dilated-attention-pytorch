"""
Ring Attention implementation using unfold and stride-based operations (v2).

This is a simplified and corrected version that properly handles the tensor dimensions.
"""

import threading
import warnings
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ring_dilated_attention import RingDilatedAttention


class UnfoldRingDilatedAttention(RingDilatedAttention):
    """
    Ring Attention with unfold-based dilated attention for O(n) memory complexity.
    
    This implementation overrides the dilated attention block to use unfold operations
    instead of index_select for better performance.
    """
    
    def _dilated_attention_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        ring_step: int = 0,
    ) -> Tensor:
        """
        Apply dilated attention using unfold operations.
        
        This method overrides the parent's implementation to use unfold instead of index_select.
        """
        b, n_q, h, d = q.shape
        b, n_kv, h, d = k.shape
        
        # Get pre-computed head distribution
        gs, head_ranges = self._get_head_groups(h)
        
        # Output accumulator
        out = torch.zeros_like(q)
        
        # Process each dilation group
        for i, ((g, (hmin, hmax)), r, s) in enumerate(
            zip(
                zip(gs, head_ranges, strict=False),
                self.dilation_rates,
                self.segment_lengths,
                strict=False,
            )
        ):
            # Skip if segments are larger than available sequence
            if n_q < s or n_kv < s:
                continue
            
            effective_s = min(s, n_kv)
            
            # Extract head groups - using contiguous for safety
            q_slice = q[:, :, hmin:hmax, :].contiguous()
            k_slice = k[:, :, hmin:hmax, :].contiguous()
            v_slice = v[:, :, hmin:hmax, :].contiguous()
            
            # Segment tensors
            q_segments = self._segment_tensor(q_slice, effective_s, n_q)
            k_segments = self._segment_tensor(k_slice, effective_s, n_kv)
            v_segments = self._segment_tensor(v_slice, effective_s, n_kv)
            
            # Apply dilation using unfold when possible
            if r > 1:
                offset = i % r
                
                if offset == 0:
                    # OPTIMIZATION: Use unfold for zero offset (most efficient)
                    # unfold creates a sliding window view: (batch, segments, windows, window_size, heads, dim)
                    # With step=r and size=1, we get every r-th element
                    q_dilated = q_segments.unfold(2, 1, r).squeeze(3)  # Remove window_size dim
                    k_dilated = k_segments.unfold(2, 1, r).squeeze(3)
                    v_dilated = v_segments.unfold(2, 1, r).squeeze(3)
                else:
                    # For non-zero offset, we need different approach
                    # First, create indices for the positions we want
                    indices = torch.arange(offset, effective_s, r, device=q.device)
                    
                    # Use gather for better performance than index_select
                    # Expand indices to match tensor dimensions
                    idx_expanded = indices.view(1, 1, -1, 1, 1).expand(
                        b, q_segments.size(1), -1, g, d
                    )
                    
                    # Use gather instead of indexing
                    q_dilated = torch.gather(q_segments, 2, idx_expanded)
                    k_dilated = torch.gather(k_segments, 2, idx_expanded)
                    v_dilated = torch.gather(v_segments, 2, idx_expanded)
            else:
                # No dilation needed
                q_dilated = q_segments
                k_dilated = k_segments
                v_dilated = v_segments
            
            # Get dimensions after dilation
            num_segments_q = q_dilated.size(1)
            num_segments_kv = k_dilated.size(1)
            seq_len = q_dilated.size(2)
            
            # Flatten for attention computation
            q_flat = q_dilated.contiguous().view(b * num_segments_q, seq_len, g, d)
            k_flat = k_dilated.contiguous().view(b * num_segments_kv, seq_len, g, d)
            v_flat = v_dilated.contiguous().view(b * num_segments_kv, seq_len, g, d)
            
            # Handle different segment counts between q and kv
            if num_segments_q != num_segments_kv:
                repeat_factor = (num_segments_q + num_segments_kv - 1) // num_segments_kv
                k_flat = k_flat.repeat(repeat_factor, 1, 1, 1)[: b * num_segments_q]
                v_flat = v_flat.repeat(repeat_factor, 1, 1, 1)[: b * num_segments_q]
            
            # Apply attention using parent's optimized method
            attn_out = self._apply_attention(
                q_flat, k_flat, v_flat, is_causal and ring_step == 0
            )
            
            # Reshape back
            attn_reshaped = attn_out.reshape(b, num_segments_q, seq_len, g, d)
            
            # Reconstruct full sequence from dilated results
            if r > 1:
                group_out = torch.zeros(
                    b, num_segments_q, effective_s, g, d,
                    device=attn_reshaped.device,
                    dtype=attn_reshaped.dtype,
                )
                
                if offset == 0:
                    # OPTIMIZATION: Use scatter for inverse of unfold
                    # Create indices for scatter
                    indices = torch.arange(0, effective_s, r, device=q.device)
                    idx_expanded = indices.view(1, 1, -1, 1, 1).expand(
                        b, num_segments_q, -1, g, d
                    )
                    group_out.scatter_(2, idx_expanded, attn_reshaped)
                else:
                    # Use scatter for non-zero offset too
                    indices = torch.arange(offset, effective_s, r, device=q.device)
                    idx_expanded = indices.view(1, 1, -1, 1, 1).expand(
                        b, num_segments_q, -1, g, d
                    )
                    group_out.scatter_(2, idx_expanded, attn_reshaped)
                
                attn_reshaped = group_out
            
            # Flatten and accumulate
            attn_flat = attn_reshaped.reshape(b, n_q, g, d)
            out[:, :, hmin:hmax, :].add_(attn_flat)
        
        # Normalize by number of groups
        out.div_(self.num_groups)
        
        # Apply dropout if configured
        out = self._apply_dropout(out)
        
        return out
    
    def _apply_attention(self, q_flat, k_flat, v_flat, is_causal):
        """Helper to apply scaled dot product attention."""
        # Try to use the parent's SDPA logic if available
        if hasattr(super(), '_apply_scaled_dot_product_attention'):
            return super()._apply_scaled_dot_product_attention(
                q_flat, k_flat, v_flat, is_causal
            )
        
        # Otherwise use standard SDPA
        return F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=None,
        )