"""
Ring Dilated Attention V3 - Complete implementation with proper autograd.

This version includes:
1. Custom autograd for proper gradient handling
2. Hilbert SFC support in both forward and backward
3. Fixed ring communication
4. Support for dilated attention patterns
"""

import logging
import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from .ring_attention_autograd import ring_attention
from .utils.gpu_utils import get_gpu_info
from .utils.hilbert_attention_mixin import HilbertAttentionMixin

logger = logging.getLogger(__name__)


class RingDilatedAttentionV3(nn.Module, HilbertAttentionMixin):
    """
    Ring Dilated Attention with proper distributed support.

    Features:
    - Custom autograd for correct gradient computation
    - Hilbert SFC optimization for both forward and backward
    - Multiple segment lengths with different dilation rates
    - Memory efficient O(n/k) scaling
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        bias: bool = True,
        use_hilbert: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Core parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_hilbert = use_hilbert

        # Validate
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        )
        assert len(segment_lengths) == len(dilation_rates), (
            "segment_lengths and dilation_rates must have same length"
        )

        # Device and dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32  # Use FP32 for stability
        self.device = device
        self.dtype = dtype

        # Get GPU info
        self.gpu_info = get_gpu_info(device)
        logger.info(f"Using GPU: {self.gpu_info.name}")

        # Linear projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Move to device
        self.qkv_proj = self.qkv_proj.to(device=device, dtype=dtype)
        self.out_proj = self.out_proj.to(device=device, dtype=dtype)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Scaling
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Distributed info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Hilbert setup
        if use_hilbert:
            self.setup_hilbert_attention(
                hidden_dim=embed_dim,
                num_heads=num_heads,
                use_hilbert_core=False,
            )
            # Pre-compute Hilbert indices for each segment length
            self._precompute_hilbert_indices()

        logger.info(
            f"Initialized RingDilatedAttentionV3 "
            f"(rank={self.rank}/{self.world_size}, "
            f"segments={segment_lengths}, dilations={dilation_rates})"
        )

    def _precompute_hilbert_indices(self):
        """Pre-compute Hilbert indices for each segment length."""
        self.hilbert_indices_cache = {}

        for seg_idx, seg_len in enumerate(self.segment_lengths):
            if hasattr(self, "hilbert_curve_ids") and seg_idx < len(
                self.hilbert_curve_ids
            ):
                # Use existing Hilbert curve IDs from mixin
                indices = self.hilbert_curve_ids[seg_idx]
            else:
                # Generate simple indices if not available
                indices = torch.arange(seg_len, device=self.device, dtype=torch.long)

            self.hilbert_indices_cache[seg_len] = indices

    def forward(
        self,
        x: Tensor,
        total_seq_len: Optional[int] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        already_split: bool = False,
    ) -> Tensor:
        """
        Forward pass of ring dilated attention.

        Args:
            x: Input tensor [batch, seq, embed_dim]
            total_seq_len: Total sequence length across all ranks
            is_causal: Whether to use causal masking
            need_weights: Whether to return attention weights (not supported)
            already_split: If True, x is already the local chunk

        Returns:
            Output tensor [batch, local_seq, embed_dim]
        """
        batch_size = x.shape[0]

        # Handle splitting for distributed
        if self.world_size > 1 and not already_split:
            seq_len = x.shape[1]
            assert seq_len % self.world_size == 0, (
                f"seq_len {seq_len} must be divisible by world_size {self.world_size}"
            )

            local_seq_len = seq_len // self.world_size
            start = self.rank * local_seq_len
            end = start + local_seq_len

            x_local = x[:, start:end, :].contiguous()
            total_seq_len = seq_len
        else:
            x_local = x.contiguous()
            local_seq_len = x.shape[1]
            if total_seq_len is None:
                total_seq_len = local_seq_len * self.world_size

        # QKV projection
        qkv = self.qkv_proj(x_local)
        qkv = qkv.reshape(batch_size, local_seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]

        # Process each segment with its dilation rate
        outputs = []

        for seg_idx, (seg_len, dilation) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            # Skip if segment is larger than local sequence
            if seg_len > local_seq_len:
                continue

            # Apply dilation pattern if needed
            if dilation > 1:
                # Simple dilation: take every dilation-th position
                indices = torch.arange(0, local_seq_len, dilation, device=self.device)
                indices = indices[:seg_len]  # Limit to segment length

                q_seg = q_local[:, :, indices, :]
                k_seg = k_local[:, :, indices, :]
                v_seg = v_local[:, :, indices, :]
            else:
                # No dilation, process full segment
                q_seg = q_local[:, :, :seg_len, :]
                k_seg = k_local[:, :, :seg_len, :]
                v_seg = v_local[:, :, :seg_len, :]

            # Get Hilbert indices for this segment
            hilbert_indices = None
            if self.use_hilbert and seg_len in self.hilbert_indices_cache:
                hilbert_indices = self.hilbert_indices_cache[seg_len]
                # Ensure indices match segment size
                if hilbert_indices.shape[0] > q_seg.shape[2]:
                    hilbert_indices = hilbert_indices[: q_seg.shape[2]]
                elif hilbert_indices.shape[0] < q_seg.shape[2]:
                    # Pad with sequential indices
                    padding = torch.arange(
                        hilbert_indices.shape[0], q_seg.shape[2], device=self.device
                    )
                    hilbert_indices = torch.cat([hilbert_indices, padding])

            # Apply ring attention
            if self.world_size > 1:
                seg_output = ring_attention(
                    q_seg,
                    k_seg,
                    v_seg,
                    scale=self.scale,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    use_hilbert=self.use_hilbert and hilbert_indices is not None,
                    hilbert_indices=hilbert_indices,
                )
            else:
                # Single GPU - use standard attention
                seg_output = self._local_attention(q_seg, k_seg, v_seg, is_causal)

            # Restore original positions if dilated
            if dilation > 1:
                full_output = torch.zeros(
                    batch_size,
                    self.num_heads,
                    local_seq_len,
                    self.head_dim,
                    device=self.device,
                    dtype=seg_output.dtype,
                )
                full_output[:, :, indices, :] = seg_output
                outputs.append(full_output)
            else:
                # Pad to full sequence length
                if seg_output.shape[2] < local_seq_len:
                    padding = torch.zeros(
                        batch_size,
                        self.num_heads,
                        local_seq_len - seg_output.shape[2],
                        self.head_dim,
                        device=self.device,
                        dtype=seg_output.dtype,
                    )
                    seg_output = torch.cat([seg_output, padding], dim=2)
                outputs.append(seg_output)

        # Combine outputs from all segments
        if len(outputs) == 0:
            # No valid segments, return zeros
            output = torch.zeros(
                batch_size,
                self.num_heads,
                local_seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
        elif len(outputs) == 1:
            output = outputs[0]
        else:
            # Average outputs from different segments
            output = torch.stack(outputs).mean(dim=0)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, local_seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout_layer(output)

        if need_weights:
            return output, None  # Ring attention doesn't return weights
        return output

    def _local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Standard attention for single GPU."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=q.device),
                diagonal=1,
            )
            scores = scores + causal_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"dropout={self.dropout}, use_hilbert={self.use_hilbert}, "
            f"world_size={self.world_size}"
        )
