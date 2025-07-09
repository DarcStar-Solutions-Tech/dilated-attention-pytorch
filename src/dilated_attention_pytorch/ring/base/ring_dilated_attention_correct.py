"""
Correct Ring Dilated Attention Implementation

This implementation actually achieves O(n/k) memory per GPU by:
1. Splitting inputs BEFORE QKV projection
2. Each GPU only processes its local chunk
3. Ring communication passes small K,V chunks only
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List, Tuple, Union
import logging

from ...utils.hilbert_attention_mixin import HilbertAttentionMixin
from ...utils.gpu_utils import get_gpu_info, select_attention_backend

logger = logging.getLogger(__name__)


def ring_pass_forward(tensor: torch.Tensor) -> torch.Tensor:
    """Pass tensor to next GPU in ring using efficient communication."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Determine source and destination
    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    # Non-blocking communication
    recv_buffer = torch.empty_like(tensor)
    send_op = dist.isend(tensor.contiguous(), dst)
    recv_op = dist.irecv(recv_buffer, src)

    send_op.wait()
    recv_op.wait()

    return recv_buffer


class RingDilatedAttentionCorrect(nn.Module, HilbertAttentionMixin):
    """
    CORRECT Ring Dilated Attention with true O(n/k) memory per GPU.

    Key differences:
    - Accepts already-split local sequences
    - Never materializes full QKV
    - Efficient ring communication
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

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_hilbert = use_hilbert

        # Validate
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        assert len(segment_lengths) == len(dilation_rates)

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # GPU optimization
        gpu_info = get_gpu_info(device)
        if dtype is None:
            dtype = gpu_info.optimal_dtype
        self.dtype = dtype

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

        # Hilbert setup
        if use_hilbert:
            self.setup_hilbert_attention(
                hidden_dim=embed_dim,
                num_heads=num_heads,
                use_hilbert_core=False,
            )

        # Get world info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Attention backend
        self.attention_backend = select_attention_backend(
            device=device,
            seq_len=max(segment_lengths) // self.world_size,  # Local seq len!
            use_dilation=any(r > 1 for r in dilation_rates),
        )

        logger.info(
            f"RingAttentionCorrect initialized: rank={self.rank}/{self.world_size}, "
            f"backend={self.attention_backend}"
        )

    def forward(
        self,
        x_local: torch.Tensor,
        total_seq_len: Optional[int] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass on LOCAL sequence chunk.

        Args:
            x_local: Local sequence chunk [batch, local_seq_len, embed_dim]
                     where local_seq_len = total_seq_len / world_size
            total_seq_len: Total sequence length across all GPUs
            is_causal: Whether to use causal masking
            need_weights: Whether to return attention weights

        Returns:
            output: Local output [batch, local_seq_len, embed_dim]
            lse: Log-sum-exp for numerical stability (if need_weights=True)
        """
        batch_size, local_seq_len, _ = x_local.shape

        # Infer total sequence length if not provided
        if total_seq_len is None:
            total_seq_len = local_seq_len * self.world_size

        # QKV projection on LOCAL sequence only!
        qkv = self.qkv_proj(x_local)  # [batch, local_seq, 3*embed_dim]
        qkv = qkv.reshape(batch_size, local_seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, local_seq, dim]

        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]

        # Perform ring attention
        if self.world_size > 1:
            output, lse = self._ring_forward(
                q_local, k_local, v_local, local_seq_len, total_seq_len, is_causal
            )
        else:
            output = self._local_forward(q_local, k_local, v_local, is_causal)
            lse = None

        # Output projection
        output = output.transpose(1, 2).contiguous()  # [batch, local_seq, heads, dim]
        output = output.reshape(batch_size, local_seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout_layer(output)

        if need_weights:
            return output, lse
        return output

    def _ring_forward(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        local_seq_len: int,
        total_seq_len: int,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ring forward with proper memory efficiency.

        Each GPU:
        - Keeps its Q fixed
        - Receives K,V chunks from other GPUs
        - Accumulates attention outputs
        """
        batch_size, num_heads, _, head_dim = q_local.shape
        device = q_local.device

        # Initialize accumulators
        output = torch.zeros_like(q_local)
        lse = torch.full(
            (batch_size, num_heads, local_seq_len, 1),
            float("-inf"),
            device=device,
            dtype=q_local.dtype,
        )

        # Current K,V (start with local)
        k_chunk = k_local.clone()
        v_chunk = v_local.clone()

        # Ring steps
        for step in range(self.world_size):
            # Which global positions does this K,V represent?
            k_rank = (self.rank - step) % self.world_size
            k_start = k_rank * local_seq_len
            _ = k_start + local_seq_len

            # My Q positions
            q_start = self.rank * local_seq_len

            # Compute attention for this chunk
            chunk_out, chunk_lse = self._compute_chunk_attention(
                q_local,
                k_chunk,
                v_chunk,
                q_start,
                k_start,
                local_seq_len,
                total_seq_len,
                is_causal,
            )

            # Numerically stable accumulation
            max_lse = torch.maximum(lse, chunk_lse)

            output = output * torch.exp(lse - max_lse) + chunk_out * torch.exp(
                chunk_lse - max_lse
            )

            lse = (
                torch.log(torch.exp(lse - max_lse) + torch.exp(chunk_lse - max_lse))
                + max_lse
            )

            # Ring pass K,V
            if step < self.world_size - 1:
                k_chunk = ring_pass_forward(k_chunk)
                v_chunk = ring_pass_forward(v_chunk)

        return output, lse

    def _compute_chunk_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_start: int,
        k_start: int,
        chunk_len: int,
        total_seq_len: int,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention between chunks with proper masking."""
        batch_size, num_heads, q_len, head_dim = q.shape

        # Process segments with dilation
        output = torch.zeros_like(q)
        lse = torch.full(
            (batch_size, num_heads, q_len, 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )

        # Find which segments overlap with our chunks
        position = 0
        for seg_idx, (seg_len, dil_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            # Global segment boundaries
            seg_start = position
            seg_end = min(position + seg_len, total_seq_len)

            # Check overlap with Q chunk
            q_overlap_start = max(0, seg_start - q_start)
            q_overlap_end = min(q_len, seg_end - q_start)

            # Check overlap with K chunk
            k_overlap_start = max(0, seg_start - k_start)
            k_overlap_end = min(chunk_len, seg_end - k_start)

            if q_overlap_start < q_overlap_end and k_overlap_start < k_overlap_end:
                # We have overlap - compute attention for this segment
                q_seg = q[:, :, q_overlap_start:q_overlap_end]
                k_seg = k[:, :, k_overlap_start:k_overlap_end]
                v_seg = v[:, :, k_overlap_start:k_overlap_end]

                # Apply Hilbert if enabled
                if self.use_hilbert and q_seg.shape[2] > 1:
                    # Apply per-segment Hilbert ordering
                    q_seg_len = q_seg.shape[2]
                    k_seg_len = k_seg.shape[2]

                    if q_seg_len == k_seg_len:
                        # Same length - apply same ordering
                        indices = self.get_hilbert_indices(q_seg_len, q.device)
                        q_seg = q_seg.index_select(2, indices)
                        k_seg = k_seg.index_select(2, indices)
                        v_seg = v_seg.index_select(2, indices)

                # Compute dilated attention
                seg_out, seg_lse = self._compute_dilated_segment(
                    q_seg,
                    k_seg,
                    v_seg,
                    dil_rate,
                    q_start + q_overlap_start,
                    k_start + k_overlap_start,
                    is_causal,
                )

                # Reverse Hilbert if applied
                if self.use_hilbert and q_seg.shape[2] > 1 and q_seg_len == k_seg_len:
                    inv_indices = self.get_inverse_hilbert_indices(q_seg_len, q.device)
                    seg_out = seg_out.index_select(2, inv_indices)

                # Accumulate with LSE
                out_slice = output[:, :, q_overlap_start:q_overlap_end]
                lse_slice = lse[:, :, q_overlap_start:q_overlap_end]

                max_lse = torch.maximum(lse_slice, seg_lse)
                output[:, :, q_overlap_start:q_overlap_end] = out_slice * torch.exp(
                    lse_slice - max_lse
                ) + seg_out * torch.exp(seg_lse - max_lse)
                lse[:, :, q_overlap_start:q_overlap_end] = (
                    torch.log(
                        torch.exp(lse_slice - max_lse) + torch.exp(seg_lse - max_lse)
                    )
                    + max_lse
                )

            position = seg_end
            if position >= total_seq_len:
                break

        return output, lse

    def _compute_dilated_segment(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dilation_rate: int,
        q_global_offset: int,
        k_global_offset: int,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dilated attention for a segment."""
        if dilation_rate == 1:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Causal mask based on global positions
            if is_causal:
                q_positions = torch.arange(
                    q_global_offset, q_global_offset + q.shape[2], device=q.device
                )
                k_positions = torch.arange(
                    k_global_offset, k_global_offset + k.shape[2], device=k.device
                )
                mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
                scores.masked_fill_(mask, float("-inf"))

            # Compute attention
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            attn = torch.exp(scores - lse)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            output = torch.matmul(attn, v)

            return output, lse

        else:
            # Dilated attention - process each offset separately
            batch_size, num_heads, q_len, head_dim = q.shape
            output = torch.zeros_like(q)
            lse = torch.full(
                (batch_size, num_heads, q_len, 1),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )

            for offset in range(dilation_rate):
                # Get indices for this offset
                q_indices = torch.arange(offset, q_len, dilation_rate, device=q.device)
                k_indices = torch.arange(
                    offset, k.shape[2], dilation_rate, device=k.device
                )

                if len(q_indices) == 0 or len(k_indices) == 0:
                    continue

                # Extract dilated subsequences
                q_sub = q.index_select(2, q_indices)
                k_sub = k.index_select(2, k_indices)
                v_sub = v.index_select(2, k_indices)

                # Compute attention
                scores = torch.matmul(q_sub, k_sub.transpose(-2, -1)) * self.scale

                # Causal mask for dilated positions
                if is_causal and offset == 0:
                    q_pos = q_global_offset + q_indices
                    k_pos = k_global_offset + k_indices
                    mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
                    scores.masked_fill_(mask, float("-inf"))

                # Compute with LSE
                sub_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                attn = torch.exp(scores - sub_lse)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                sub_out = torch.matmul(attn, v_sub)

                # Scatter back with LSE accumulation
                output.index_copy_(2, q_indices, sub_out)
                lse.index_copy_(2, q_indices, sub_lse)

            return output, lse

    def _local_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """Single GPU forward."""
        # For single GPU, we can use the chunk attention directly
        output, _ = self._compute_chunk_attention(
            q, k, v, 0, 0, q.shape[2], q.shape[2], is_causal
        )
        return output


# Wrapper that handles sequence splitting
class RingAttentionWrapper(nn.Module):
    """
    Wrapper that ensures proper sequence splitting for ring attention.

    This handles the split of global sequences into local chunks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        **kwargs,
    ):
        super().__init__()
        self.attention = RingDilatedAttentionCorrect(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            **kwargs,
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        need_weights: bool = False,
        pre_split: bool = False,
    ) -> torch.Tensor:
        """
        Forward with automatic splitting.

        Args:
            x: Input tensor [batch, seq_len, embed_dim] OR
               [batch, local_seq_len, embed_dim] if pre_split=True
            is_causal: Causal masking
            need_weights: Return attention weights
            pre_split: If True, x is already split across GPUs
        """
        if pre_split or self.world_size == 1:
            # Already split or single GPU
            return self.attention(x, is_causal=is_causal, need_weights=need_weights)

        # Split sequence for ring attention
        batch_size, seq_len, embed_dim = x.shape
        assert seq_len % self.world_size == 0, (
            f"Sequence length {seq_len} must be divisible by world_size {self.world_size}"
        )

        local_seq_len = seq_len // self.world_size
        start = self.rank * local_seq_len
        end = start + local_seq_len

        # Get local chunk
        x_local = x[:, start:end, :].contiguous()

        # Process with ring attention
        output_local = self.attention(
            x_local,
            total_seq_len=seq_len,
            is_causal=is_causal,
            need_weights=need_weights,
        )

        if need_weights:
            output_local, lse = output_local

        # Optionally gather outputs (only if needed)
        if need_weights:
            return output_local, lse
        return output_local
