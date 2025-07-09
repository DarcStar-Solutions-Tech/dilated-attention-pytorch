"""
Fixed Ring Dilated Attention with Hilbert optimization - Version 2.

This version fixes the CUDA illegal memory access issues by:
1. Using safer ring communication utilities
2. Properly handling tensor contiguity
3. Adding synchronization barriers where needed
4. Using blocking send/recv for 2 GPU case to avoid deadlocks
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .ring_attention_utils_fixed import RingCommunicator, get_memory_info
from .utils.attention_utils import AttentionBackend, select_attention_backend
from .utils.gpu_utils import get_gpu_info
from .utils.hilbert_attention_mixin import HilbertAttentionMixin

logger = logging.getLogger(__name__)


class StableAttentionAccumulator:
    """Numerically stable attention accumulation using LSE trick."""

    def __init__(
        self, output_shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ):
        self.output = torch.zeros(output_shape, dtype=dtype, device=device)
        self.lse = torch.full(
            output_shape[:-1] + (1,), float("-inf"), dtype=dtype, device=device
        )

    def update(self, new_output: torch.Tensor, new_lse: torch.Tensor):
        """Update with numerically stable accumulation."""
        if new_lse.dim() == new_output.dim() - 1:
            new_lse = new_lse.unsqueeze(-1)

        max_lse = torch.maximum(self.lse, new_lse)

        # Numerically stable update
        self.output = self.output * torch.exp(
            self.lse - max_lse
        ) + new_output * torch.exp(new_lse - max_lse)
        self.lse = max_lse

    def get(self) -> torch.Tensor:
        """Get accumulated output."""
        return self.output


class RingDilatedAttentionHilbertOptimizedFixedV2(nn.Module, HilbertAttentionMixin):
    """
    Fixed Ring Dilated Attention with proper distributed communication.

    This version fixes CUDA errors by:
    - Using RingCommunicator for safe P2P operations
    - Ensuring all tensors are contiguous before communication
    - Proper synchronization between ranks
    - Special handling for 2-GPU case
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: list,
        dilation_rates: list,
        dropout: float = 0.1,
        use_hilbert: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        attention_backend: Optional[str] = None,
        memory_efficient: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_hilbert = use_hilbert
        self.memory_efficient = memory_efficient

        # Validate
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        assert len(segment_lengths) == len(dilation_rates)

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # GPU info
        self.gpu_info = get_gpu_info(device)

        # Optimal dtype
        if dtype is None:
            dtype = torch.float32  # Use FP32 for stability
        self.dtype = dtype

        # Backend selection
        if attention_backend is None:
            self.attention_backend = select_attention_backend(
                device=device,
                seq_len=max(segment_lengths),
                has_custom_mask=False,
                is_causal=False,
                use_dilation=any(rate > 1 for rate in dilation_rates),
            )
        else:
            self.attention_backend = attention_backend

        # Linear layers
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

        # Initialize ring communicator
        self.ring_comm = RingCommunicator()

        logger.info(
            f"Initialized RingDilatedAttentionHilbertOptimizedFixedV2 "
            f"(rank={self.ring_comm.rank}, world_size={self.ring_comm.world_size})"
        )

    def forward(
        self,
        x: torch.Tensor,
        total_seq_len: Optional[int] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        already_split: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with fixed ring communication.

        Args:
            x: Input tensor - either full sequence or local chunk
            total_seq_len: Total sequence length
            is_causal: Causal masking
            need_weights: Return attention weights
            already_split: If True, x is already the local chunk
        """
        batch_size = x.shape[0]

        # Handle distributed vs single GPU
        if (
            self.memory_efficient
            and self.ring_comm.world_size > 1
            and not already_split
        ):
            # Split sequence BEFORE QKV projection
            seq_len = x.shape[1]
            assert seq_len % self.ring_comm.world_size == 0

            local_seq_len = seq_len // self.ring_comm.world_size
            start = self.ring_comm.rank * local_seq_len
            end = start + local_seq_len

            x_local = x[:, start:end, :].contiguous()
            total_seq_len = seq_len
        else:
            x_local = x.contiguous()  # Ensure contiguous
            local_seq_len = x.shape[1]
            if total_seq_len is None:
                total_seq_len = local_seq_len * self.ring_comm.world_size

        # QKV projection on LOCAL sequence
        qkv = self.qkv_proj(x_local)
        qkv = qkv.reshape(batch_size, local_seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # Ensure contiguous

        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]

        # Make sure all are contiguous
        q_local = q_local.contiguous()
        k_local = k_local.contiguous()
        v_local = v_local.contiguous()

        # Ring attention or local attention
        if self.ring_comm.world_size > 1 and self.memory_efficient:
            output = self._ring_forward_fixed(
                q_local, k_local, v_local, local_seq_len, total_seq_len, is_causal
            )
        else:
            output = self._local_forward_optimized(q_local, k_local, v_local, is_causal)

        # Output projection
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, local_seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout_layer(output)

        if need_weights:
            return output, None
        return output

    def _ring_forward_fixed(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        local_seq_len: int,
        total_seq_len: int,
        is_causal: bool,
    ) -> torch.Tensor:
        """Fixed ring forward with safe communication."""
        batch_size, num_heads, _, head_dim = q_local.shape

        # Log memory at start
        if self.ring_comm.rank == 0:
            mem_info = get_memory_info(self.device)
            logger.info(f"Ring forward start - Memory: {mem_info}")

        # Initialize accumulator
        accumulator = StableAttentionAccumulator(
            output_shape=(batch_size, num_heads, local_seq_len, head_dim),
            dtype=q_local.dtype,
            device=q_local.device,
        )

        # Clone K,V for ring passing
        k_chunk = k_local.clone().contiguous()
        v_chunk = v_local.clone().contiguous()

        # Synchronize before starting ring passes
        self.ring_comm.barrier()

        for step in range(self.ring_comm.world_size):
            # Calculate which rank's K,V we're processing
            k_rank = (self.ring_comm.rank - step) % self.ring_comm.world_size
            k_start = k_rank * local_seq_len

            # Compute attention for this chunk
            chunk_out, chunk_lse = self._compute_chunk_attention_optimized(
                q_local,
                k_chunk,
                v_chunk,
                self.ring_comm.rank * local_seq_len,  # q_start
                k_start,  # k_start
                local_seq_len,
                total_seq_len,
                is_causal,
            )

            # Update accumulator
            accumulator.update(chunk_out, chunk_lse)

            # Ring pass (except on last step)
            if step < self.ring_comm.world_size - 1:
                # Use the safe ring communicator
                k_chunk, v_chunk = self.ring_comm.pass_kv(k_chunk, v_chunk)

                # Ensure contiguous after communication
                k_chunk = k_chunk.contiguous()
                v_chunk = v_chunk.contiguous()

        # Final synchronization
        self.ring_comm.barrier()

        return accumulator.get()

    def _compute_chunk_attention_optimized(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_start: int,
        k_start: int,
        seq_len: int,
        total_seq_len: int,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention for a chunk with optimization."""
        batch_size, num_heads, q_len, head_dim = q.shape
        k_len = k.shape[2]

        # Apply scaling
        q = q * self.scale

        # For each segment
        outputs = []
        lses = []

        for seg_idx, (seg_len, dilation) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            if seg_len > seq_len:
                continue

            # Apply Hilbert ordering if enabled
            if self.use_hilbert and hasattr(self, "hilbert_curve_ids"):
                q_seg = self.apply_hilbert_ordering(
                    q, self.hilbert_curve_ids[seg_idx], inverse=False
                )
                k_seg = self.apply_hilbert_ordering(
                    k, self.hilbert_curve_ids[seg_idx], inverse=False
                )
                v_seg = self.apply_hilbert_ordering(
                    v, self.hilbert_curve_ids[seg_idx], inverse=False
                )
            else:
                q_seg = q
                k_seg = k
                v_seg = v

            # Compute attention based on backend
            if self.attention_backend == AttentionBackend.FLASH_ATTENTION:
                # Use Flash Attention if available
                out = self._flash_attention_forward(
                    q_seg, k_seg, v_seg, causal=is_causal, window_size=seg_len
                )
                # Approximate LSE for Flash Attention
                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1))
                lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            else:
                # Standard attention
                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1))

                # Apply causal mask if needed
                if is_causal and k_start >= q_start:
                    causal_mask = torch.triu(
                        torch.full((q_len, k_len), float("-inf"), device=q.device),
                        diagonal=k_start - q_start + 1,
                    )
                    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

                # Compute LSE for numerical stability
                lse = torch.logsumexp(scores, dim=-1, keepdim=True)

                # Apply softmax
                attn_weights = torch.exp(scores - lse)

                # Apply attention
                out = torch.matmul(attn_weights, v_seg)

            # Apply inverse Hilbert if needed
            if self.use_hilbert and hasattr(self, "hilbert_curve_ids"):
                out = self.apply_hilbert_ordering(
                    out, self.hilbert_curve_ids[seg_idx], inverse=True
                )

            outputs.append(out)
            lses.append(lse.squeeze(-1))

        # Combine outputs from all segments
        if len(outputs) == 1:
            return outputs[0], lses[0]
        else:
            # Use stable accumulation for multiple segments
            combined_accumulator = StableAttentionAccumulator(
                output_shape=outputs[0].shape,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
            )
            for out, lse in zip(outputs, lses):
                combined_accumulator.update(out, lse)

            final_output = combined_accumulator.get()
            final_lse = combined_accumulator.lse.squeeze(-1)
            return final_output, final_lse

    def _local_forward_optimized(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """Local attention forward (single GPU or non-distributed)."""
        # For single GPU, just compute attention normally
        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))

        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=q.device),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        window_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Flash Attention forward pass."""
        try:
            from flash_attn import flash_attn_func

            # Flash attention expects (batch, seq, heads, dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
                window_size=(window_size, window_size) if window_size else (-1, -1),
            )

            # Transpose back
            return out.transpose(1, 2)
        except ImportError:
            # Fall back to standard attention
            return self._local_forward_optimized(q, k, v, causal)
