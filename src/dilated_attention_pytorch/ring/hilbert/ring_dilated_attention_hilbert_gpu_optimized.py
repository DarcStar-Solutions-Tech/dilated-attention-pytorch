"""
Ring Dilated Attention with Hilbert optimization and GPU-aware backend selection.

This implementation combines:
1. Ring attention with proper isend/irecv communication
2. Per-segment dilated attention
3. Per-segment Hilbert SFC ordering
4. Automatic GPU detection and optimal backend selection
5. Support for Flash Attention, SDPA, xformers, and manual backends
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List, Tuple, Union, Dict
import logging

from ..utils.ring_attention_utils import all_ring_pass, split_by_rank
from ...utils.hilbert_attention_mixin import HilbertAttentionMixin
from ...utils.gpu_utils import get_gpu_info, select_attention_backend
from ...utils.flash_attention_utils import flash_attention_forward

logger = logging.getLogger(__name__)


class StableAttentionAccumulator:
    """Numerically stable attention accumulation using LSE trick."""

    def __init__(
        self, output_shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ):
        """Initialize accumulator with zeros and -inf for LSE."""
        self.output = torch.zeros(output_shape, dtype=dtype, device=device)
        self.lse = torch.full(
            output_shape[:-1] + (1,), float("-inf"), dtype=dtype, device=device
        )

    def update(self, new_output: torch.Tensor, new_lse: torch.Tensor):
        """Update accumulator with new attention output and its LSE."""
        # Ensure LSE has correct shape
        if new_lse.dim() == new_output.dim() - 1:
            new_lse = new_lse.unsqueeze(-1)

        # Stable accumulation: out = out * exp(lse - new_lse) + new_out
        max_lse = torch.maximum(self.lse, new_lse)

        self.output = self.output * torch.exp(
            self.lse - max_lse
        ) + new_output * torch.exp(new_lse - max_lse)

        # Update LSE: lse = log(exp(lse - max_lse) + exp(new_lse - max_lse)) + max_lse
        self.lse = (
            torch.log(torch.exp(self.lse - max_lse) + torch.exp(new_lse - max_lse))
            + max_lse
        )

    def get(self) -> torch.Tensor:
        """Get final accumulated output."""
        return self.output


class RingDilatedAttentionHilbertGPUOptimized(nn.Module, HilbertAttentionMixin):
    """
    GPU-optimized Ring Dilated Attention with per-segment Hilbert optimization.

    Features:
    - Automatic GPU detection and backend selection
    - Support for Flash Attention 3/2/1, SDPA, xformers, and manual
    - Optimized dtype selection based on GPU architecture
    - Per-segment dilated attention with Hilbert ordering
    - Efficient ring communication without all_gather
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        bias: bool = True,
        ring_size: Optional[int] = None,
        use_hilbert: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        attention_backend: Optional[str] = None,
        benchmark_backends: bool = False,
    ):
        """
        Initialize GPU-optimized Ring Dilated Attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            segment_lengths: List of segment lengths for dilated attention
            dilation_rates: Corresponding dilation rates for each segment
            dropout: Dropout probability
            bias: Whether to use bias in projections
            ring_size: Number of devices in ring (auto-detected if None)
            use_hilbert: Whether to use Hilbert curve ordering
            device: Device for parameters (auto-detected if None)
            dtype: Data type for parameters (auto-selected if None)
            attention_backend: Force specific backend (auto-select if None)
            benchmark_backends: Whether to benchmark backends on init
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_hilbert = use_hilbert

        # Validate parameters
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        )
        assert len(segment_lengths) == len(dilation_rates), (
            "segment_lengths and dilation_rates must have same length"
        )

        # GPU detection and optimization
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Get GPU info and optimal settings
        self.gpu_info = get_gpu_info(device)
        logger.info(
            f"Detected GPU: {self.gpu_info.name} ({self.gpu_info.architecture})"
        )
        logger.info(f"Compute capability: {self.gpu_info.compute_capability}")

        # Set dtype based on GPU architecture if not specified
        if dtype is None:
            dtype = self.gpu_info.optimal_dtype
            logger.info(f"Selected optimal dtype: {dtype}")
        self.dtype = dtype

        # Select attention backend
        if attention_backend is None:
            # Check if we're using dilation
            use_dilation = any(rate > 1 for rate in dilation_rates)
            self.attention_backend = select_attention_backend(
                device=device,
                seq_len=max(segment_lengths),
                has_custom_mask=False,
                is_causal=False,  # Will be set dynamically
                use_dilation=use_dilation,
            )
            logger.info(f"Selected attention backend: {self.attention_backend}")
        else:
            self.attention_backend = attention_backend

        # Benchmark backends if requested
        if benchmark_backends and device.type == "cuda":
            self._benchmark_backends()

        # QKV projection
        self.qkv_proj = nn.Linear(
            embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype
        )

        # Output projection
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Setup Hilbert attention mixin
        self.setup_hilbert_attention(
            hidden_dim=embed_dim,
            num_heads=num_heads,
            use_hilbert_core=False,  # We'll use manual Hilbert ordering
        )

        # Ring configuration
        self.ring_size = ring_size
        if dist.is_initialized() and self.ring_size is None:
            self.ring_size = dist.get_world_size()
        elif self.ring_size is None:
            self.ring_size = 1

        self.is_distributed = dist.is_initialized() and self.ring_size > 1
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            logger.info(f"Initialized ring attention with size {self.ring_size}")

        # Cache for backend performance
        self._backend_cache: Dict[str, float] = {}

    def _benchmark_backends(self):
        """Benchmark available attention backends."""
        logger.info("Benchmarking attention backends...")

        # Test with typical segment size
        test_seq_len = min(max(self.segment_lengths), 2048)

        from ...utils.gpu_utils import benchmark_attention_backends

        results = benchmark_attention_backends(
            device=self.device,
            batch_size=1,
            seq_len=test_seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # Log results
        for backend, time_ms in sorted(results.items(), key=lambda x: x[1]):
            if time_ms < float("inf"):
                logger.info(f"  {backend}: {time_ms:.2f} ms")
            else:
                logger.info(f"  {backend}: FAILED")

        # Update backend if auto-selected and a faster one exists
        if hasattr(self, "_auto_selected_backend"):
            current_time = results.get(self.attention_backend, float("inf"))
            best_backend = min(results.items(), key=lambda x: x[1])
            if best_backend[1] < current_time * 0.8:  # At least 20% faster
                logger.info(
                    f"Switching backend from {self.attention_backend} to {best_backend[0]}"
                )
                self.attention_backend = best_backend[0]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass with GPU-optimized ring dilated attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking
            need_weights: Whether to return attention weights

        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            attn_weights: Optional attention weights (always None for ring attention)
        """
        batch_size, seq_len, _ = x.shape

        # Convert to optimal dtype if needed
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply attention
        if self.is_distributed and self.ring_size > 1:
            output = self._ring_forward(q, k, v, attn_mask, is_causal)
        else:
            output = self._single_gpu_forward(q, k, v, attn_mask, is_causal)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()  # [batch, seq, heads, dim]
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout_layer(output)

        if need_weights:
            # Ring attention doesn't support returning weights
            return output, None
        return output

    def _single_gpu_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Single GPU forward with per-segment dilated attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Initialize output
        output = torch.zeros_like(q)

        # Process each segment
        position = 0
        for seg_idx, (seg_len, dil_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            if position >= seq_len:
                break

            # Calculate segment boundaries
            seg_end = min(position + seg_len, seq_len)
            actual_seg_len = seg_end - position

            # Extract segment
            q_seg = q[:, :, position:seg_end]
            k_seg = k[:, :, position:seg_end]
            v_seg = v[:, :, position:seg_end]

            # Apply Hilbert ordering to segment if enabled
            if self.use_hilbert and actual_seg_len > 1:
                # Get Hilbert indices for this segment size
                hilbert_indices = self.get_hilbert_indices(actual_seg_len, device)

                # Apply Hilbert ordering
                q_seg = q_seg.index_select(2, hilbert_indices)
                k_seg = k_seg.index_select(2, hilbert_indices)
                v_seg = v_seg.index_select(2, hilbert_indices)

            # Apply dilated attention within segment
            seg_output = self._compute_dilated_attention_optimized(
                q_seg, k_seg, v_seg, dil_rate, is_causal and seg_idx == 0
            )

            # Reverse Hilbert ordering if applied
            if self.use_hilbert and actual_seg_len > 1:
                inverse_indices = self.get_inverse_hilbert_indices(
                    actual_seg_len, device
                )
                seg_output = seg_output.index_select(2, inverse_indices)

            # Store segment output
            output[:, :, position:seg_end] = seg_output
            position = seg_end

        return output

    def _compute_dilated_attention_optimized(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dilation_rate: int,
        is_causal: bool,
    ) -> torch.Tensor:
        """Compute dilated attention using optimal backend."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # For dilation rate 1, use optimized backend
        if dilation_rate == 1:
            # Convert to [batch, seq, heads, dim] for flash_attention_forward
            q_reshaped = q.transpose(1, 2).contiguous()
            k_reshaped = k.transpose(1, 2).contiguous()
            v_reshaped = v.transpose(1, 2).contiguous()

            # Use appropriate backend
            if self.attention_backend in [
                "flash_attn_3",
                "flash_attn_2",
                "flash_attn",
                "xformers",
            ]:
                # These backends handle the computation efficiently
                output = flash_attention_forward(
                    q_reshaped,
                    k_reshaped,
                    v_reshaped,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    backend=self.attention_backend,
                )
                # Convert back to [batch, heads, seq, dim]
                return output.transpose(1, 2).contiguous()

            elif self.attention_backend == "sdpa":
                # SDPA expects [batch, heads, seq, dim]
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True,
                ):
                    output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal,
                        scale=self.scale,
                    )
                return output

            else:  # manual
                return self._compute_manual_attention(q, k, v, is_causal)

        else:
            # For dilated attention, we need custom implementation
            return self._compute_dilated_attention_manual(
                q, k, v, dilation_rate, is_causal
            )

    def _compute_manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """Manual attention computation."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), float("-inf"), device=q.device, dtype=q.dtype
                ),
                diagonal=1,
            )
            scores = scores + causal_mask

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Dropout
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # Apply attention
        return torch.matmul(attn_weights, v)

    def _compute_dilated_attention_manual(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dilation_rate: int,
        is_causal: bool,
    ) -> torch.Tensor:
        """Manual dilated attention computation."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Initialize output
        output = torch.zeros_like(q)

        for offset in range(dilation_rate):
            # Get dilated indices
            indices = torch.arange(offset, seq_len, dilation_rate, device=device)
            if len(indices) == 0:
                continue

            # Extract dilated subsequences
            q_dilated = q.index_select(2, indices)
            k_dilated = k.index_select(2, indices)
            v_dilated = v.index_select(2, indices)

            # Compute attention on dilated subsequence
            if len(indices) == 1:
                # Single position - just copy
                dilated_output = v_dilated
            else:
                # Use optimized backend for subsequence
                if self.attention_backend == "sdpa":
                    dilated_output = F.scaled_dot_product_attention(
                        q_dilated,
                        k_dilated,
                        v_dilated,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal and offset == 0,
                        scale=self.scale,
                    )
                else:
                    # Manual computation
                    scores = (
                        torch.matmul(q_dilated, k_dilated.transpose(-2, -1))
                        * self.scale
                    )

                    if is_causal and offset == 0:
                        dilated_len = len(indices)
                        causal_mask = torch.triu(
                            torch.full(
                                (dilated_len, dilated_len),
                                float("-inf"),
                                device=device,
                                dtype=q.dtype,
                            ),
                            diagonal=1,
                        )
                        scores = scores + causal_mask

                    attn_weights = F.softmax(scores, dim=-1)
                    if self.dropout > 0 and self.training:
                        attn_weights = F.dropout(
                            attn_weights, p=self.dropout, training=True
                        )

                    dilated_output = torch.matmul(attn_weights, v_dilated)

            # Scatter back to full sequence
            output.index_copy_(2, indices, dilated_output)

        return output

    def _ring_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Multi-GPU ring forward with proper communication."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Split sequences across ring
        q_local = split_by_rank(q, self.rank, self.world_size, dim=2)
        k_local = split_by_rank(k, self.rank, self.world_size, dim=2)
        v_local = split_by_rank(v, self.rank, self.world_size, dim=2)

        local_seq_len = q_local.shape[2]

        # Initialize accumulator
        accumulator = StableAttentionAccumulator(
            output_shape=(batch_size, num_heads, local_seq_len, head_dim),
            dtype=q.dtype,
            device=q.device,
        )

        # Ring communication loop
        k_chunk = k_local.clone()
        v_chunk = v_local.clone()

        for step in range(self.ring_size):
            # Compute position in global sequence
            chunk_start = ((self.rank - step) % self.ring_size) * local_seq_len

            # Compute attention for current chunk with dilated segments
            attn_out, attn_lse = self._compute_ring_chunk_attention(
                q_local,
                k_chunk,
                v_chunk,
                chunk_start,
                seq_len,
                is_causal and (self.rank == 0) and (step == 0),
            )

            # Update accumulator
            if step == 0:
                accumulator.output = attn_out
                accumulator.lse = attn_lse
            else:
                accumulator.update(attn_out, attn_lse)

            # Ring pass K and V chunks
            if step < self.ring_size - 1:
                k_chunk = all_ring_pass(self.ring_size, k_chunk)
                v_chunk = all_ring_pass(self.ring_size, v_chunk)

        return accumulator.get()

    def _compute_ring_chunk_attention(
        self,
        q_chunk: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        chunk_start: int,
        total_seq_len: int,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention for one ring chunk with proper segments."""
        batch_size, num_heads, chunk_len, head_dim = q_chunk.shape
        device = q_chunk.device

        # Initialize outputs
        output = torch.zeros_like(q_chunk)
        lse = torch.full(
            (batch_size, num_heads, chunk_len, 1),
            float("-inf"),
            dtype=q_chunk.dtype,
            device=device,
        )

        # Find which segments this chunk overlaps with
        position = 0
        for seg_idx, (seg_len, dil_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            seg_end = min(position + seg_len, total_seq_len)

            # Check if this segment overlaps with our query chunk
            if position < chunk_start + chunk_len and seg_end > chunk_start:
                # Calculate overlap
                overlap_start = max(0, position - chunk_start)
                overlap_end = min(chunk_len, seg_end - chunk_start)

                if overlap_end > overlap_start:
                    # Extract overlapping portions
                    q_overlap = q_chunk[:, :, overlap_start:overlap_end]
                    k_overlap = k_chunk[:, :, overlap_start:overlap_end]
                    v_overlap = v_chunk[:, :, overlap_start:overlap_end]

                    # Apply Hilbert ordering if enabled
                    overlap_len = overlap_end - overlap_start
                    if self.use_hilbert and overlap_len > 1:
                        hilbert_indices = self.get_hilbert_indices(overlap_len, device)
                        q_overlap = q_overlap.index_select(2, hilbert_indices)
                        k_overlap = k_overlap.index_select(2, hilbert_indices)
                        v_overlap = v_overlap.index_select(2, hilbert_indices)

                    # Compute dilated attention with LSE
                    seg_out, seg_lse = self._compute_dilated_attention_with_lse(
                        q_overlap,
                        k_overlap,
                        v_overlap,
                        dil_rate,
                        is_causal and seg_idx == 0 and overlap_start == 0,
                    )

                    # Reverse Hilbert ordering if applied
                    if self.use_hilbert and overlap_len > 1:
                        inverse_indices = self.get_inverse_hilbert_indices(
                            overlap_len, device
                        )
                        seg_out = seg_out.index_select(2, inverse_indices)
                        seg_lse = seg_lse.index_select(2, inverse_indices)

                    # Update output with proper LSE accumulation
                    out_slice = output[:, :, overlap_start:overlap_end]
                    lse_slice = lse[:, :, overlap_start:overlap_end]

                    # Stable accumulation
                    max_lse = torch.maximum(lse_slice, seg_lse)
                    output[:, :, overlap_start:overlap_end] = out_slice * torch.exp(
                        lse_slice - max_lse
                    ) + seg_out * torch.exp(seg_lse - max_lse)
                    lse[:, :, overlap_start:overlap_end] = (
                        torch.log(
                            torch.exp(lse_slice - max_lse)
                            + torch.exp(seg_lse - max_lse)
                        )
                        + max_lse
                    )

            position = seg_end
            if position >= total_seq_len:
                break

        return output, lse

    def _compute_dilated_attention_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dilation_rate: int,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dilated attention with LSE for stable accumulation."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        if dilation_rate == 1:
            # Standard attention - compute LSE
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if is_causal:
                causal_mask = torch.triu(
                    torch.full(
                        (seq_len, seq_len), float("-inf"), device=device, dtype=q.dtype
                    ),
                    diagonal=1,
                )
                scores = scores + causal_mask

            # Compute LSE for numerical stability
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            attn_weights = torch.exp(scores - lse)

            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

            output = torch.matmul(attn_weights, v)
            return output, lse

        else:
            # Dilated attention with LSE
            output = torch.zeros_like(q)
            lse = torch.full(
                (batch_size, num_heads, seq_len, 1),
                float("-inf"),
                dtype=q.dtype,
                device=device,
            )

            for offset in range(dilation_rate):
                indices = torch.arange(offset, seq_len, dilation_rate, device=device)
                if len(indices) == 0:
                    continue

                # Extract dilated subsequences
                q_dilated = q.index_select(2, indices)
                k_dilated = k.index_select(2, indices)
                v_dilated = v.index_select(2, indices)

                # Compute attention on dilated subsequence
                scores = (
                    torch.matmul(q_dilated, k_dilated.transpose(-2, -1)) * self.scale
                )

                if is_causal and offset == 0:
                    dilated_len = len(indices)
                    causal_mask = torch.triu(
                        torch.full(
                            (dilated_len, dilated_len),
                            float("-inf"),
                            device=device,
                            dtype=q.dtype,
                        ),
                        diagonal=1,
                    )
                    scores = scores + causal_mask

                # Compute LSE for this offset
                offset_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                attn_weights = torch.exp(scores - offset_lse)

                if self.dropout > 0 and self.training:
                    attn_weights = F.dropout(
                        attn_weights, p=self.dropout, training=True
                    )

                # Compute output for dilated positions
                dilated_output = torch.matmul(attn_weights, v_dilated)

                # Scatter back with LSE
                output.index_copy_(2, indices, dilated_output)
                lse.index_copy_(2, indices, offset_lse)

            return output, lse

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"ring_size={self.ring_size}, "
            f"use_hilbert={self.use_hilbert}, "
            f"gpu={self.gpu_info.name}, "
            f"backend={self.attention_backend}, "
            f"dtype={self.dtype}"
        )
