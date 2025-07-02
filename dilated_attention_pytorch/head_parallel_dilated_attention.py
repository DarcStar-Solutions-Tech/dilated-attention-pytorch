"""
Head-Parallel Dilated Attention - A better approach for multi-GPU.

Instead of splitting sequences across GPUs (ring attention), we split attention heads.
This preserves the locality needed for efficient dilated attention patterns.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist


class HeadParallelDilatedAttention(nn.Module):
    """
    Head-parallel dilated attention for multi-GPU training.

    Key insight: Dilated attention patterns require sequence locality.
    Ring attention breaks this locality, causing massive overhead.
    Head parallelism keeps full sequences on each GPU, splitting by attention heads.

    Benefits:
    - Each GPU processes complete sequences (no pattern breaking)
    - Only one AllReduce needed at the end
    - Efficient batch processing of segments
    - No ring communication overhead
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        use_xformers: bool = True,
        use_flex_attention: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize head-parallel dilated attention."""
        # Initialize distributed if available
        self.world_size = 1
        self.rank = 0
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        # Call parent constructor
        super().__init__()

        # Store configuration
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_xformers = use_xformers
        self.use_flex_attention = use_flex_attention
        self.device = device or torch.cuda.current_device()
        self.dtype = dtype or torch.float32

        # Create persistent attention implementation
        self._attention_impl = None
        self._create_attention_impl()

        print(
            f"HeadParallelDilatedAttention initialized on rank {self.rank}/{self.world_size}"
        )

    def _create_attention_impl(self):
        """Create persistent attention implementation with optimizations."""
        try:
            from .improved_dilated_attention import ImprovedDilatedAttention

            # Create persistent model with optimizations enabled
            self._attention_impl = ImprovedDilatedAttention(
                segment_lengths=self.segment_lengths,
                dilation_rates=self.dilation_rates,
                dropout=self.dropout,
            )

            # Set optimization flags if available
            if hasattr(self._attention_impl, "use_memory_pool"):
                self._attention_impl.use_memory_pool = True
            if hasattr(self._attention_impl, "use_pattern_cache"):
                self._attention_impl.use_pattern_cache = True

            if self.rank == 0:
                print(
                    "HeadParallelDilatedAttention: Using ImprovedDilatedAttention with optimizations"
                )
        except Exception as e:
            if self.rank == 0:
                print(
                    f"HeadParallelDilatedAttention: ImprovedDilatedAttention not available: {e}"
                )
            self._attention_impl = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with head parallelism.

        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            is_causal: Whether to apply causal masking

        Returns:
            output: (batch, seq_len, num_heads, head_dim)
        """
        b, n, h, d = q.shape

        # Single GPU fallback
        if self.world_size == 1:
            return self._process_all_heads(q, k, v, is_causal)

        # Multi-GPU: Split heads across GPUs
        assert h % self.world_size == 0, (
            f"Number of heads {h} must be divisible by world size {self.world_size}"
        )

        heads_per_gpu = h // self.world_size
        head_start = self.rank * heads_per_gpu
        head_end = head_start + heads_per_gpu

        # Get this GPU's subset of heads
        q_local = q[:, :, head_start:head_end, :].contiguous()
        k_local = k[:, :, head_start:head_end, :].contiguous()
        v_local = v[:, :, head_start:head_end, :].contiguous()

        # Process local heads with FULL sequence
        local_output = self._process_all_heads(q_local, k_local, v_local, is_causal)

        # AllGather to combine results from all GPUs
        output_list = [torch.zeros_like(q) for _ in range(self.world_size)]

        # Create padded tensor for allgather
        full_output = torch.zeros_like(q)
        full_output[:, :, head_start:head_end, :] = local_output

        # AllGather across GPUs
        dist.all_gather(output_list, full_output)

        # Sum the outputs (each GPU contributed different heads)
        output = torch.zeros_like(q)
        for i, gpu_output in enumerate(output_list):
            gpu_head_start = i * heads_per_gpu
            gpu_head_end = gpu_head_start + heads_per_gpu
            output[:, :, gpu_head_start:gpu_head_end, :] = gpu_output[
                :, :, gpu_head_start:gpu_head_end, :
            ]

        return output

    def _process_all_heads(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Process all heads with dilated attention."""
        b, n, h, d = q.shape

        # Use persistent implementation if available
        if self._attention_impl is not None:
            # Update dropout for training mode
            self._attention_impl.dropout = self.dropout if self.training else 0.0

            # Process with improved implementation
            with torch.cuda.amp.autocast(enabled=False):
                output = self._attention_impl(q, k, v, is_causal=is_causal)

            return output
        else:
            # Fallback to optimized computation
            return self._compute_dilated_attention_optimized(q, k, v, is_causal)

    def _compute_dilated_attention_optimized(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Optimized dilated attention computation with segment processing."""
        b, n, h, d = q.shape

        # Import attention utilities
        try:
            from .utils.attention_utils import optimize_attention_computation

            attention_fn = optimize_attention_computation()
        except ImportError:
            # Fallback to scaled_dot_product_attention
            attention_fn = torch.nn.functional.scaled_dot_product_attention

        # Calculate head groups
        heads_per_group = h // len(self.segment_lengths)
        output = torch.zeros_like(q)

        # Process each head group with its segment configuration
        for i, (seg_len, dil_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            head_start = i * heads_per_group
            head_end = (
                (i + 1) * heads_per_group if i < len(self.segment_lengths) - 1 else h
            )

            # Process this head group
            group_q = q[:, :, head_start:head_end, :]
            group_k = k[:, :, head_start:head_end, :]
            group_v = v[:, :, head_start:head_end, :]

            # Process segments with dilation
            group_output = self._process_segments_with_dilation(
                group_q, group_k, group_v, seg_len, dil_rate, is_causal, attention_fn
            )

            output[:, :, head_start:head_end, :] = group_output

        return output

    def _process_segments_with_dilation(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        segment_length: int,
        dilation_rate: int,
        is_causal: bool,
        attention_fn,
    ) -> Tensor:
        """Process segments with proper dilation pattern."""
        b, n, h, d = q.shape

        # Ensure divisibility
        if n % segment_length != 0:
            # Pad to make divisible
            pad_len = segment_length - (n % segment_length)
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
            padded_n = n + pad_len
        else:
            padded_n = n
            pad_len = 0

        num_segments = padded_n // segment_length
        output = torch.zeros_like(q)

        # Process each segment
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_length
            seg_end = seg_start + segment_length

            # Get segment queries
            q_seg = q[:, seg_start:seg_end]

            if dilation_rate == 1:
                # No dilation needed
                k_seg = k[:, seg_start:seg_end]
                v_seg = v[:, seg_start:seg_end]
            else:
                # Apply dilation pattern
                # For each position in segment, gather keys/values with dilation
                indices = []
                for i in range(segment_length):
                    for j in range(0, segment_length, dilation_rate):
                        idx = seg_start + i - (i % dilation_rate) + j
                        if 0 <= idx < padded_n:
                            indices.append(idx)

                # Ensure we have exactly segment_length indices
                indices = indices[:segment_length]
                while len(indices) < segment_length:
                    indices.append(min(seg_start, padded_n - 1))

                indices_tensor = torch.tensor(
                    indices, device=q.device, dtype=torch.long
                )
                k_seg = k.index_select(1, indices_tensor)
                v_seg = v.index_select(1, indices_tensor)

            # Compute attention for this segment
            seg_output = attention_fn(
                q_seg,
                k_seg,
                v_seg,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and seg_idx == 0,  # Only first segment needs causal
            )

            output[:, seg_start:seg_end] = seg_output

        # Remove padding if added
        if pad_len > 0:
            output = output[:, :n]

        return output


class HeadParallelMultiheadDilatedAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention with head parallelism.

    This is the recommended way to use dilated attention in multi-GPU settings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_xformers: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # Head-parallel attention
        self.attention = HeadParallelDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_xformers=use_xformers,
            device=device,
            dtype=dtype,
        )

        # Get distributed info
        self.world_size = 1
        self.rank = 0
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.

        Args:
            query, key, value: (batch, seq_len, embed_dim)
            is_causal: Whether to apply causal masking
            need_weights: Whether to return attention weights (not supported)

        Returns:
            output: (batch, seq_len, embed_dim)
            attn_weights: None (not supported for efficiency)
        """
        if need_weights:
            raise ValueError(
                "need_weights=True is not supported for HeadParallelMultiheadDilatedAttention"
            )

        b, n, _ = query.shape

        # Project to Q, K, V
        q = self.q_proj(query).view(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(b, n, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(b, n, self.num_heads, self.head_dim)

        # Apply head-parallel attention
        attn_output = self.attention(q, k, v, is_causal=is_causal)

        # Reshape and project output
        attn_output = attn_output.reshape(b, n, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, None
