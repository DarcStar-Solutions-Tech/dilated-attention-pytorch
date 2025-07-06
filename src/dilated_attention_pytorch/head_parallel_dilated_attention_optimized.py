"""
Head-Parallel Dilated Attention - Optimized version with proper integration.

This version ensures all optimizations are properly utilized:
- Memory pool for efficient allocation
- Pattern caching for repeated patterns
- Flash Attention / xFormers backends
- Proper dilated attention computation
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

# Import optimization utilities
from .utils.attention_utils import optimize_attention_computation
from .core.constants import HAS_FLASH, HAS_XFORMERS, HAS_ENHANCED_MEMORY_POOL
from .core.memory_pool import get_enhanced_memory_pool


class HeadParallelDilatedAttentionOptimized(nn.Module):
    """
    Optimized head-parallel dilated attention for multi-GPU training.

    This version ensures all optimizations are properly utilized:
    - Persistent attention implementation (not recreated each forward)
    - Memory pool integration
    - Pattern caching
    - Optimized attention backends
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        use_xformers: bool = True,
        use_flex_attention: bool = False,
        use_flash_attention: bool = True,
        use_memory_pool: bool = True,
        use_pattern_cache: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize optimized head-parallel dilated attention."""
        super().__init__()

        # Store configuration
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_xformers = use_xformers
        self.use_flex_attention = use_flex_attention
        self.use_flash_attention = use_flash_attention
        self.use_memory_pool = use_memory_pool
        self.use_pattern_cache = use_pattern_cache
        self.device = device or (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.dtype = dtype or torch.float32

        # Initialize distributed if available
        self.world_size = 1
        self.rank = 0
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        # Create persistent attention implementation
        self._create_attention_impl()

        # Initialize memory pool if requested
        self._memory_pool = None
        if self.use_memory_pool and HAS_ENHANCED_MEMORY_POOL:
            try:
                self._memory_pool = get_enhanced_memory_pool()
                if self.rank == 0:
                    print("HeadParallelDilatedAttention: Memory pool initialized")
            except Exception as e:
                if self.rank == 0:
                    print(
                        f"HeadParallelDilatedAttention: Memory pool initialization failed: {e}"
                    )

        # Get optimized attention function
        self._attention_fn = optimize_attention_computation(
            prefer_flash=self.use_flash_attention and HAS_FLASH,
            prefer_xformers=self.use_xformers and HAS_XFORMERS,
        )

        if self.rank == 0:
            backend_name = "standard"
            if self._attention_fn.__name__ == "flash_attention":
                backend_name = "Flash Attention"
            elif HAS_XFORMERS and "xformers" in str(self._attention_fn):
                backend_name = "xFormers"
            print(f"HeadParallelDilatedAttention: Using {backend_name} backend")
            print(
                f"HeadParallelDilatedAttention: Initialized on {self.world_size} GPU(s)"
            )

    def _create_attention_impl(self):
        """Create persistent attention implementation."""
        try:
            from .improved_dilated_attention import ImprovedDilatedAttention

            # Create persistent model with all optimizations
            self._attention_impl = ImprovedDilatedAttention(
                segment_lengths=self.segment_lengths,
                dilation_rates=self.dilation_rates,
                dropout=self.dropout,
                use_xformers=self.use_xformers,
                use_flex_attention=self.use_flex_attention,
                use_memory_pool=self.use_memory_pool,
                use_pattern_cache=self.use_pattern_cache,
            )
            self._use_improved = True

        except ImportError:
            # Create optimized fallback
            self._attention_impl = None
            self._use_improved = False
            if self.rank == 0:
                print(
                    "Warning: ImprovedDilatedAttention not available, using optimized fallback"
                )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with head parallelism and optimizations.

        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            is_causal: Whether to apply causal masking

        Returns:
            output: (batch, seq_len, num_heads, head_dim)
        """
        b, n, h, d = q.shape

        # Single GPU fallback
        if self.world_size == 1:
            return self._process_all_heads_optimized(q, k, v, is_causal)

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
        local_output = self._process_all_heads_optimized(
            q_local, k_local, v_local, is_causal
        )

        # Efficient AllGather
        if self._memory_pool:
            # Use memory pool for communication buffers
            full_output = self._memory_pool.allocate(
                q.shape, dtype=q.dtype, device=q.device
            )
        else:
            full_output = torch.zeros_like(q)

        # Place local results
        full_output[:, :, head_start:head_end, :] = local_output

        # AllGather across GPUs (in-place operation)
        dist.all_reduce(full_output, op=dist.ReduceOp.SUM)

        return full_output

    def _process_all_heads_optimized(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Process all heads with optimized dilated attention."""
        # Use improved implementation if available
        if self._use_improved and self._attention_impl is not None:
            # Ensure we're not in autocast (let the impl handle it)
            with torch.cuda.amp.autocast(enabled=False):
                return self._attention_impl(q, k, v, is_causal=is_causal)

        # Otherwise use optimized fallback
        return self._compute_dilated_attention_optimized(q, k, v, is_causal)

    def _compute_dilated_attention_optimized(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
    ) -> Tensor:
        """Optimized dilated attention computation with proper backends."""
        b, n, h, d = q.shape

        # Calculate head groups
        num_groups = len(self.segment_lengths)
        heads_per_group = h // num_groups

        # Allocate output
        if self._memory_pool:
            output = self._memory_pool.allocate(q.shape, dtype=q.dtype, device=q.device)
        else:
            output = torch.zeros_like(q)

        # Process each head group with its segment configuration
        for group_idx, (seg_len, dil_rate) in enumerate(
            zip(self.segment_lengths, self.dilation_rates)
        ):
            head_start = group_idx * heads_per_group
            head_end = (
                (group_idx + 1) * heads_per_group if group_idx < num_groups - 1 else h
            )

            # Get head group tensors
            group_q = q[:, :, head_start:head_end, :]
            group_k = k[:, :, head_start:head_end, :]
            group_v = v[:, :, head_start:head_end, :]

            # Process segments with dilation
            group_output = self._process_dilated_segments(
                group_q, group_k, group_v, seg_len, dil_rate, is_causal
            )

            output[:, :, head_start:head_end, :] = group_output

        return output

    def _process_dilated_segments(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        segment_length: int,
        dilation_rate: int,
        is_causal: bool,
    ) -> Tensor:
        """Process segments with dilation using optimized backends."""
        b, n, h, d = q.shape

        # Ensure divisibility
        assert n % segment_length == 0, (
            f"Sequence length {n} must be divisible by segment length {segment_length}"
        )

        num_segments = n // segment_length
        output = torch.zeros_like(q)

        # Process each segment
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_length
            seg_end = seg_start + segment_length

            # Get segment queries
            q_seg = q[:, seg_start:seg_end]

            if dilation_rate == 1:
                # No dilation - process normally
                k_seg = k[:, seg_start:seg_end]
                v_seg = v[:, seg_start:seg_end]

                # Use optimized attention function
                seg_output = self._attention_fn(
                    q_seg,
                    k_seg,
                    v_seg,
                    is_causal=is_causal
                    and (seg_idx == 0),  # Only first segment needs causal
                    scale=1.0 / math.sqrt(d),
                )
            else:
                # Apply dilation pattern
                dilated_indices = self._get_dilated_indices(
                    seg_idx, segment_length, dilation_rate, n
                )

                # Gather dilated K,V
                k_dilated = k[:, dilated_indices]
                v_dilated = v[:, dilated_indices]

                # Compute attention with gathered values
                seg_output = self._attention_fn(
                    q_seg,
                    k_dilated,
                    v_dilated,
                    is_causal=False,  # Dilation breaks causality
                    scale=1.0 / math.sqrt(d),
                )

            output[:, seg_start:seg_end] = seg_output

        return output

    def _get_dilated_indices(
        self,
        segment_idx: int,
        segment_length: int,
        dilation_rate: int,
        total_length: int,
    ) -> Tensor:
        """Get indices for dilated attention pattern."""
        # Start position for this segment
        base_pos = segment_idx * segment_length

        # Create dilated indices
        indices = []
        for i in range(segment_length):
            # Dilated positions
            for j in range(segment_length):
                pos = base_pos + i * dilation_rate + j
                if pos < total_length:
                    indices.append(pos)

        # Ensure we have exactly segment_length indices
        indices = indices[:segment_length]
        while len(indices) < segment_length:
            indices.append(min(base_pos, total_length - 1))

        return torch.tensor(indices, device=self.device, dtype=torch.long)


class HeadParallelMultiheadDilatedAttentionOptimized(nn.Module):
    """
    Optimized drop-in replacement for nn.MultiheadAttention with head parallelism.
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
        use_flash_attention: bool = True,
        use_memory_pool: bool = True,
        use_pattern_cache: bool = True,
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

        # Optimized head-parallel attention
        self.attention = HeadParallelDilatedAttentionOptimized(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_xformers=use_xformers,
            use_flash_attention=use_flash_attention,
            use_memory_pool=use_memory_pool,
            use_pattern_cache=use_pattern_cache,
            device=device,
            dtype=dtype,
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with optimizations.

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

        # Apply optimized head-parallel attention
        attn_output = self.attention(q, k, v, is_causal=is_causal)

        # Reshape and project output
        attn_output = attn_output.reshape(b, n, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, None
