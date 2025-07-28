"""
Block-Sparse Dilated Attention

This implementation combines block-sparse patterns with true dilated attention
(token-level dilation within segments). It provides the memory efficiency of
block sparsity with the multi-scale modeling capability of dilated attention.

Key features:
- Block-level sparsity for computational efficiency
- Token-level dilation within attending blocks
- Segment-based processing with configurable dilation rates
- Memory-efficient implementation
- Support for causal masking
"""

import math
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from torch import Tensor

from .block_sparse_attention import (
    BlockSparseAttention,
    SparsePatternConfig,
)
from ..base.dilated_attention import DilatedAttention


class BlockSparseDilatedAttention(torch.nn.Module):
    """
    Block-Sparse Dilated Attention combining block sparsity with token-level dilation.

    This implementation:
    1. Uses block-sparse patterns to determine which blocks can attend to each other
    2. Within attending block pairs, applies dilated attention with segment-based dilation

    Args:
        segment_lengths: List of segment lengths for dilated attention
        dilation_rates: List of dilation rates corresponding to each segment
        sparse_config: Configuration for block-sparse patterns
        dropout: Dropout probability
        use_xformers: Whether to use xformers for efficient attention
        device: Device to run on
        dtype: Data type to use
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: Optional[SparsePatternConfig] = None,
        dropout: float = 0.0,
        use_xformers: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Validate inputs
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError(
                f"segment_lengths and dilation_rates must have same length, "
                f"got {len(segment_lengths)} and {len(dilation_rates)}"
            )

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.use_xformers = use_xformers

        # Default sparse config if not provided
        if sparse_config is None:
            sparse_config = SparsePatternConfig(
                pattern_type="local_window",
                sparsity_ratio=0.1,  # 90% sparse
                block_size=max(segment_lengths),  # Block size = largest segment
            )
        self.sparse_config = sparse_config

        # Create block sparse attention for pattern generation
        self.block_sparse = BlockSparseAttention(
            sparse_config=sparse_config,
            dropout=0.0,  # We'll handle dropout ourselves
            device=device,
            dtype=dtype,
        )

        # Create dilated attention for token-level processing
        # We'll use a single instance and reconfigure per block pair
        self.dilated_attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
            op=None if not use_xformers else None,  # Let it auto-detect
        )

        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Forward pass combining block sparsity with dilated attention.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            return_attention_weights: Whether to return attention weights

        Returns:
            output: Attention output [batch, seq_len, num_heads, head_dim]
            attention_weights: Optional dict with sparse attention info
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        block_size = self.sparse_config.block_size

        # Ensure sequence length is compatible
        if seq_len % block_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by block size {block_size}"
            )

        num_blocks = seq_len // block_size

        # Get sparse block indices
        row_indices, col_indices = self.block_sparse._get_sparse_block_indices(
            num_blocks, num_heads, q.device
        )

        # Initialize output
        output = torch.zeros_like(q)

        # Process each block pair that should attend
        attention_info = {
            "block_pairs": [],
            "sparsity_ratio": 1.0 - len(row_indices) / (num_blocks * num_blocks),
        }

        for row_idx, col_idx in zip(row_indices, col_indices):
            # Extract block queries and keys/values
            q_start = row_idx * block_size
            q_end = (row_idx + 1) * block_size
            k_start = col_idx * block_size
            k_end = (col_idx + 1) * block_size

            q_block = q[:, q_start:q_end]
            k_block = k[:, k_start:k_end]
            v_block = v[:, k_start:k_end]

            # Apply causal mask if needed
            causal_mask = None
            if is_causal and row_idx == col_idx:
                # Only need causal mask for diagonal blocks
                causal_mask = torch.ones(
                    block_size, block_size, dtype=torch.bool, device=q.device
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
            elif is_causal and row_idx < col_idx:
                # Skip blocks above diagonal in causal mode
                continue

            # Apply dilated attention within this block pair
            block_output = self._apply_dilated_attention_to_block(
                q_block, k_block, v_block, causal_mask
            )

            # Accumulate to output
            output[:, q_start:q_end] += block_output

            if return_attention_weights:
                attention_info["block_pairs"].append((row_idx.item(), col_idx.item()))

        # Apply dropout if configured
        if self.dropout_layer is not None:
            output = self.dropout_layer(output)

        if return_attention_weights:
            return output, attention_info
        return output

    def _apply_dilated_attention_to_block(
        self,
        q_block: Tensor,
        k_block: Tensor,
        v_block: Tensor,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply dilated attention within a single block pair.

        Args:
            q_block: Query block [batch, block_size, num_heads, head_dim]
            k_block: Key block [batch, block_size, num_heads, head_dim]
            v_block: Value block [batch, block_size, num_heads, head_dim]
            causal_mask: Optional causal mask for diagonal blocks

        Returns:
            Block output [batch, block_size, num_heads, head_dim]
        """
        batch_size, block_size, num_heads, head_dim = q_block.shape

        # For small blocks, just use regular attention
        min_segment_size = min(self.segment_lengths)
        if block_size <= min_segment_size:
            # Direct attention computation
            # Input is [batch, seq_len, heads, head_dim]
            # Convert to [batch, heads, seq_len, head_dim] for attention computation
            q_reshaped = q_block.transpose(1, 2)
            k_reshaped = k_block.transpose(1, 2)
            v_reshaped = v_block.transpose(1, 2)

            scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / math.sqrt(
                head_dim
            )

            if causal_mask is not None:
                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(1), float("-inf")
                )

            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout_layer is not None:
                attn_weights = self.dropout_layer(attn_weights)

            # Convert back to [batch, seq_len, heads, head_dim]
            output = torch.matmul(attn_weights, v_reshaped)
            return output.transpose(1, 2)

        # For larger blocks, apply dilated attention
        # We need to adjust segment lengths to fit within the block
        adjusted_segments = []
        adjusted_dilations = []

        remaining_size = block_size
        for seg_len, dil_rate in zip(self.segment_lengths, self.dilation_rates):
            if remaining_size <= 0:
                break

            # Adjust segment to fit in remaining space
            actual_seg_len = min(seg_len, remaining_size)
            if actual_seg_len > 0:
                adjusted_segments.append(actual_seg_len)
                adjusted_dilations.append(dil_rate)
                remaining_size -= actual_seg_len

        # If we have remaining space, add it to the last segment
        if remaining_size > 0 and adjusted_segments:
            adjusted_segments[-1] += remaining_size

        # Create temporary dilated attention with adjusted segments
        temp_dilated = DilatedAttention(
            segment_lengths=adjusted_segments,
            dilation_rates=adjusted_dilations,
            attention_dropout=0.0,  # Dropout already handled
            op=None if not self.use_xformers else None,  # Let it auto-detect
        )

        # DilatedAttention expects [batch, heads, seq_len, head_dim]
        # Convert from [batch, seq_len, heads, head_dim]
        q_reshaped = q_block.transpose(1, 2)
        k_reshaped = k_block.transpose(1, 2)
        v_reshaped = v_block.transpose(1, 2)

        # Apply dilated attention
        output = temp_dilated(
            q_reshaped, k_reshaped, v_reshaped, is_causal=(causal_mask is not None)
        )

        # Convert back to [batch, seq_len, heads, head_dim]
        return output.transpose(1, 2)

    def extra_repr(self) -> str:
        """String representation of module."""
        return (
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"sparse_pattern={self.sparse_config.pattern_type}, "
            f"sparsity_ratio={self.sparse_config.sparsity_ratio}, "
            f"block_size={self.sparse_config.block_size}, "
            f"dropout={self.dropout}"
        )
