"""
Hilbert-Optimized Attention V2: Apply Hilbert ordering at the data level.

This implementation applies Hilbert space-filling curve ordering to the sequence
itself, maintaining this ordering throughout the computation for better cache locality.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict

from .utils.hilbert_curve import generate_hilbert_indices


class HilbertSequencePreprocessor(nn.Module):
    """
    Preprocessor that reorders sequences using Hilbert space-filling curves.

    This module reorders the sequence dimension to improve spatial locality.
    The Hilbert ordering is maintained throughout the model, and only reversed
    at the very end.
    """

    def __init__(self, cache_mappings: bool = True):
        super().__init__()
        self.cache_mappings = cache_mappings
        if cache_mappings:
            self._forward_mapping_cache: Dict[int, Tensor] = {}
            self._inverse_mapping_cache: Dict[int, Tensor] = {}

    def _compute_hilbert_mapping(
        self, seq_len: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Compute Hilbert mapping for a given sequence length."""
        # Check cache first
        if self.cache_mappings and seq_len in self._forward_mapping_cache:
            forward_map = self._forward_mapping_cache[seq_len].to(device)
            inverse_map = self._inverse_mapping_cache[seq_len].to(device)
            return forward_map, inverse_map

        # Find suitable 2D dimensions for Hilbert curve
        # We want the smallest square that can contain seq_len elements
        grid_size = int(math.ceil(math.sqrt(seq_len)))

        # Make it a power of 2 for standard Hilbert curve
        grid_size_pow2 = 1
        while grid_size_pow2 < grid_size:
            grid_size_pow2 *= 2

        # Generate Hilbert indices
        n_levels = int(math.log2(grid_size_pow2))
        hilbert_indices = generate_hilbert_indices(n_levels)

        # Only keep indices that are within seq_len
        valid_indices = [idx for idx in hilbert_indices if idx < seq_len]

        # If we don't have enough indices, fill remaining sequentially
        if len(valid_indices) < seq_len:
            remaining = set(range(seq_len)) - set(valid_indices)
            valid_indices.extend(sorted(remaining))

        # Create forward mapping (original position -> Hilbert position)
        forward_mapping = torch.tensor(
            valid_indices[:seq_len], dtype=torch.long, device=device
        )

        # Create inverse mapping (Hilbert position -> original position)
        inverse_mapping = torch.zeros(seq_len, dtype=torch.long, device=device)
        for i, idx in enumerate(forward_mapping):
            inverse_mapping[idx] = i

        # Cache if enabled
        if self.cache_mappings:
            self._forward_mapping_cache[seq_len] = forward_mapping.cpu()
            self._inverse_mapping_cache[seq_len] = inverse_mapping.cpu()

        return forward_mapping, inverse_mapping

    def reorder_to_hilbert(self, tensor: Tensor) -> Tensor:
        """Reorder tensor from standard to Hilbert ordering."""
        batch_size, seq_len = tensor.shape[:2]

        # Get Hilbert mapping
        forward_mapping, _ = self._compute_hilbert_mapping(seq_len, tensor.device)

        # Apply reordering along sequence dimension
        # We use index_select for efficiency
        return torch.index_select(tensor, dim=1, index=forward_mapping)

    def reorder_from_hilbert(self, tensor: Tensor) -> Tensor:
        """Reorder tensor from Hilbert back to standard ordering."""
        batch_size, seq_len = tensor.shape[:2]

        # Get Hilbert mapping
        _, inverse_mapping = self._compute_hilbert_mapping(seq_len, tensor.device)

        # Apply inverse reordering
        return torch.index_select(tensor, dim=1, index=inverse_mapping)

    def get_hilbert_aware_mask(
        self, seq_len: int, device: torch.device, is_causal: bool = True
    ) -> Optional[Tensor]:
        """Get attention mask that respects Hilbert ordering."""
        if not is_causal:
            return None

        # Get mapping
        forward_mapping, _ = self._compute_hilbert_mapping(seq_len, device)

        # Create causal mask in original space
        original_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        # Reorder mask to Hilbert space
        # Mask[i,j] = 1 means position i cannot attend to position j
        # In Hilbert space: HilbertMask[h_i, h_j] = OriginalMask[orig_i, orig_j]
        hilbert_mask = original_mask[forward_mapping][:, forward_mapping]

        return hilbert_mask


class BlockSparseHilbertAttentionV2(nn.Module):
    """
    Block-sparse attention designed to work with Hilbert-ordered sequences.

    This implementation assumes the input is already in Hilbert order and
    designs sparse patterns that work well with this ordering.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        sparsity_ratio: float = 0.05,
        use_hilbert_pattern: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        self.use_hilbert_pattern = use_hilbert_pattern
        self.scale = self.head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Cache for Hilbert-aware patterns
        self._pattern_cache: Dict[Tuple[int, bool], Tuple[Tensor, Tensor]] = {}

    def _create_hilbert_aware_block_pattern(
        self, seq_len: int, is_causal: bool, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Create block sparse pattern optimized for Hilbert-ordered sequences."""

        cache_key = (seq_len, is_causal)
        if cache_key in self._pattern_cache:
            row_idx, col_idx = self._pattern_cache[cache_key]
            return row_idx.to(device), col_idx.to(device)

        num_blocks = seq_len // self.block_size
        total_blocks = num_blocks * num_blocks

        # Calculate number of blocks to keep
        keep_blocks = int(total_blocks * (1 - self.sparsity_ratio))

        if self.use_hilbert_pattern:
            # Hilbert-aware pattern: keep blocks that are close in Hilbert space
            # This means keeping blocks along the Hilbert curve path

            block_indices = []

            # 1. Keep diagonal blocks (important for local dependencies)
            for i in range(num_blocks):
                block_indices.append((i, i))

            # 2. Keep blocks within Hilbert distance
            # Since data is Hilbert-ordered, nearby positions in sequence
            # are nearby in 2D space, so we keep blocks that maintain this locality

            # Create a "band" around the diagonal in Hilbert space
            band_width = max(1, int(math.sqrt(keep_blocks - num_blocks) / 2))

            for i in range(num_blocks):
                for offset in range(1, band_width + 1):
                    # Forward direction
                    if i + offset < num_blocks:
                        block_indices.append((i, i + offset))
                        if not is_causal:
                            block_indices.append((i + offset, i))

                    # Backward direction (if not causal)
                    if not is_causal and i - offset >= 0:
                        block_indices.append((i, i - offset))

            # 3. Add some long-range connections following Hilbert curve
            # These help capture dependencies between distant but related regions
            skip_sizes = [2, 4, 8, 16]
            for skip in skip_sizes:
                if len(block_indices) >= keep_blocks:
                    break
                for i in range(0, num_blocks - skip, skip):
                    if len(block_indices) < keep_blocks:
                        block_indices.append((i, i + skip))
                        if not is_causal and len(block_indices) < keep_blocks:
                            block_indices.append((i + skip, i))

            # Remove duplicates and respect causality
            block_indices = list(set(block_indices))
            if is_causal:
                block_indices = [(r, c) for r, c in block_indices if r >= c]

            # Trim to desired number of blocks
            block_indices = block_indices[:keep_blocks]

        else:
            # Standard block pattern (for comparison)
            block_indices = []
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if not is_causal or i >= j:
                        if len(block_indices) < keep_blocks:
                            # Simple distance-based sparsity
                            if abs(i - j) <= band_width:
                                block_indices.append((i, j))

        # Convert to tensors
        if block_indices:
            row_indices = torch.tensor([r for r, c in block_indices], dtype=torch.long)
            col_indices = torch.tensor([c for r, c in block_indices], dtype=torch.long)
        else:
            row_indices = torch.tensor([], dtype=torch.long)
            col_indices = torch.tensor([], dtype=torch.long)

        # Cache the pattern
        self._pattern_cache[cache_key] = (row_indices.cpu(), col_indices.cpu())

        return row_indices.to(device), col_indices.to(device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass assuming input is in Hilbert order.

        Args:
            query, key, value: [batch, seq_len, embed_dim] in Hilbert order
            attn_mask: Optional attention mask respecting Hilbert order
            is_causal: Whether to use causal masking

        Returns:
            output: [batch, seq_len, embed_dim] in Hilbert order
        """
        batch_size, seq_len, _ = query.shape

        # Ensure sequence length is divisible by block size
        assert seq_len % self.block_size == 0, (
            f"Sequence length {seq_len} must be divisible by block size {self.block_size}"
        )

        # Project Q, K, V
        q = (
            self.q_proj(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Get block pattern
        row_indices, col_indices = self._create_hilbert_aware_block_pattern(
            seq_len, is_causal, query.device
        )

        # Initialize output
        output = torch.zeros_like(q)

        # Compute attention for each block pair
        _ = seq_len // self.block_size

        for idx in range(len(row_indices)):
            row_idx = row_indices[idx].item()
            col_idx = col_indices[idx].item()

            # Extract blocks
            q_block = q[
                :, :, row_idx * self.block_size : (row_idx + 1) * self.block_size
            ]
            k_block = k[
                :, :, col_idx * self.block_size : (col_idx + 1) * self.block_size
            ]
            v_block = v[
                :, :, col_idx * self.block_size : (col_idx + 1) * self.block_size
            ]

            # Compute attention scores
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale

            # Apply mask if needed
            if attn_mask is not None:
                block_mask = attn_mask[
                    row_idx * self.block_size : (row_idx + 1) * self.block_size,
                    col_idx * self.block_size : (col_idx + 1) * self.block_size,
                ]
                scores = scores.masked_fill(
                    block_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Apply causal mask for diagonal blocks if needed
            if is_causal and row_idx == col_idx:
                causal_mask = torch.triu(
                    torch.ones(self.block_size, self.block_size, device=scores.device),
                    diagonal=1,
                ).bool()
                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Compute attention weights
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v_block)

            # Accumulate in output
            output[
                :, :, row_idx * self.block_size : (row_idx + 1) * self.block_size
            ] += attn_output

        # Reshape and project output
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.out_proj(output)

        return output


class HilbertTransformerLayer(nn.Module):
    """
    Transformer layer that maintains Hilbert ordering throughout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        block_size: int = 64,
        sparsity_ratio: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Attention with Hilbert-aware patterns
        self.attention = BlockSparseHilbertAttentionV2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            sparsity_ratio=sparsity_ratio,
            dropout=dropout,
        )

        # Standard transformer components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass maintaining Hilbert order."""
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class HilbertTransformer(nn.Module):
    """
    Full transformer that uses Hilbert ordering throughout all layers.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        block_size: int = 64,
        sparsity_ratio: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.preprocessor = HilbertSequencePreprocessor(cache_mappings=True)

        self.layers = nn.ModuleList(
            [
                HilbertTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    block_size=block_size,
                    sparsity_ratio=sparsity_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        is_causal: bool = False,
        return_hilbert_ordered: bool = False,
    ) -> Tensor:
        """
        Forward pass with automatic Hilbert ordering.

        Args:
            x: Input tensor [batch, seq_len, embed_dim] in standard order
            is_causal: Whether to use causal masking
            return_hilbert_ordered: If True, return output in Hilbert order

        Returns:
            output: [batch, seq_len, embed_dim] in standard order (unless return_hilbert_ordered=True)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Convert to Hilbert order
        x = self.preprocessor.reorder_to_hilbert(x)

        # Get Hilbert-aware mask if causal
        attn_mask = self.preprocessor.get_hilbert_aware_mask(
            seq_len, x.device, is_causal=is_causal
        )

        # Process through layers (all in Hilbert order)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, is_causal=is_causal)

        x = self.norm(x)

        # Convert back to standard order unless requested otherwise
        if not return_hilbert_ordered:
            x = self.preprocessor.reorder_from_hilbert(x)

        return x
