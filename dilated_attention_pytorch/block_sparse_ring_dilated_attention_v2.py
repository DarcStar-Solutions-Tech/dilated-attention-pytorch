"""
Block-Sparse Ring Dilated Attention V2 - Memory Efficient Implementation

This implementation truly leverages sparsity for memory efficiency by:
1. Never materializing full attention matrices
2. Processing only active blocks
3. Using sparse tensor representations where possible
4. Returning sparse attention weights in COO format
"""

import math
import threading
from dataclasses import dataclass

import torch
from torch import Tensor

from .ring_dilated_attention import RingDilatedAttention


@dataclass
class SparsePatternConfig:
    """Configuration for sparse attention patterns"""

    pattern_type: str = "dilated_sparse"  # 'local_window', 'dilated_sparse', 'global_local'
    sparsity_ratio: float = 0.1  # Fraction of blocks to compute (0.1 = 90% sparse)
    block_size: int = 128  # Tokens per block
    local_window_size: int = 512  # For local window patterns
    global_tokens: int = 64  # For global+local patterns


class BlockSparseRingDilatedAttentionV2(RingDilatedAttention):
    """
    Memory-efficient block-sparse ring dilated attention.

    Key improvements:
    - Never materializes full attention matrices
    - Returns sparse attention weights in COO format
    - Processes only active blocks
    - Minimal memory overhead
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        sparse_config: SparsePatternConfig | None = None,
        **kwargs,
    ):
        """Initialize block-sparse ring dilated attention."""
        # Extract sparse_config from kwargs if passed there (for compatibility)
        if "sparse_config" in kwargs:
            sparse_config = kwargs.pop("sparse_config")
        if "sparsity_config" in kwargs:
            sparse_config = kwargs.pop("sparsity_config")

        # Filter out any remaining BlockSparse-specific parameters
        block_sparse_params = {
            "use_adaptive_sparsity",
            "quality_threshold",
            "enable_memory_pool",
            "enable_packed_comm",
            "enable_hardware_opt",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in block_sparse_params}

        super().__init__(segment_lengths, dilation_rates, **filtered_kwargs)

        self.sparse_config = sparse_config or SparsePatternConfig()
        self.block_size = self.sparse_config.block_size

        # Pattern cache - stores only block indices, not full matrices
        self.pattern_cache: dict[tuple, tuple[Tensor, Tensor]] = {}
        self._cache_lock = threading.Lock()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor] | None]:
        """
        Forward pass with memory-efficient block-sparse attention.

        Returns:
            output: Attention output [batch, seq_len, num_heads, head_dim]
            attention_weights: If requested, returns dict with:
                - 'indices': COO format indices of non-zero blocks
                - 'values': Attention values for those blocks
                - 'shape': Full attention shape for reference
        """
        batch, seq_len, num_heads, head_dim = q.shape

        # Ensure sequence length is divisible by block size
        if seq_len % self.block_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by block size {self.block_size}"
            )

        num_blocks = seq_len // self.block_size

        # Get active block pairs for sparse pattern
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, q.device)

        # Initialize output
        output = torch.zeros_like(q)

        # Process attention in blocks
        if return_attention_weights:
            attention_data = self._compute_sparse_attention_with_weights(
                q, k, v, output, block_indices, is_causal
            )
        else:
            self._compute_sparse_attention(q, k, v, output, block_indices, is_causal)
            attention_data = None

        return (output, attention_data) if return_attention_weights else output

    def _get_sparse_block_indices(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Get indices of active blocks in sparse pattern."""
        _ = num_heads  # Reserved for future head-specific patterns
        cache_key = (
            num_blocks,
            self.sparse_config.pattern_type,
            self.sparse_config.sparsity_ratio,
            self.sparse_config.block_size,
        )

        with self._cache_lock:
            if cache_key in self.pattern_cache:
                row_idx, col_idx = self.pattern_cache[cache_key]
                return row_idx.to(device), col_idx.to(device)

        # Generate pattern based on type
        if self.sparse_config.pattern_type == "local_window":
            row_idx, col_idx = self._create_local_window_indices(num_blocks)
        elif self.sparse_config.pattern_type == "dilated_sparse":
            row_idx, col_idx = self._create_dilated_sparse_indices(num_blocks)
        elif self.sparse_config.pattern_type == "global_local":
            row_idx, col_idx = self._create_global_local_indices(num_blocks)
        else:
            raise ValueError(f"Unknown pattern type: {self.sparse_config.pattern_type}")

        # Convert to tensors
        row_idx = torch.tensor(row_idx, dtype=torch.long)
        col_idx = torch.tensor(col_idx, dtype=torch.long)

        # Cache the indices (on CPU to save GPU memory)
        with self._cache_lock:
            self.pattern_cache[cache_key] = (row_idx.cpu(), col_idx.cpu())

        return row_idx.to(device), col_idx.to(device)

    def _create_local_window_indices(self, num_blocks: int) -> tuple[list, list]:
        """Create indices for local window pattern."""
        window_blocks = self.sparse_config.local_window_size // self.block_size
        row_indices = []
        col_indices = []

        for i in range(num_blocks):
            start = max(0, i - window_blocks // 2)
            end = min(num_blocks, i + window_blocks // 2 + 1)
            for j in range(start, end):
                row_indices.append(i)
                col_indices.append(j)

        return row_indices, col_indices

    def _create_dilated_sparse_indices(self, num_blocks: int) -> tuple[list, list]:
        """Create indices for dilated sparse pattern."""
        row_indices = []
        col_indices = []

        # Use dilation rates from parent class
        for i in range(num_blocks):
            # Always attend to self
            row_indices.append(i)
            col_indices.append(i)

            # Add dilated connections
            for rate in self.dilation_rates[:3]:  # Use first 3 dilation rates
                for direction in [-1, 1]:
                    j = i + direction * rate
                    if 0 <= j < num_blocks:
                        row_indices.append(i)
                        col_indices.append(j)

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for r, c in zip(row_indices, col_indices, strict=False):
            if (r, c) not in seen:
                seen.add((r, c))
                unique_pairs.append((r, c))

        if unique_pairs:
            row_indices, col_indices = zip(*unique_pairs, strict=False)
        else:
            row_indices, col_indices = [], []

        return list(row_indices), list(col_indices)

    def _create_global_local_indices(self, num_blocks: int) -> tuple[list, list]:
        """Create indices for global + local pattern."""
        row_indices = []
        col_indices = []

        global_blocks = min(self.sparse_config.global_tokens // self.block_size, num_blocks)
        local_radius = (self.sparse_config.local_window_size // self.block_size) // 2

        for i in range(num_blocks):
            # Global attention to first few blocks
            for j in range(global_blocks):
                row_indices.append(i)
                col_indices.append(j)

            # Local window attention
            start = max(global_blocks, i - local_radius)
            end = min(num_blocks, i + local_radius + 1)
            for j in range(start, end):
                if j not in range(global_blocks):  # Avoid duplicates
                    row_indices.append(i)
                    col_indices.append(j)

        return row_indices, col_indices

    def _compute_sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: tuple[Tensor, Tensor],
        is_causal: bool,
    ):
        """Compute sparse attention without storing weights."""
        batch, seq_len, num_heads, head_dim = q.shape
        row_idx, col_idx = block_indices
        num_active_blocks = len(row_idx)

        if num_active_blocks == 0:
            return

        # Reshape to blocks
        q_blocks = q.view(batch, -1, self.block_size, num_heads, head_dim)
        k_blocks = k.view(batch, -1, self.block_size, num_heads, head_dim)
        v_blocks = v.view(batch, -1, self.block_size, num_heads, head_dim)
        output_blocks = output.view(batch, -1, self.block_size, num_heads, head_dim)

        # Process each active block pair
        scale = 1.0 / math.sqrt(head_dim)

        for idx in range(num_active_blocks):
            q_idx = row_idx[idx]
            k_idx = col_idx[idx]

            # Extract blocks
            q_block = q_blocks[:, q_idx]  # [batch, block_size, num_heads, head_dim]
            k_block = k_blocks[:, k_idx]
            v_block = v_blocks[:, k_idx]

            # Compute attention for this block
            # Reshape for batch matrix multiply
            q_block = q_block.transpose(1, 2)  # [batch, num_heads, block_size, head_dim]
            k_block = k_block.transpose(1, 2)
            v_block = v_block.transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # Apply causal mask if needed
            if is_causal:
                # Only apply mask if this is a causal block (q_idx >= k_idx)
                if q_idx > k_idx:
                    # Full block is allowed
                    pass
                elif q_idx == k_idx:
                    # Diagonal block needs causal mask
                    causal_mask = torch.triu(
                        torch.ones(self.block_size, self.block_size, device=scores.device),
                        diagonal=1,
                    ).bool()
                    scores.masked_fill_(causal_mask, float("-inf"))
                else:
                    # Skip future blocks entirely
                    continue

            # Softmax and attention
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_block)

            # Accumulate to output (back to original shape)
            attn_output = attn_output.transpose(1, 2)  # [batch, block_size, num_heads, head_dim]
            output_blocks[:, q_idx] += attn_output

    def _compute_sparse_attention_with_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> dict[str, Tensor]:
        """Compute sparse attention and return weights in sparse format."""
        batch, seq_len, num_heads, head_dim = q.shape
        row_idx, col_idx = block_indices

        # Prepare sparse storage for attention weights
        # We'll store block indices and their attention values
        weight_indices = []
        weight_values = []

        # Reshape to blocks
        q_blocks = q.view(batch, -1, self.block_size, num_heads, head_dim)
        k_blocks = k.view(batch, -1, self.block_size, num_heads, head_dim)
        v_blocks = v.view(batch, -1, self.block_size, num_heads, head_dim)
        output_blocks = output.view(batch, -1, self.block_size, num_heads, head_dim)

        scale = 1.0 / math.sqrt(head_dim)

        for idx in range(len(row_idx)):
            q_idx = row_idx[idx].item()
            k_idx = col_idx[idx].item()

            # Skip future blocks for causal attention
            if is_causal and q_idx < k_idx:
                continue

            # Extract blocks
            q_block = q_blocks[:, q_idx].transpose(1, 2)
            k_block = k_blocks[:, k_idx].transpose(1, 2)
            v_block = v_blocks[:, k_idx].transpose(1, 2)

            # Compute attention
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # Apply causal mask if needed
            if is_causal and q_idx == k_idx:
                causal_mask = torch.triu(
                    torch.ones(self.block_size, self.block_size, device=scores.device), diagonal=1
                ).bool()
                scores.masked_fill_(causal_mask, float("-inf"))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # Store sparse weights (only the block, not full matrix)
            weight_indices.append((q_idx, k_idx))
            weight_values.append(attn_weights.cpu())  # Store on CPU to save GPU memory

            # Compute output
            attn_output = torch.matmul(attn_weights, v_block)
            output_blocks[:, q_idx] += attn_output.transpose(1, 2)

        # Return sparse representation
        return {
            "block_indices": weight_indices,
            "block_values": weight_values,
            "block_size": self.block_size,
            "shape": (batch, num_heads, seq_len, seq_len),
            "num_blocks": (seq_len // self.block_size, seq_len // self.block_size),
        }
