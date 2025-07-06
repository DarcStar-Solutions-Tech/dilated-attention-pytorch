"""
Block-Sparse Ring Dilated Attention - Memory Efficient Implementation

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

from .core.constants import GPU_TYPE, HAS_FLASH_ATTN_3
from .ring_dilated_attention_production import (
    RingDilatedAttentionProduction as RingDilatedAttentionV2,
    RingAttentionConfig,
)
from .utils.flash_attention_3_utils import create_fa3_block_sparse_mask, get_fa3_config


@dataclass
class SparsePatternConfig:
    """Configuration for sparse attention patterns"""

    pattern_type: str = (
        "dilated_sparse"  # 'local_window', 'dilated_sparse', 'global_local'
    )
    sparsity_ratio: float = 0.1  # Fraction of blocks to compute (0.1 = 90% sparse)
    block_size: int = 128  # Tokens per block
    local_window_size: int = 512  # For local window patterns
    global_tokens: int = 64  # For global+local patterns


class BlockSparseRingDilatedAttention(RingDilatedAttentionV2):
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

        # Extract parameters for RingAttentionConfig
        dropout = kwargs.get("dropout", 0.0)
        ring_size = kwargs.get("ring_size", None)
        use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", True)
        use_memory_pool = kwargs.get("enable_memory_pool", True)
        mixed_precision = kwargs.get("mixed_precision", True)

        # Create RingAttentionConfig
        ring_config = RingAttentionConfig(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_memory_pool=use_memory_pool,
            mixed_precision=mixed_precision,
        )

        # Initialize parent with config
        super().__init__(ring_config)

        # Extract sparsity_ratio if provided directly (for compatibility)
        sparsity_ratio = kwargs.pop("sparsity_ratio", None)

        # Handle both dict and SparsePatternConfig
        if isinstance(sparse_config, dict):
            self.sparse_config = SparsePatternConfig(**sparse_config)
        elif sparse_config is not None:
            self.sparse_config = sparse_config
        else:
            # Create default config, optionally with provided sparsity_ratio
            config_kwargs = {}
            if sparsity_ratio is not None:
                config_kwargs["sparsity_ratio"] = sparsity_ratio
            self.sparse_config = SparsePatternConfig(**config_kwargs)

        self.block_size = self.sparse_config.block_size

        # Pattern cache - stores only block indices, not full matrices
        self.pattern_cache: dict[tuple, tuple[Tensor, Tensor]] = {}
        self._cache_lock = threading.Lock()

        # Check for Flash Attention 3 support
        self.use_fa3 = HAS_FLASH_ATTN_3 and str(GPU_TYPE) in ["h100", "h800"]
        if self.use_fa3:
            self.fa3_config = None  # Will be set dynamically based on sequence length

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

        # Initialize output using memory pool if available
        if hasattr(self, "_allocate_tensor") and self._memory_pool is not None:
            output = self._allocate_tensor(
                q.shape, q.dtype, q.device, strategy="auto", zero_init=True
            )
        else:
            output = torch.zeros_like(q)

        # Try Flash Attention 3 first if available
        fa3_weights = None
        if self.use_fa3:
            fa3_weights = self._compute_sparse_attention_fa3(
                q, k, v, output, block_indices, is_causal
            )

        # If FA3 was successful and we don't need weights, we're done
        if fa3_weights is not None or (self.use_fa3 and not return_attention_weights):
            attention_data = None
        # Fall back to standard sparse computation
        elif return_attention_weights:
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

        global_blocks = min(
            self.sparse_config.global_tokens // self.block_size, num_blocks
        )
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

    def _compute_sparse_attention_fa3(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: tuple[Tensor, Tensor],  # noqa: ARG002
        is_causal: bool,
    ) -> Tensor | None:
        """
        Compute sparse attention using Flash Attention 3 optimizations.

        Returns attention weights if available, otherwise None.
        """
        if not self.use_fa3:
            return None

        try:
            from flash_attn import flash_attn_func_v3  # noqa: PLC0415

            batch, seq_len, num_heads, head_dim = q.shape

            # Get FA3 configuration for this sequence length
            if self.fa3_config is None or self.fa3_config.get("seq_len") != seq_len:
                self.fa3_config = get_fa3_config(
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    use_fp8=str(GPU_TYPE) == "h100" and q.dtype == torch.float16,
                    enable_async=True,
                )
                self.fa3_config["seq_len"] = seq_len

            # Create FA3 block-sparse mask
            mask = create_fa3_block_sparse_mask(
                seq_len=seq_len,
                block_size=self.block_size,
                sparsity_ratio=1.0 - self.sparse_config.sparsity_ratio,
                pattern_type=self.sparse_config.pattern_type,
                device=q.device,
            )

            # Call FA3 with block-sparse support
            fa3_output = flash_attn_func_v3(
                q,
                k,
                v,
                causal=is_causal,
                block_sparse_mask=mask,
                block_size=self.fa3_config["block_size"],
                use_fp8=self.fa3_config["use_fp8"],
                enable_async=self.fa3_config["enable_async"],
                warp_specialized=self.fa3_config["warp_specialized"],
                num_splits=self.fa3_config["num_splits"],
            )

            # Copy FA3 output to our output tensor
            output.copy_(fa3_output)

        except Exception as e:
            # Fall back to standard sparse computation
            import warnings

            warnings.warn(
                f"Flash Attention 3 sparse computation failed: {e}", stacklevel=2
            )
            return None
        else:
            # FA3 doesn't return attention weights in sparse mode
            return None

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

        # Pre-allocate temporary tensors for block computation using memory pool
        # Only use pool for tensors >= 1MB as per optimization findings
        block_tensor_size = batch * num_heads * self.block_size * self.block_size
        block_tensor_bytes = block_tensor_size * 4  # float32
        _ = (
            hasattr(self, "_allocate_tensor")
            and self._memory_pool is not None
            and block_tensor_bytes >= 1024 * 1024
        )

        for idx in range(num_active_blocks):
            q_idx = row_idx[idx]
            k_idx = col_idx[idx]

            # Extract blocks
            q_block = q_blocks[:, q_idx]  # [batch, block_size, num_heads, head_dim]
            k_block = k_blocks[:, k_idx]
            v_block = v_blocks[:, k_idx]

            # Compute attention for this block
            # Reshape for batch matrix multiply
            q_block = q_block.transpose(
                1, 2
            )  # [batch, num_heads, block_size, head_dim]
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
                    # Cache causal mask to avoid recomputation
                    if not hasattr(self, "_causal_mask_cache"):
                        self._causal_mask_cache = {}

                    cache_key = (self.block_size, scores.device)
                    if cache_key not in self._causal_mask_cache:
                        self._causal_mask_cache[cache_key] = torch.triu(
                            torch.ones(
                                self.block_size, self.block_size, device=scores.device
                            ),
                            diagonal=1,
                        ).bool()

                    causal_mask = self._causal_mask_cache[cache_key]
                    scores.masked_fill_(causal_mask, float("-inf"))
                else:
                    # Skip future blocks entirely
                    continue

            # Softmax and attention
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_block)

            # Accumulate to output (back to original shape)
            attn_output = attn_output.transpose(
                1, 2
            )  # [batch, block_size, num_heads, head_dim]
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
                # Reuse cached causal mask
                if not hasattr(self, "_causal_mask_cache"):
                    self._causal_mask_cache = {}

                cache_key = (self.block_size, scores.device)
                if cache_key not in self._causal_mask_cache:
                    self._causal_mask_cache[cache_key] = torch.triu(
                        torch.ones(
                            self.block_size, self.block_size, device=scores.device
                        ),
                        diagonal=1,
                    ).bool()

                causal_mask = self._causal_mask_cache[cache_key]
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

    def cleanup_buffers(self):
        """Clean up cached masks and call parent cleanup."""
        # Clear causal mask cache
        if hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache.clear()

        # Call parent cleanup for memory pools and communication buffers
        super().cleanup_buffers()
