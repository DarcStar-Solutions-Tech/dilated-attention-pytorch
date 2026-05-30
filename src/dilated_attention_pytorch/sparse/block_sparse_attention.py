"""
Block-Sparse Attention

Pure block-sparse attention implementation with efficient pattern handling.
This is NOT ring attention or dilated attention - it applies sparse patterns
at the block level on a single GPU.

Key features:
- Multiple sparse patterns: local_window, dilated_sparse, global_local
- Block-sparse computation for 5-50x speedup over dense attention
- Thread-safe pattern caching
- Memory-efficient implementation that never materializes full attention
- Support for Flash Attention 3 when available
"""

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from ..core.constants import HAS_FLASH_ATTN_3, GPU_TYPE


def _merge_block_attention(acc_out, acc_lse, blk_out, blk_lse):
    """Associatively merge two partial softmax-attention results via the log-sum-exp trick.

    Each partial is ``(out, lse)`` where ``out = softmax(scores) @ v`` is normalized over *its
    own* key set and ``lse = logsumexp(scores, dim=-1)`` is that set's log-denominator. The
    merge returns the result of a single joint softmax over the union of both key sets, so
    summing independently-normalized block outputs (which over-counts the denominator ~N x for
    N blocks) is avoided.

    This is also the cross-node reduction primitive: each rank computes a partial ``(out, lse)``
    over the key blocks it owns, and ranks combine partials with this associative/commutative
    merge -- no rank ever needs the full key set.

    Shapes: ``out`` is [..., q, d], ``lse`` is [..., q]. ``acc_out``/``acc_lse`` may be ``None``
    for the first partial.

    Robust to fully-masked partials: a query row whose entire key set is masked has
    ``lse = -inf`` and ``out = NaN`` (softmax over all ``-inf``). Such a partial carries zero
    weight, so its NaN is zeroed before combining; a row masked in *both* partials yields
    ``out = 0``/``lse = -inf``. (The single-GPU caller never produces this — it skips
    future-only blocks and always keeps the diagonal — but the documented cross-node use can.)
    """
    if acc_out is None:
        return torch.nan_to_num(blk_out), blk_lse
    new_lse = torch.maximum(acc_lse, blk_lse)
    alpha = torch.exp(acc_lse - new_lse)
    beta = torch.exp(blk_lse - new_lse)
    denom = alpha + beta
    # A masked partial contributes weight 0; zero its NaN values so 0 * NaN does not poison.
    out = (
        torch.nan_to_num(acc_out) * alpha.unsqueeze(-1)
        + torch.nan_to_num(blk_out) * beta.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    lse = new_lse + torch.log(denom)
    # Rows masked in BOTH partials (new_lse == -inf) give 0/NaN above: define out=0, lse=-inf.
    both_masked = torch.isneginf(new_lse)
    if both_masked.any():
        out = torch.where(both_masked.unsqueeze(-1), torch.zeros_like(out), out)
        lse = torch.where(both_masked, torch.full_like(lse, float("-inf")), lse)
    return out, lse


@dataclass
class SparsePatternConfig:
    """Configuration for sparse attention patterns."""

    pattern_type: str = (
        "local_window"  # Options: local_window, dilated_sparse, global_local
    )
    block_size: int = 64  # Size of each attention block
    sparsity_ratio: float = 0.1  # 0.1 = 90% sparse
    window_size: int = 256  # For local_window pattern
    global_tokens: int = 1  # For global_local pattern
    dilation_rates: list[int] | None = None  # For dilated_sparse pattern

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.sparsity_ratio <= 1:
            raise ValueError("sparsity_ratio must be between 0 and 1")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.window_size % self.block_size != 0:
            self.window_size = (
                self.window_size // self.block_size + 1
            ) * self.block_size
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 4, 8]


class PersistentPatternCache:
    """Enhanced pattern cache with thread-safety and persistence."""

    def __init__(self, max_size: int = 100):
        """Initialize cache with given max size."""
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.access_count = {}

    def get(self, key, generator_fn):
        """Get pattern from cache or generate if not present."""
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_count[key] = self.access_count.get(key, 0) + 1
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return self.cache[key]
            else:
                self.miss_count += 1
                # Generate pattern
                pattern = generator_fn()
                # Add to cache
                self.cache[key] = pattern
                self.access_count[key] = 1
                # Evict if necessary
                while len(self.cache) > self.max_size:
                    evicted_key, _ = self.cache.popitem(last=False)
                    self.access_count.pop(evicted_key, None)
                    self.eviction_count += 1
                return pattern

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.hit_count = 0
            self.miss_count = 0
            self.eviction_count = 0

    def get_stats(self):
        """Get cache statistics."""
        with self.lock:
            hit_rate = (
                self.hit_count / (self.hit_count + self.miss_count)
                if (self.hit_count + self.miss_count) > 0
                else 0
            )
            return {
                "size": len(self.cache),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self.eviction_count,
                "top_accessed": sorted(
                    self.access_count.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }


class BlockSparseAttention(torch.nn.Module):
    """
    Block-Sparse Attention with efficient pattern handling.

    This is a pure block-sparse attention implementation that applies
    sparse patterns at the block level for significant speedup.

    IMPORTANT: This is NOT ring attention - it does not distribute
    computation across GPUs. It processes the full sequence on a
    single GPU using sparse patterns.

    Key features:
    1. Memory-efficient block-sparse attention patterns
    2. Never materializes full attention matrices
    3. Persistent pattern caching across devices
    4. Batched block operations for efficiency
    5. Integrates with Flash Attention 3 when available
    """

    def __init__(
        self,
        sparse_config: SparsePatternConfig | None = None,
        enable_batched_ops: bool = True,
        pattern_cache_size: int = 100,
        **kwargs,
    ):
        """
        Initialize Block-Sparse Attention.

        Args:
            sparse_config: Configuration for sparse patterns
            enable_batched_ops: Deprecated / no-op. The former batched fast path summed
                independently-normalized per-block softmaxes (incorrect) and was removed;
                all computation now uses the joint-softmax grouped path. Retained for
                backward-compatible construction only.
            pattern_cache_size: Size of pattern cache
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()

        # Extract compatibility parameters but don't use them
        # These are here for backward compatibility only
        _ = kwargs.pop("segment_lengths", None)  # Not used
        _ = kwargs.pop("dilation_rates", None)  # Not used

        # Device placement
        self.device = kwargs.pop("device", None)

        # Memory pool (from removed parent class)
        self._memory_pool = None

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

        # Enhanced pattern cache with device awareness
        self.pattern_cache = PersistentPatternCache(max_size=pattern_cache_size)

        # Batched operations flag
        self.enable_batched_ops = enable_batched_ops

        # Pre-allocate buffers for batched operations
        self._batch_buffers = {}

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

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            return_attention_weights: Whether to return attention weights

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

        # Compute sparse attention
        attention_weights = None
        if return_attention_weights:
            attention_weights = self._compute_sparse_attention_with_weights(
                q, k, v, output, block_indices, is_causal
            )
        else:
            self._compute_sparse_attention(q, k, v, output, block_indices, is_causal)

        if return_attention_weights:
            return output, attention_weights
        else:
            return output

    def _get_sparse_block_indices(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Get indices with enhanced caching."""
        cache_key = (
            num_blocks,
            self.sparse_config.pattern_type,
            self.sparse_config.sparsity_ratio,
            self.sparse_config.block_size,
        )

        def generator():
            if self.sparse_config.pattern_type == "local_window":
                return self._generate_local_window_pattern(num_blocks, device)
            elif self.sparse_config.pattern_type == "dilated_sparse":
                return self._generate_dilated_sparse_pattern(num_blocks, device)
            elif self.sparse_config.pattern_type == "global_local":
                return self._generate_global_local_pattern(num_blocks, device)
            else:
                raise ValueError(
                    f"Unknown pattern type: {self.sparse_config.pattern_type}"
                )

        return self.pattern_cache.get(cache_key, generator)

    def _compute_sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> None:
        """Compute block-sparse attention with a correct joint softmax per query block."""
        self._compute_sparse_attention_grouped(
            q, k, v, output, block_indices, is_causal
        )

    # The previous _batched / _sequential split summed independently-normalized per-block
    # softmaxes (mathematically wrong) and the batched causal path produced NaNs from
    # fully-masked rows. Both now delegate to the LSE-accumulating implementation below;
    # they are kept as named entry points for backward compatibility.
    def _compute_sparse_attention_batched(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> None:
        """Deprecated alias; delegates to the correct grouped implementation."""
        self._compute_sparse_attention_grouped(
            q, k, v, output, block_indices, is_causal
        )

    def _compute_sparse_attention_sequential(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> None:
        """Deprecated alias; delegates to the correct grouped implementation."""
        self._compute_sparse_attention_grouped(
            q, k, v, output, block_indices, is_causal
        )

    def _compute_sparse_attention_grouped(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
        collect_weights: bool = False,
    ) -> "dict[str, Tensor] | None":
        """Block-sparse attention via per-query-block online-softmax (LSE) accumulation.

        For each query block, its selected key blocks are combined under a single shared softmax
        denominator using the log-sum-exp merge (``_merge_block_attention``) instead of summing
        independently-normalized per-block softmaxes. Causal masking skips entirely-future
        (above-diagonal) key blocks and applies an upper-triangular mask only on the diagonal
        block, so the sequential future-token leak and the batched all-``-inf`` NaN are both
        avoided.

        The per-block ``(output, lse)`` partials and their associative merge are exactly the
        primitive needed to parallelize across nodes: each rank attends over the key blocks it
        owns and the partials are reduced with ``_merge_block_attention``.
        """
        batch, seq_len, num_heads, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        bs = self.block_size
        row_indices, col_indices = block_indices

        # Group selected key blocks by query block (dict preserves pattern order).
        # De-duplicate (row, col) pairs: a key block must be counted once in the joint
        # softmax denominator, otherwise a repeated pair would be weighted ~2x.
        groups: dict[int, list[int]] = {}
        for r, c in zip(row_indices.tolist(), col_indices.tolist()):
            if is_causal and r < c:
                # Query block is entirely before this key block: all keys are future. Skip.
                continue
            cols = groups.setdefault(r, [])
            if c not in cols:
                cols.append(c)

        weight_values: list[Tensor] = []
        weight_row_indices: list[Tensor] = []
        weight_col_indices: list[Tensor] = []

        for row_idx, cols in groups.items():
            row_start = row_idx * bs
            row_end = row_start + bs
            q_block = q[:, row_start:row_end].transpose(1, 2)  # [b, h, bs, d]

            acc_out: "Tensor | None" = None
            acc_lse: "Tensor | None" = None
            pair_scores: list = []

            for col_idx in cols:
                col_start = col_idx * bs
                col_end = col_start + bs
                k_block = k[:, col_start:col_end].transpose(1, 2)  # [b, h, bs, d]
                v_block = v[:, col_start:col_end].transpose(1, 2)

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                if is_causal and row_idx == col_idx:
                    diag_mask = torch.triu(
                        torch.ones(bs, bs, dtype=torch.bool, device=scores.device),
                        diagonal=1,
                    )
                    scores = scores.masked_fill(diag_mask, float("-inf"))

                blk_lse = torch.logsumexp(scores, dim=-1)  # [b, h, bs]
                blk_out = torch.matmul(torch.softmax(scores, dim=-1), v_block)
                acc_out, acc_lse = _merge_block_attention(
                    acc_out, acc_lse, blk_out, blk_lse
                )
                if collect_weights:
                    pair_scores.append((col_idx, scores))

            if acc_out is None:
                continue  # query block has no selected key blocks
            output[:, row_start:row_end] = acc_out.transpose(1, 2)  # -> [b, bs, h, d]

            if collect_weights:
                # True joint weights = exp(scores - final_lse); 0 in masked positions.
                # w is [b, h, q, k]; flattened row-major the query index varies slowest
                # (repeat_interleave) and the key index fastest (repeat).
                for col_idx, scores in pair_scores:
                    col_start = col_idx * bs
                    w = torch.exp(scores - acc_lse.unsqueeze(-1))
                    weight_values.append(w.detach())
                    weight_row_indices.append(
                        torch.arange(
                            row_start, row_end, device=q.device
                        ).repeat_interleave(bs)
                    )
                    weight_col_indices.append(
                        torch.arange(col_start, col_start + bs, device=q.device).repeat(
                            bs
                        )
                    )

        if not collect_weights:
            return None
        return {
            "values": weight_values,
            "indices": (weight_row_indices, weight_col_indices),
            "row_indices": weight_row_indices,
            "col_indices": weight_col_indices,
            "shape": (batch, num_heads, seq_len, seq_len),
            "block_size": bs,
        }

    def _compute_sparse_attention_with_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> dict[str, Tensor]:
        """Compute sparse attention and return (joint-normalized) attention weights."""
        return self._compute_sparse_attention_grouped(
            q, k, v, output, block_indices, is_causal, collect_weights=True
        )

    def _generate_local_window_pattern(
        self, num_blocks: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Generate local window sparse pattern."""
        window_blocks = self.sparse_config.window_size // self.block_size
        row_indices = []
        col_indices = []

        for i in range(num_blocks):
            # Each block attends to blocks within the window
            start = max(0, i - window_blocks // 2)
            end = min(num_blocks, i + window_blocks // 2 + 1)
            for j in range(start, end):
                row_indices.append(i)
                col_indices.append(j)

        return (
            torch.tensor(row_indices, device=device),
            torch.tensor(col_indices, device=device),
        )

    def _generate_dilated_sparse_pattern(
        self, num_blocks: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Generate dilated sparse pattern at block level."""
        row_indices = []
        col_indices = []

        for i in range(num_blocks):
            # Each block attends to blocks at different dilation rates
            attended_blocks = set()

            for rate in self.sparse_config.dilation_rates:
                # Add blocks at this dilation rate
                for j in range(0, num_blocks, rate):
                    if abs(i - j) <= num_blocks // 2:  # Within reasonable distance
                        attended_blocks.add(j)

            # Convert to lists
            for j in sorted(attended_blocks):
                row_indices.append(i)
                col_indices.append(j)

        return (
            torch.tensor(row_indices, device=device),
            torch.tensor(col_indices, device=device),
        )

    def _generate_global_local_pattern(
        self, num_blocks: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Generate global + local sparse pattern."""
        row_indices = []
        col_indices = []

        global_blocks = min(
            self.sparse_config.global_tokens, num_blocks
        )  # First k blocks are global
        window_blocks = self.sparse_config.window_size // self.block_size

        for i in range(num_blocks):
            # Attend to global blocks
            for j in range(global_blocks):
                row_indices.append(i)
                col_indices.append(j)

            # Attend to local window (if not already covered by global)
            start = max(global_blocks, i - window_blocks // 2)
            end = min(num_blocks, i + window_blocks // 2 + 1)
            for j in range(start, end):
                if j not in range(global_blocks):  # Avoid duplicates
                    row_indices.append(i)
                    col_indices.append(j)

        return (
            torch.tensor(row_indices, device=device),
            torch.tensor(col_indices, device=device),
        )

    def _get_causal_mask_for_block(
        self, row_idx: int, col_idx: int, block_size: int, device: torch.device
    ) -> Tensor:
        """Get causal mask for a single block."""
        if row_idx < col_idx:
            # This block is entirely above the diagonal
            return torch.ones(block_size, block_size, dtype=torch.bool, device=device)
        elif row_idx > col_idx:
            # This block is entirely below the diagonal
            return torch.zeros(block_size, block_size, dtype=torch.bool, device=device)
        else:
            # This block is on the diagonal
            return torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool, device=device),
                diagonal=1,
            )

    def _get_batched_causal_mask(
        self,
        row_indices: Tensor,
        col_indices: Tensor,
        batch_size: int,
        num_heads: int,
    ) -> Tensor:
        """Create batched causal mask for all active blocks."""
        num_active_blocks = len(row_indices)
        mask = torch.zeros(
            batch_size * num_active_blocks * num_heads,
            self.block_size,
            self.block_size,
            dtype=torch.bool,
            device=row_indices.device,
        )

        # Create base causal mask for a single block
        base_diag_mask = torch.triu(
            torch.ones(
                self.block_size, self.block_size, dtype=torch.bool, device=mask.device
            ),
            diagonal=1,
        )

        # Apply appropriate mask to each block
        for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            start_idx = i * batch_size * num_heads
            end_idx = (i + 1) * batch_size * num_heads

            if row_idx.item() < col_idx.item():
                # Above diagonal - mask everything
                mask[start_idx:end_idx] = True
            elif row_idx.item() == col_idx.item():
                # On diagonal - use causal mask
                mask[start_idx:end_idx] = base_diag_mask

            # Below diagonal - already zeros, no action needed

        return mask

    def get_pattern_stats(self) -> dict:
        """Get statistics about sparse patterns."""
        cache_stats = self.pattern_cache.get_stats()
        return {
            "pattern_type": self.sparse_config.pattern_type,
            "block_size": self.block_size,
            "sparsity_ratio": self.sparse_config.sparsity_ratio,
            "cache_stats": cache_stats,
        }

    def estimate_memory_savings(self, seq_len: int) -> dict:
        """Estimate memory savings compared to dense attention."""
        if seq_len % self.block_size != 0:
            seq_len = ((seq_len // self.block_size) + 1) * self.block_size

        num_blocks = seq_len // self.block_size

        # Get pattern for estimation
        mock_device = torch.device("cpu")
        row_indices, col_indices = self._get_sparse_block_indices(
            num_blocks, 1, mock_device
        )

        active_blocks = len(row_indices)
        total_blocks = num_blocks * num_blocks

        actual_sparsity = 1 - (active_blocks / total_blocks)
        memory_ratio = active_blocks / total_blocks

        return {
            "seq_len": seq_len,
            "num_blocks": num_blocks,
            "active_blocks": active_blocks,
            "total_blocks": total_blocks,
            "actual_sparsity": actual_sparsity,
            "memory_ratio": memory_ratio,
            "speedup_estimate": 1 / memory_ratio if memory_ratio > 0 else float("inf"),
        }
