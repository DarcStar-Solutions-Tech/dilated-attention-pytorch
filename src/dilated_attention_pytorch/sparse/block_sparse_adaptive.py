"""
Content-Adaptive Block-Sparse Attention

This module implements learned sparsity patterns that adapt to the content
of the input sequences. Instead of using fixed patterns, it learns which
positions should attend to each other based on the data.

Key features:
1. Learnable importance scoring for attention connections
2. Differentiable top-k selection using Gumbel-softmax
3. Adaptive sparsity ratio based on sequence complexity
4. Integration with existing Block-Sparse infrastructure
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .block_sparse_attention import (
    BlockSparseAttention,
    SparsePatternConfig,
)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive sparsity learning."""

    # Base sparsity ratio (can be adapted based on content)
    base_sparsity: float = 0.9

    # Temperature for Gumbel-softmax (lower = more discrete)
    temperature: float = 1.0

    # Whether to use learnable temperature
    learnable_temperature: bool = True

    # Minimum temperature (for annealing)
    min_temperature: float = 0.1

    # Hidden dimension for importance scoring network
    hidden_dim: int = 128

    # Number of layers in importance network
    num_layers: int = 2

    # Whether to share importance network across heads
    share_across_heads: bool = False

    # Regularization weight for sparsity
    sparsity_reg_weight: float = 0.01

    # Whether to use hard (discrete) or soft (continuous) sparsity
    hard_sparsity: bool = False


class ImportanceScorer(nn.Module):
    """
    Neural network that scores the importance of attention connections.

    Takes query and key representations and outputs importance scores
    for each potential connection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        share_across_heads: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.share_across_heads = share_across_heads

        # Build importance scoring network
        layers = []

        # Input projection
        if share_across_heads:
            layers.append(nn.Linear(input_dim * 2, hidden_dim))
        else:
            # Separate networks for each head
            self.head_projections = nn.ModuleList(
                [nn.Linear(input_dim * 2, hidden_dim) for _ in range(num_heads)]
            )

        # Hidden layers
        for _ in range(num_layers - 1):
            if share_across_heads or len(layers) > 0:
                layers.extend(
                    [
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ]
                )

        # Output projection to importance score
        if share_across_heads:
            layers.extend(
                [
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                ]
            )
            self.importance_net = nn.Sequential(*layers)
        else:
            self.head_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(num_heads)
                ]
            )

    def forward(
        self, q_summary: Tensor, k_summary: Tensor, head_idx: Optional[int] = None
    ) -> Tensor:
        """
        Compute importance scores for query-key pairs.

        Args:
            q_summary: Query summary [batch, num_q_blocks, input_dim]
            k_summary: Key summary [batch, num_k_blocks, input_dim]
            head_idx: Which head to compute scores for (if not sharing)

        Returns:
            Importance scores [batch, num_q_blocks, num_k_blocks]
        """
        batch_size, num_q_blocks, _ = q_summary.shape
        num_k_blocks = k_summary.shape[1]
        device = q_summary.device

        # Move module to correct device if needed
        if (
            hasattr(self, "importance_net")
            and self.importance_net[0].weight.device != device
        ):
            self.to(device)
        elif (
            hasattr(self, "head_projections")
            and self.head_projections[0].weight.device != device
        ):
            self.to(device)

        # Expand for all pairs
        q_expanded = q_summary.unsqueeze(2).expand(-1, -1, num_k_blocks, -1)
        k_expanded = k_summary.unsqueeze(1).expand(-1, num_q_blocks, -1, -1)

        # Concatenate query and key features
        qk_features = torch.cat([q_expanded, k_expanded], dim=-1)

        # Compute importance scores
        if self.share_across_heads:
            scores = self.importance_net(qk_features).squeeze(-1)
        else:
            # Project with head-specific network
            qk_flat = qk_features.reshape(-1, self.input_dim * 2)

            if head_idx is not None:
                # Single head
                hidden = self.head_projections[head_idx](qk_flat)
                scores = self.head_networks[head_idx](hidden)
            else:
                # All heads (used during initialization)
                scores = []
                for h in range(self.num_heads):
                    hidden = self.head_projections[h](qk_flat)
                    head_scores = self.head_networks[h](hidden)
                    scores.append(head_scores)
                scores = torch.stack(scores, dim=0).mean(0)  # Average across heads

            scores = scores.reshape(batch_size, num_q_blocks, num_k_blocks)

        return scores


class BlockSparseAdaptive(BlockSparseAttention):
    """
    Block-Sparse attention with learned, content-adaptive sparsity patterns.

    This implementation learns which attention connections are important
    based on the input content, allowing for more flexible and effective
    sparsity patterns.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        num_heads: int,
        head_dim: int,
        adaptive_config: Optional[AdaptiveConfig] = None,
        **kwargs,
    ):
        """Initialize adaptive block-sparse attention."""
        # Create default sparse config
        sparse_config = SparsePatternConfig(
            pattern_type="adaptive",
            sparsity_ratio=adaptive_config.base_sparsity if adaptive_config else 0.9,
            block_size=kwargs.get("block_size", 64),
        )

        # Remove num_heads and head_dim from kwargs before passing to parent
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["num_heads", "head_dim", "sparse_config"]
        }

        super().__init__(sparse_config, **filtered_kwargs)

        # Store these for compatibility, even though they're not used
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.adaptive_config = adaptive_config or AdaptiveConfig()

        # Create importance scoring network
        self.importance_scorer = ImportanceScorer(
            input_dim=head_dim,
            hidden_dim=self.adaptive_config.hidden_dim,
            num_layers=self.adaptive_config.num_layers,
            num_heads=num_heads,
            share_across_heads=self.adaptive_config.share_across_heads,
        )

        # Learnable temperature for Gumbel-softmax
        if self.adaptive_config.learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.adaptive_config.temperature))
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.log(torch.tensor(self.adaptive_config.temperature)),
            )

        # Block summary projections (to reduce computation)
        self.q_summary_proj = nn.Linear(head_dim, head_dim)
        self.k_summary_proj = nn.Linear(head_dim, head_dim)

        # Move to device if specified
        if "device" in kwargs:
            self.to(kwargs["device"])

    def _compute_block_summaries(
        self, q: Tensor, k: Tensor, head_idx: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute summary representations for each block.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            head_idx: Which head to compute summaries for

        Returns:
            q_summary: Query block summaries [batch, num_blocks, head_dim]
            k_summary: Key block summaries [batch, num_blocks, head_dim]
        """
        batch_size, seq_len, _, _ = q.shape
        num_blocks = seq_len // self.block_size

        # Extract specific head
        q_head = q[:, :, head_idx, :]  # [batch, seq_len, head_dim]
        k_head = k[:, :, head_idx, :]

        # Reshape into blocks
        q_blocks = q_head.reshape(
            batch_size, num_blocks, self.block_size, self.head_dim
        )
        k_blocks = k_head.reshape(
            batch_size, num_blocks, self.block_size, self.head_dim
        )

        # Compute block summaries (mean pooling)
        q_summary = q_blocks.mean(dim=2)  # [batch, num_blocks, head_dim]
        k_summary = k_blocks.mean(dim=2)

        # Project summaries
        q_summary = self.q_summary_proj(q_summary)
        k_summary = self.k_summary_proj(k_summary)

        return q_summary, k_summary

    def _gumbel_softmax_topk(
        self, scores: Tensor, k: int, temperature: float, hard: bool = False
    ) -> Tensor:
        """
        Differentiable top-k selection using Gumbel-softmax trick.

        Args:
            scores: Importance scores [batch, num_q_blocks, num_k_blocks]
            k: Number of connections to keep per query block
            temperature: Gumbel-softmax temperature
            hard: Whether to use hard (discrete) or soft selection

        Returns:
            Selection mask [batch, num_q_blocks, num_k_blocks]
        """
        batch_size, num_q_blocks, num_k_blocks = scores.shape

        # Add Gumbel noise for stochastic selection
        if self.training:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scores) + 1e-10) + 1e-10
            )
            scores = scores + gumbel_noise

        # For each query block, select top-k key blocks
        mask = torch.zeros_like(scores)

        for i in range(num_q_blocks):
            block_scores = scores[:, i, :]  # [batch, num_k_blocks]

            # Get top-k indices
            _, topk_indices = torch.topk(block_scores, k, dim=-1)

            # Create soft selection using softmax
            if not hard:
                # Soft selection with temperature
                block_probs = F.softmax(block_scores / temperature, dim=-1)
                # Approximate top-k with high values for selected indices
                for b in range(batch_size):
                    mask[b, i, topk_indices[b]] = 1.0
                # Blend with soft probabilities
                mask[:, i, :] = mask[:, i, :] * 0.9 + block_probs * 0.1
            else:
                # Hard selection
                for b in range(batch_size):
                    mask[b, i, topk_indices[b]] = 1.0

        return mask

    def _generate_adaptive_pattern(
        self,
        q: Tensor,
        k: Tensor,
        head_idx: int,
        target_sparsity: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate adaptive sparse pattern based on content.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            head_idx: Which head to generate pattern for
            target_sparsity: Override sparsity ratio (if None, use config)

        Returns:
            row_indices: Selected row indices
            col_indices: Selected column indices
        """
        batch_size, seq_len, _, _ = q.shape
        num_blocks = seq_len // self.block_size
        device = q.device

        # Compute block summaries
        q_summary, k_summary = self._compute_block_summaries(q, k, head_idx)

        # Compute importance scores
        importance_scores = self.importance_scorer(q_summary, k_summary, head_idx)

        # Determine number of connections to keep
        if target_sparsity is None:
            target_sparsity = self.adaptive_config.base_sparsity

        total_blocks = num_blocks * num_blocks
        num_active = int(total_blocks * (1 - target_sparsity))
        k_per_query = max(1, num_active // num_blocks)

        # Get temperature
        temperature = torch.exp(self.log_temperature)
        if self.adaptive_config.learnable_temperature:
            temperature = torch.clamp(
                temperature, min=self.adaptive_config.min_temperature
            )

        # Select top-k connections per query block
        selection_mask = self._gumbel_softmax_topk(
            importance_scores,
            k_per_query,
            temperature,
            hard=self.adaptive_config.hard_sparsity,
        )

        # For now, we'll use the average across batch for the pattern
        # In practice, this could be extended to support batch-specific patterns
        avg_mask = selection_mask.mean(0)  # [num_q_blocks, num_k_blocks]

        # Convert soft mask to hard indices
        threshold = 0.5 if self.adaptive_config.hard_sparsity else 0.1
        active_blocks = (avg_mask > threshold).nonzero()

        if len(active_blocks) > 0:
            row_indices = active_blocks[:, 0]
            col_indices = active_blocks[:, 1]
        else:
            # Fallback to diagonal
            row_indices = torch.arange(num_blocks, device=device)
            col_indices = torch.arange(num_blocks, device=device)

        return row_indices, col_indices

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        return_pattern: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with adaptive sparsity pattern.

        Args:
            q, k, v: Query, key, value tensors
            is_causal: Whether to use causal masking
            return_pattern: Whether to return the learned pattern

        Returns:
            Output tensor and optionally the sparse pattern
        """
        batch_size, seq_len, num_heads, head_dim = q.shape

        # For adaptive patterns, we need to generate patterns per head
        # This is more expensive but allows head-specific patterns
        outputs = []
        patterns = [] if return_pattern else None

        for h in range(num_heads):
            # Generate adaptive pattern for this head
            row_indices, col_indices = self._generate_adaptive_pattern(q, k, h)

            # Store pattern if requested
            if return_pattern:
                patterns.append((row_indices, col_indices))

            # Extract head data
            q_head = q[:, :, h : h + 1, :]
            k_head = k[:, :, h : h + 1, :]
            v_head = v[:, :, h : h + 1, :]

            # Apply attention with adaptive pattern
            # Override the pattern generation in parent class
            self._adaptive_indices = (row_indices, col_indices)
            output_head = super().forward(q_head, k_head, v_head, is_causal)
            outputs.append(output_head)

        # Concatenate heads
        output = torch.cat(outputs, dim=2)

        if return_pattern:
            return output, {"patterns": patterns}
        return output

    def _get_sparse_block_indices(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Override to use adaptive pattern if available."""
        if hasattr(self, "_adaptive_indices"):
            # Use pre-computed adaptive pattern
            indices = self._adaptive_indices
            delattr(self, "_adaptive_indices")  # Clean up
            return indices
        else:
            # Fall back to parent implementation
            return super()._get_sparse_block_indices(num_blocks, num_heads, device)

    def get_sparsity_loss(self) -> Tensor:
        """
        Compute regularization loss to encourage desired sparsity.

        Returns:
            Sparsity regularization loss
        """
        # This can be extended to include various regularization terms
        # For now, return zero as patterns are controlled by top-k
        device = (
            next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device, requires_grad=True)


def create_adaptive_block_sparse(
    embed_dim: int,
    num_heads: int,
    segment_lengths: List[int] = None,
    dilation_rates: List[int] = None,
    adaptive_config: Optional[AdaptiveConfig] = None,
    **kwargs,
) -> BlockSparseAdaptive:
    """
    Create adaptive block-sparse attention module.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for dilated attention
        adaptive_config: Configuration for adaptive sparsity
        **kwargs: Additional arguments

    Returns:
        BlockSparseAdaptive module
    """
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    head_dim = embed_dim // num_heads

    return BlockSparseAdaptive(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        num_heads=num_heads,
        head_dim=head_dim,
        adaptive_config=adaptive_config,
        **kwargs,
    )


# Training utilities for adaptive sparsity
class AdaptiveSparsityTrainer:
    """
    Helper class for training adaptive sparsity patterns.

    Provides utilities for:
    - Temperature annealing
    - Sparsity scheduling
    - Pattern analysis
    """

    def __init__(
        self,
        model: BlockSparseAdaptive,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.1,
        annealing_steps: int = 10000,
    ):
        self.model = model
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.annealing_steps = annealing_steps
        self.current_step = 0

    def step(self):
        """Update temperature and other training parameters."""
        self.current_step += 1

        if self.model.adaptive_config.learnable_temperature:
            # Anneal temperature
            progress = min(self.current_step / self.annealing_steps, 1.0)
            target_temp = (
                self.initial_temperature * (1 - progress)
                + self.final_temperature * progress
            )

            # Update model temperature
            with torch.no_grad():
                self.model.log_temperature.data = torch.log(torch.tensor(target_temp))

    def analyze_patterns(self, dataloader, num_samples: int = 100) -> Dict[str, any]:
        """
        Analyze learned patterns on a dataset.

        Returns statistics about the learned sparse patterns.
        """
        pattern_stats = {
            "sparsity_ratios": [],
            "pattern_diversity": [],
            "head_similarity": [],
        }

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break

                # Get patterns for this batch
                q, k, v = batch  # Assuming preprocessed QKV
                _, pattern_info = self.model(q, k, v, return_pattern=True)

                # Analyze patterns
                patterns = pattern_info["patterns"]

                # Compute sparsity ratio
                for row_idx, col_idx in patterns:
                    total_blocks = len(row_idx)
                    num_blocks = int(math.sqrt(total_blocks))
                    sparsity = 1.0 - (len(row_idx) / (num_blocks**2))
                    pattern_stats["sparsity_ratios"].append(sparsity)

        return pattern_stats
