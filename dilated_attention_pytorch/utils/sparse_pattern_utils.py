"""
Sparse Pattern Utilities for Block-Sparse Attention

This module provides utility classes and functions for creating, manipulating,
and optimizing sparse attention patterns. Includes pattern generators, analyzers,
and optimization tools for maximum performance and quality.

Key Features:
- Multiple pattern generation strategies
- Pattern quality analysis and optimization
- Hardware-specific pattern optimization
- Dynamic pattern adaptation
- Pattern compression and serialization
- Visualization and debugging tools
"""

import math
import pickle
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class PatternType(Enum):
    """Types of sparse attention patterns"""

    LOCAL_WINDOW = "local_window"
    DILATED_SPARSE = "dilated_sparse"
    GLOBAL_LOCAL = "global_local"
    STRIDED = "strided"
    RANDOM = "random"
    LEARNED = "learned"
    HIERARCHICAL = "hierarchical"
    CONTENT_ADAPTIVE = "content_adaptive"


@dataclass
class PatternQualityMetrics:
    """Metrics for evaluating sparse pattern quality"""

    coverage_ratio: float  # Fraction of important attention weights covered
    locality_score: float  # How well pattern preserves local dependencies
    global_connectivity: float  # How well pattern preserves global connections
    efficiency_score: float  # Performance vs quality trade-off
    compression_ratio: float  # Memory/computation reduction
    approximation_error: float  # Error vs dense attention


@dataclass
class PatternConfig:
    """Configuration for sparse pattern generation"""

    pattern_type: PatternType = PatternType.DILATED_SPARSE
    sparsity_ratio: float = 0.25  # Fraction of connections to keep (density), not drop
    block_size: int = 128
    local_window_size: int = 512
    global_tokens: int = 64
    dilation_rates: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    stride: int = 4
    random_seed: int | None = None
    quality_threshold: float = 0.95
    enable_caching: bool = True
    enable_compression: bool = False


class SparsePatternGenerator:
    """
    Advanced sparse pattern generator with multiple strategies.

    Provides various sparse attention patterns optimized for different
    use cases and hardware configurations.
    """

    def __init__(self, config: PatternConfig):
        self.config = config
        self.pattern_cache: dict[tuple, torch.Tensor] = {}
        self.quality_cache: dict[tuple, PatternQualityMetrics] = {}
        self._cache_lock = threading.Lock()

        # Set random seed for reproducibility
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)

    def generate_pattern(
        self, seq_len: int, num_heads: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Generate sparse attention pattern based on configuration.

        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            device: Device to place pattern on

        Returns:
            Boolean tensor of shape [num_heads, num_blocks, num_blocks]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_blocks = seq_len // self.config.block_size
        cache_key = (
            seq_len,
            num_heads,
            self.config.pattern_type.value,
            self.config.sparsity_ratio,
            self.config.block_size,
        )

        # Check cache
        if self.config.enable_caching:
            with self._cache_lock:
                if cache_key in self.pattern_cache:
                    return self.pattern_cache[cache_key].to(device)

        # Generate pattern based on type
        if self.config.pattern_type == PatternType.LOCAL_WINDOW:
            pattern = self._generate_local_window_pattern(num_blocks, num_heads, device)
        elif self.config.pattern_type == PatternType.DILATED_SPARSE:
            pattern = self._generate_dilated_sparse_pattern(num_blocks, num_heads, device)
        elif self.config.pattern_type == PatternType.GLOBAL_LOCAL:
            pattern = self._generate_global_local_pattern(num_blocks, num_heads, device)
        elif self.config.pattern_type == PatternType.STRIDED:
            pattern = self._generate_strided_pattern(num_blocks, num_heads, device)
        elif self.config.pattern_type == PatternType.RANDOM:
            pattern = self._generate_random_pattern(num_blocks, num_heads, device)
        elif self.config.pattern_type == PatternType.HIERARCHICAL:
            pattern = self._generate_hierarchical_pattern(num_blocks, num_heads, device)
        else:
            raise ValueError(f"Unsupported pattern type: {self.config.pattern_type}")

        # Enforce target sparsity
        pattern = self._enforce_sparsity_ratio(pattern, self.config.sparsity_ratio)

        # Cache pattern
        if self.config.enable_caching:
            with self._cache_lock:
                self.pattern_cache[cache_key] = pattern.cpu()

        return pattern

    def _generate_local_window_pattern(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        """Generate local window attention pattern"""
        pattern = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)
        window_blocks = self.config.local_window_size // self.config.block_size

        for h in range(num_heads):
            for i in range(num_blocks):
                start = max(0, i - window_blocks // 2)
                end = min(num_blocks, i + window_blocks // 2 + 1)
                pattern[h, i, start:end] = True

        return pattern

    def _generate_dilated_sparse_pattern(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        """Generate dilated sparse attention pattern"""
        pattern = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)

        # Calculate how many connections to keep based on sparsity ratio
        total_connections = num_blocks * num_blocks
        target_connections = int(total_connections * self.config.sparsity_ratio)

        for h in range(num_heads):
            # Use different dilation rates for different heads
            dilation_idx = h % len(self.config.dilation_rates)
            dilation = self.config.dilation_rates[dilation_idx]

            # Create dilated pattern
            for i in range(num_blocks):
                # Always connect to self
                pattern[h, i, i] = True

                # Connect to positions at multiples of dilation
                for offset in range(
                    -target_connections // (2 * num_blocks),
                    target_connections // (2 * num_blocks) + 1,
                ):
                    j = i + offset * dilation
                    if 0 <= j < num_blocks:
                        pattern[h, i, j] = True

        return pattern

    def _generate_global_local_pattern(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        """Generate global + local attention pattern"""
        pattern = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)

        global_blocks = self.config.global_tokens // self.config.block_size
        local_window = self.config.local_window_size // self.config.block_size

        for h in range(num_heads):
            # Global tokens attend to everything
            pattern[h, :global_blocks, :] = True
            pattern[h, :, :global_blocks] = True

            # Local attention for remaining tokens
            for i in range(global_blocks, num_blocks):
                start = max(global_blocks, i - local_window // 2)
                end = min(num_blocks, i + local_window // 2 + 1)
                pattern[h, i, start:end] = True

        return pattern

    def _generate_strided_pattern(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        """Generate strided attention pattern"""
        pattern = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)

        for h in range(num_heads):
            for i in range(num_blocks):
                # Strided positions
                for j in range(0, num_blocks, self.config.stride):
                    pattern[h, i, j] = True

                # Local connections
                if i > 0:
                    pattern[h, i, i - 1] = True
                if i < num_blocks - 1:
                    pattern[h, i, i + 1] = True

        return pattern

    def _generate_random_pattern(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        """Generate random sparse attention pattern"""
        pattern = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)

        total_elements = num_blocks * num_blocks
        num_active = int(total_elements * self.config.sparsity_ratio)

        for h in range(num_heads):
            # Random active positions
            flat_indices = torch.randperm(total_elements)[:num_active]
            row_indices = flat_indices // num_blocks
            col_indices = flat_indices % num_blocks
            pattern[h, row_indices, col_indices] = True

        return pattern

    def _generate_hierarchical_pattern(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        """Generate hierarchical attention pattern"""
        pattern = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)

        # Combine multiple pattern types with different densities
        local_pattern = self._generate_local_window_pattern(num_blocks, num_heads, device)
        dilated_pattern = self._generate_dilated_sparse_pattern(num_blocks, num_heads, device)
        global_pattern = self._generate_global_local_pattern(num_blocks, num_heads, device)

        # Logical OR combination of patterns
        pattern = local_pattern | dilated_pattern | global_pattern

        return pattern

    def _enforce_sparsity_ratio(self, pattern: torch.Tensor, target_ratio: float) -> torch.Tensor:
        """Enforce target sparsity ratio"""
        current_ratio = pattern.float().mean()

        if abs(current_ratio - target_ratio) < 0.01:
            return pattern  # Close enough

        if current_ratio > target_ratio:
            # Too dense, remove connections
            num_remove = int((current_ratio - target_ratio) * pattern.numel())
            active_indices = torch.nonzero(pattern, as_tuple=False)

            if len(active_indices) > num_remove:
                # Create a copy to avoid in-place modification issues
                pattern = pattern.clone()
                remove_indices = torch.randperm(len(active_indices))[:num_remove]
                for idx in remove_indices:
                    pattern[tuple(active_indices[idx])] = False

        else:
            # Too sparse, add connections
            num_add = int((target_ratio - current_ratio) * pattern.numel())
            inactive_indices = torch.nonzero(~pattern, as_tuple=False)

            if len(inactive_indices) > num_add:
                # Create a copy to avoid in-place modification issues
                pattern = pattern.clone()
                add_indices = torch.randperm(len(inactive_indices))[:num_add]
                for idx in add_indices:
                    pattern[tuple(inactive_indices[idx])] = True

        return pattern


class PatternQualityAnalyzer:
    """
    Analyzer for evaluating sparse attention pattern quality.

    Provides metrics and analysis tools for understanding how well
    sparse patterns approximate dense attention.
    """

    def __init__(self):
        self.analysis_cache: dict[str, Any] = {}

    def analyze_pattern_quality(
        self,
        sparse_pattern: torch.Tensor,
        reference_attention: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
    ) -> PatternQualityMetrics:
        """
        Analyze quality of sparse attention pattern.

        Args:
            sparse_pattern: Sparse pattern to analyze
            reference_attention: Reference dense attention weights (optional)
            q: Query tensor for computing reference (optional)
            k: Key tensor for computing reference (optional)

        Returns:
            Quality metrics for the pattern
        """
        # Compute reference attention if not provided
        if reference_attention is None and q is not None and k is not None:
            reference_attention = self._compute_reference_attention(q, k)

        # Calculate quality metrics
        coverage_ratio = self._calculate_coverage_ratio(sparse_pattern, reference_attention)
        locality_score = self._calculate_locality_score(sparse_pattern)
        global_connectivity = self._calculate_global_connectivity(sparse_pattern)
        efficiency_score = self._calculate_efficiency_score(sparse_pattern, coverage_ratio)
        compression_ratio = 1.0 - sparse_pattern.float().mean().item()
        approximation_error = self._calculate_approximation_error(
            sparse_pattern, reference_attention
        )

        return PatternQualityMetrics(
            coverage_ratio=coverage_ratio,
            locality_score=locality_score,
            global_connectivity=global_connectivity,
            efficiency_score=efficiency_score,
            compression_ratio=compression_ratio,
            approximation_error=approximation_error,
        )

    def _compute_reference_attention(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute reference dense attention weights"""
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        return torch.softmax(scores, dim=-1)

    def _calculate_coverage_ratio(
        self, sparse_pattern: torch.Tensor, reference_attention: torch.Tensor | None
    ) -> float:
        """Calculate how well sparse pattern covers important attention weights"""
        if reference_attention is None:
            return 1.0  # Cannot calculate without reference

        # Find top-k important attention weights
        k = int(0.1 * reference_attention.numel())  # Top 10%
        top_k_mask = torch.zeros_like(reference_attention, dtype=torch.bool)

        flat_ref = reference_attention.flatten()
        _, top_indices = torch.topk(flat_ref, k)
        top_k_mask.view(-1)[top_indices] = True

        # Calculate overlap with sparse pattern
        if sparse_pattern.shape != reference_attention.shape:
            # Handle block-level pattern vs token-level attention
            sparse_pattern = self._expand_block_pattern_to_tokens(
                sparse_pattern, reference_attention.shape
            )

        overlap = (sparse_pattern & top_k_mask).sum().float()
        total_important = top_k_mask.sum().float()

        return (overlap / total_important).item() if total_important > 0 else 0.0

    def _calculate_locality_score(self, sparse_pattern: torch.Tensor) -> float:
        """Calculate how well pattern preserves local dependencies"""
        num_heads, num_blocks, _ = sparse_pattern.shape

        # Count connections within local windows
        local_window = min(8, num_blocks // 4)  # Reasonable local window
        local_connections = 0
        total_connections = sparse_pattern.sum().item()

        for h in range(num_heads):
            for i in range(num_blocks):
                start = max(0, i - local_window)
                end = min(num_blocks, i + local_window + 1)
                local_connections += sparse_pattern[h, i, start:end].sum().item()

        return local_connections / total_connections if total_connections > 0 else 0.0

    def _calculate_global_connectivity(self, sparse_pattern: torch.Tensor) -> float:
        """Calculate global connectivity of pattern"""
        num_heads, num_blocks, _ = sparse_pattern.shape

        # Check if each block can reach all other blocks (transitively)
        connectivity_matrix = sparse_pattern.float()

        # Floyd-Warshall for reachability
        for k in range(num_blocks):
            for i in range(num_blocks):
                for j in range(num_blocks):
                    connectivity_matrix[:, i, j] = torch.max(
                        connectivity_matrix[:, i, j],
                        torch.min(connectivity_matrix[:, i, k], connectivity_matrix[:, k, j]),
                    )

        # Calculate average reachability
        total_reachable = connectivity_matrix.sum()
        total_possible = num_heads * num_blocks * num_blocks

        return (total_reachable / total_possible).item()

    def _calculate_efficiency_score(
        self, sparse_pattern: torch.Tensor, coverage_ratio: float
    ) -> float:
        """Calculate efficiency as quality per computation cost"""
        computation_ratio = sparse_pattern.float().mean().item()
        return coverage_ratio / computation_ratio if computation_ratio > 0 else 0.0

    def _calculate_approximation_error(
        self, sparse_pattern: torch.Tensor, reference_attention: torch.Tensor | None
    ) -> float:
        """Calculate approximation error vs reference attention"""
        if reference_attention is None:
            return 0.0  # Cannot calculate without reference

        # Expand pattern to token level if needed
        if sparse_pattern.shape != reference_attention.shape:
            sparse_pattern = self._expand_block_pattern_to_tokens(
                sparse_pattern, reference_attention.shape
            )

        # Calculate Frobenius norm error
        masked_attention = reference_attention * sparse_pattern.float()
        error = torch.norm(reference_attention - masked_attention, p="fro")
        norm = torch.norm(reference_attention, p="fro")

        return (error / norm).item() if norm > 0 else 0.0

    def _expand_block_pattern_to_tokens(
        self, block_pattern: torch.Tensor, target_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Expand block-level pattern to token-level pattern"""
        num_heads, num_blocks, _ = block_pattern.shape
        batch, heads, seq_len, _ = target_shape

        block_size = seq_len // num_blocks

        # Expand each block to tokens
        expanded = torch.zeros(
            batch,
            heads,
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=block_pattern.device,
        )

        for h in range(min(heads, num_heads)):
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if block_pattern[h, i, j]:
                        i_start, i_end = i * block_size, (i + 1) * block_size
                        j_start, j_end = j * block_size, (j + 1) * block_size
                        expanded[:, h, i_start:i_end, j_start:j_end] = True

        return expanded


class PatternOptimizer:
    """
    Optimizer for improving sparse attention patterns.

    Uses various optimization techniques to improve pattern quality
    while maintaining target sparsity levels.
    """

    def __init__(self, quality_threshold: float = 0.95):
        self.quality_threshold = quality_threshold
        self.analyzer = PatternQualityAnalyzer()

    def optimize_pattern(
        self,
        initial_pattern: torch.Tensor,
        reference_attention: torch.Tensor | None = None,
        max_iterations: int = 10,
    ) -> torch.Tensor:
        """
        Optimize sparse pattern for better quality.

        Args:
            initial_pattern: Initial sparse pattern
            reference_attention: Reference attention for optimization
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized sparse pattern
        """
        current_pattern = initial_pattern.clone()

        for iteration in range(max_iterations):
            # Analyze current quality
            metrics = self.analyzer.analyze_pattern_quality(current_pattern, reference_attention)

            if metrics.efficiency_score >= self.quality_threshold:
                break  # Good enough

            # Apply optimization step
            current_pattern = self._optimization_step(current_pattern, reference_attention, metrics)

        return current_pattern

    def _optimization_step(
        self,
        pattern: torch.Tensor,
        reference_attention: torch.Tensor | None,
        metrics: PatternQualityMetrics,
    ) -> torch.Tensor:
        """Single optimization step"""
        optimized_pattern = pattern.clone()

        # Strategy 1: Improve coverage by adding important connections
        if metrics.coverage_ratio < 0.9 and reference_attention is not None:
            optimized_pattern = self._improve_coverage(optimized_pattern, reference_attention)

        # Strategy 2: Improve locality by adding local connections
        if metrics.locality_score < 0.8:
            optimized_pattern = self._improve_locality(optimized_pattern)

        # Strategy 3: Improve global connectivity
        if metrics.global_connectivity < 0.5:
            optimized_pattern = self._improve_global_connectivity(optimized_pattern)

        return optimized_pattern

    def _improve_coverage(
        self, pattern: torch.Tensor, reference_attention: torch.Tensor
    ) -> torch.Tensor:
        """Improve coverage of important attention weights"""
        # Find important attention weights not covered by pattern
        k = int(0.05 * reference_attention.numel())  # Top 5%
        flat_ref = reference_attention.flatten()
        _, top_indices = torch.topk(flat_ref, k)

        # Add some of these to pattern
        pattern_flat = pattern.view(-1)
        for idx in top_indices[: k // 2]:  # Add half of them
            pattern_flat[idx] = True

        return pattern.view_as(pattern)

    def _improve_locality(self, pattern: torch.Tensor) -> torch.Tensor:
        """Improve local connectivity"""
        num_heads, num_blocks, _ = pattern.shape

        for h in range(num_heads):
            for i in range(num_blocks):
                # Add connections to immediate neighbors
                if i > 0:
                    pattern[h, i, i - 1] = True
                if i < num_blocks - 1:
                    pattern[h, i, i + 1] = True

        return pattern

    def _improve_global_connectivity(self, pattern: torch.Tensor) -> torch.Tensor:
        """Improve global connectivity"""
        num_heads, num_blocks, _ = pattern.shape

        # Add some random long-range connections
        num_global = max(1, num_blocks // 8)

        for h in range(num_heads):
            for i in range(num_blocks):
                # Add a few random long-range connections
                long_range_indices = torch.randperm(num_blocks)[:2]
                pattern[h, i, long_range_indices] = True

        return pattern


class PatternVisualizer:
    """
    Visualization tools for sparse attention patterns.

    Provides various visualization methods for understanding
    and debugging sparse attention patterns.
    """

    def __init__(self):
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available. Visualization features disabled.")

    def visualize_pattern(
        self,
        pattern: torch.Tensor,
        title: str = "Sparse Attention Pattern",
        save_path: str | None = None,
        show: bool = True,
    ) -> Any | None:
        """
        Visualize sparse attention pattern as heatmap.

        Args:
            pattern: Sparse pattern to visualize
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure

        Returns:
            Matplotlib figure if matplotlib available
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for visualization")
            return None

        # Take first head if multi-head
        if pattern.dim() == 3:
            pattern_2d = pattern[0]
        else:
            pattern_2d = pattern

        # Convert to numpy
        if HAS_NUMPY:
            pattern_np = pattern_2d.cpu().numpy().astype(float)
        else:
            pattern_np = pattern_2d.cpu().float()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(pattern_np, cmap="Blues", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key Blocks")
        ax.set_ylabel("Query Blocks")

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add sparsity info
        sparsity = pattern.float().mean().item()
        ax.text(
            0.02,
            0.98,
            f"Sparsity: {sparsity:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def compare_patterns(
        self,
        patterns: dict[str, torch.Tensor],
        save_path: str | None = None,
        show: bool = True,
    ) -> Any | None:
        """
        Compare multiple sparse patterns side by side.

        Args:
            patterns: Dictionary of pattern name -> pattern tensor
            save_path: Path to save figure
            show: Whether to display figure

        Returns:
            Matplotlib figure if matplotlib available
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for visualization")
            return None

        num_patterns = len(patterns)
        fig, axes = plt.subplots(1, num_patterns, figsize=(5 * num_patterns, 5))

        if num_patterns == 1:
            axes = [axes]

        for i, (name, pattern) in enumerate(patterns.items()):
            # Take first head if multi-head
            if pattern.dim() == 3:
                pattern_2d = pattern[0]
            else:
                pattern_2d = pattern

            # Convert to numpy
            if HAS_NUMPY:
                pattern_np = pattern_2d.cpu().numpy().astype(float)
            else:
                pattern_np = pattern_2d.cpu().float()

            # Plot
            im = axes[i].imshow(pattern_np, cmap="Blues", aspect="auto")
            axes[i].set_title(f"{name}\nSparsity: {pattern.float().mean():.3f}")
            axes[i].set_xlabel("Key Blocks")
            if i == 0:
                axes[i].set_ylabel("Query Blocks")

            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig


# Utility functions
def save_sparse_pattern(pattern: torch.Tensor, filepath: str, compress: bool = True):
    """Save sparse pattern to file"""
    pattern_data = {
        "pattern": pattern.cpu(),
        "shape": pattern.shape,
        "sparsity": pattern.float().mean().item(),
        "dtype": str(pattern.dtype),
    }

    if compress:
        import gzip

        with gzip.open(filepath + ".gz", "wb") as f:
            pickle.dump(pattern_data, f)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(pattern_data, f)


def load_sparse_pattern(filepath: str) -> torch.Tensor:
    """Load sparse pattern from file"""
    try:
        # Try compressed format first
        import gzip

        with gzip.open(filepath + ".gz", "rb") as f:
            pattern_data = pickle.load(f)
    except:
        # Fall back to uncompressed
        with open(filepath, "rb") as f:
            pattern_data = pickle.load(f)

    return pattern_data["pattern"]


def analyze_pattern_statistics(pattern: torch.Tensor) -> dict[str, float]:
    """Calculate comprehensive statistics for sparse pattern"""
    stats = {}

    # Basic statistics
    stats["sparsity_ratio"] = pattern.float().mean().item()
    stats["density_ratio"] = 1.0 - stats["sparsity_ratio"]
    stats["total_elements"] = pattern.numel()
    stats["active_elements"] = pattern.sum().item()

    # Pattern properties
    if pattern.dim() >= 2:
        # Row/column statistics
        row_sums = pattern.sum(dim=-1).float()
        col_sums = pattern.sum(dim=-2).float()

        stats["avg_row_density"] = row_sums.mean().item()
        stats["avg_col_density"] = col_sums.mean().item()
        stats["row_density_std"] = row_sums.std().item()
        stats["col_density_std"] = col_sums.std().item()

        # Diagonal statistics (for square matrices)
        if pattern.shape[-1] == pattern.shape[-2]:
            diag = torch.diagonal(pattern, dim1=-2, dim2=-1)
            stats["diagonal_density"] = diag.float().mean().item()

    return stats


def optimize_pattern_for_hardware(
    pattern: torch.Tensor, hardware: str = "auto", block_size: int = 16
) -> torch.Tensor:
    """
    Optimize sparse pattern for specific hardware architectures.

    Args:
        pattern: Input sparse pattern
        hardware: Target hardware ("auto", "a100", "h100", "v100", "cpu")
        block_size: Block size for block-sparse optimizations

    Returns:
        Optimized sparse pattern
    """
    if hardware == "auto":
        # Detect hardware automatically
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            if "A100" in device_name:
                hardware = "a100"
            elif "H100" in device_name:
                hardware = "h100"
            elif "V100" in device_name:
                hardware = "v100"
            else:
                hardware = "gpu"
        else:
            hardware = "cpu"

    # Apply hardware-specific optimizations
    if hardware in ["a100", "h100"]:
        # Align to tensor core block sizes (16x16)
        pattern = _align_to_blocks(pattern, block_size=16)
    elif hardware == "v100":
        # V100 prefers 8x8 blocks
        pattern = _align_to_blocks(pattern, block_size=8)
    elif hardware == "cpu":
        # CPU prefers larger blocks for cache efficiency
        pattern = _align_to_blocks(pattern, block_size=32)
    else:
        # Default block alignment
        pattern = _align_to_blocks(pattern, block_size=block_size)

    return pattern


def _align_to_blocks(pattern: torch.Tensor, block_size: int) -> torch.Tensor:
    """Align sparse pattern to block boundaries for hardware efficiency."""
    # Ensure pattern dimensions are compatible with block size
    h, w = pattern.shape[-2:]

    # Pad if necessary
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    if pad_h > 0 or pad_w > 0:
        pattern = F.pad(pattern, (0, pad_w, 0, pad_h), value=False)

    # Convert to block representation
    blocks = pattern.view(
        *pattern.shape[:-2],
        pattern.shape[-2] // block_size,
        block_size,
        pattern.shape[-1] // block_size,
        block_size,
    )

    # If any element in a block is True, make the whole block True
    block_pattern = blocks.any(dim=-1).any(dim=-2)

    # Expand back to full pattern
    optimized = block_pattern.unsqueeze(-1).unsqueeze(-1)
    optimized = optimized.repeat(1, 1, block_size, 1, block_size)
    optimized = optimized.view(*pattern.shape)

    # Remove padding if we added any
    if pad_h > 0 or pad_w > 0:
        optimized = optimized[..., :h, :w]

    return optimized


# Export main classes and functions
__all__ = [
    "PatternConfig",
    "PatternOptimizer",
    "PatternQualityAnalyzer",
    "PatternQualityMetrics",
    "PatternType",
    "PatternVisualizer",
    "SparsePatternGenerator",
    "load_pattern",
    "pattern_statistics",
    "save_pattern",
]
