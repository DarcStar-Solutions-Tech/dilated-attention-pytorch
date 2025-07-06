"""
Sparse pattern generation for distributed attention.

This module contains pattern generation utilities for creating hierarchical
sparse attention patterns optimized for distributed training.
"""

import os
import threading
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .distributed_sparse_config import DistributedSparseConfig


class HierarchicalSparsePatternGenerator:
    """
    Sparse pattern generator for distributed systems.
    
    This class creates hierarchical sparse attention patterns that are optimized
    for multi-node distributed training. It generates different sparsity levels
    for local (within-node), global, and inter-node attention patterns.
    
    Features:
    - Three-level hierarchy: local, global, and inter-node patterns
    - Pattern caching for efficient reuse
    - Load balancing based on computation statistics
    - Thread-safe pattern generation
    
    The generator adapts patterns based on:
    - Node topology (GPUs per node)
    - Load imbalance across ranks
    - Memory and communication constraints
    """

    def __init__(self, config: DistributedSparseConfig, world_size: int, rank: int):
        """Initialize the pattern generator.
        
        Args:
            config: Distributed sparse attention configuration
            world_size: Total number of distributed processes
            rank: Current process rank
        """
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.node_size = self._detect_node_size()
        self.node_rank = rank // self.node_size
        self.local_rank = rank % self.node_size

        # Pattern caches for different hierarchy levels
        self.local_patterns: Dict[Tuple, Tensor] = {}
        self.global_patterns: Dict[Tuple, Tensor] = {}
        self.inter_node_patterns: Dict[Tuple, Tensor] = {}

        # Load balancing statistics
        self.load_stats = {
            "computation_times": [],
            "communication_volumes": [],
            "memory_usage": [],
        }

        self._pattern_lock = threading.Lock()

    def _detect_node_size(self) -> int:
        """Detect number of GPUs per node.
        
        Returns:
            Number of GPUs per node
        """
        if "LOCAL_WORLD_SIZE" in os.environ:
            return int(os.environ["LOCAL_WORLD_SIZE"])
        elif "OMPI_COMM_WORLD_LOCAL_SIZE" in os.environ:
            return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        else:
            # Default assumption: 8 GPUs per node
            return min(8, self.world_size)

    def create_hierarchical_pattern(
        self, seq_len: int, num_heads: int
    ) -> Dict[str, Tensor]:
        """Create hierarchical sparse pattern for distributed attention.
        
        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            
        Returns:
            Dictionary containing local, global, and inter_node patterns
        """
        num_blocks = seq_len // self.config.block_size

        patterns = {}

        # Level 1: Local node patterns (higher density)
        patterns["local"] = self._create_local_node_pattern(num_blocks, num_heads)

        # Level 2: Global patterns within node (medium density)
        patterns["global"] = self._create_global_pattern(num_blocks, num_heads)

        # Level 3: Inter-node patterns (sparse)
        patterns["inter_node"] = self._create_inter_node_pattern(num_blocks, num_heads)

        # Level 4: Load-balanced pattern adjustments
        if self.config.enable_load_balancing:
            patterns = self._apply_load_balancing(patterns, num_blocks)

        return patterns

    def _create_local_node_pattern(
        self, num_blocks: int, num_heads: int
    ) -> Tensor:
        """Create pattern for local node attention.
        
        Args:
            num_blocks: Number of blocks in the sequence
            num_heads: Number of attention heads
            
        Returns:
            Local attention pattern tensor
        """
        cache_key = (num_blocks, num_heads, self.config.local_sparsity, "local")

        with self._pattern_lock:
            if cache_key in self.local_patterns:
                return self.local_patterns[cache_key]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pattern = torch.zeros(
            num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
        )

        # Dense local attention window
        local_window = min(512, num_blocks // 4) // self.config.block_size

        for h in range(num_heads):
            for i in range(num_blocks):
                # Local window around each position
                start = max(0, i - local_window)
                end = min(num_blocks, i + local_window + 1)

                # Apply local sparsity
                window_size = end - start
                keep_indices = torch.randperm(window_size)[
                    : int(window_size * self.config.local_sparsity)
                ]
                pattern[h, i, start : start + len(keep_indices)] = True

        with self._pattern_lock:
            self.local_patterns[cache_key] = pattern

        return pattern

    def _create_global_pattern(self, num_blocks: int, num_heads: int) -> Tensor:
        """Create pattern for global attention within node.
        
        Args:
            num_blocks: Number of blocks in the sequence
            num_heads: Number of attention heads
            
        Returns:
            Global attention pattern tensor
        """
        cache_key = (num_blocks, num_heads, self.config.global_sparsity, "global")

        with self._pattern_lock:
            if cache_key in self.global_patterns:
                return self.global_patterns[cache_key]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pattern = torch.zeros(
            num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
        )

        # Global landmark tokens (first few blocks attend to everything)
        global_blocks = min(16, num_blocks // 8)

        for h in range(num_heads):
            # Global tokens attend to everything with sparsity
            for i in range(global_blocks):
                keep_indices = torch.randperm(num_blocks)[
                    : int(num_blocks * self.config.global_sparsity)
                ]
                pattern[h, i, keep_indices] = True

            # Everything attends to global tokens
            pattern[h, :, :global_blocks] = True

            # Dilated attention for remaining blocks
            for dilation in [1, 2, 4, 8]:
                for i in range(global_blocks, num_blocks):
                    for j in range(0, num_blocks, dilation):
                        if torch.rand(1).item() < self.config.global_sparsity:
                            pattern[h, i, j] = True

        with self._pattern_lock:
            self.global_patterns[cache_key] = pattern

        return pattern

    def _create_inter_node_pattern(
        self, num_blocks: int, num_heads: int
    ) -> Tensor:
        """Create pattern for inter-node attention (very sparse).
        
        Args:
            num_blocks: Number of blocks in the sequence
            num_heads: Number of attention heads
            
        Returns:
            Inter-node attention pattern tensor
        """
        cache_key = (
            num_blocks,
            num_heads,
            self.config.inter_node_sparsity,
            "inter_node",
        )

        with self._pattern_lock:
            if cache_key in self.inter_node_patterns:
                return self.inter_node_patterns[cache_key]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pattern = torch.zeros(
            num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
        )

        # Very sparse inter-node connections - only most important blocks
        num_inter_connections = int(
            num_blocks * num_blocks * self.config.inter_node_sparsity
        )

        for h in range(num_heads):
            # Random sparse connections for inter-node communication
            flat_indices = torch.randperm(num_blocks * num_blocks)[
                :num_inter_connections
            ]
            row_indices = flat_indices // num_blocks
            col_indices = flat_indices % num_blocks
            pattern[h, row_indices, col_indices] = True

        with self._pattern_lock:
            self.inter_node_patterns[cache_key] = pattern

        return pattern

    def _apply_load_balancing(
        self, patterns: Dict[str, Tensor], num_blocks: int
    ) -> Dict[str, Tensor]:
        """Apply load balancing adjustments to patterns.
        
        Args:
            patterns: Dictionary of attention patterns
            num_blocks: Number of blocks
            
        Returns:
            Adjusted patterns
        """
        if not self.load_stats["computation_times"]:
            return patterns  # No history for balancing yet

        # Calculate load imbalance
        recent_times = self.load_stats["computation_times"][
            -10:
        ]  # Last 10 measurements
        avg_time = sum(recent_times) / len(recent_times)

        # Check if this rank is overloaded
        is_overloaded = avg_time > (1 + self.config.load_balance_threshold) * avg_time
        is_underloaded = avg_time < (1 - self.config.load_balance_threshold) * avg_time

        if is_overloaded:
            # Reduce computation by increasing sparsity
            for pattern_name, pattern in patterns.items():
                sparsity_adjustment = 0.9  # Reduce by 10%
                patterns[pattern_name] = self._adjust_pattern_sparsity(
                    pattern, sparsity_adjustment
                )

        elif is_underloaded:
            # Increase computation by decreasing sparsity
            for pattern_name, pattern in patterns.items():
                sparsity_adjustment = 1.1  # Increase by 10%
                patterns[pattern_name] = self._adjust_pattern_sparsity(
                    pattern, sparsity_adjustment
                )

        return patterns

    def _adjust_pattern_sparsity(
        self, pattern: Tensor, adjustment: float
    ) -> Tensor:
        """Adjust pattern sparsity by given factor.
        
        Args:
            pattern: Attention pattern tensor
            adjustment: Sparsity adjustment factor
            
        Returns:
            Adjusted pattern
        """
        current_density = pattern.float().mean()
        target_density = torch.clamp(current_density * adjustment, 0.01, 0.95)

        if target_density < current_density:
            # Increase sparsity (remove connections)
            num_remove = int((current_density - target_density) * pattern.numel())
            active_indices = torch.nonzero(pattern, as_tuple=False)
            if len(active_indices) > num_remove:
                remove_indices = torch.randperm(len(active_indices))[:num_remove]
                for idx in remove_indices:
                    pattern[tuple(active_indices[idx])] = False

        elif target_density > current_density:
            # Decrease sparsity (add connections)
            num_add = int((target_density - current_density) * pattern.numel())
            inactive_indices = torch.nonzero(~pattern, as_tuple=False)
            if len(inactive_indices) > num_add:
                add_indices = torch.randperm(len(inactive_indices))[:num_add]
                for idx in add_indices:
                    pattern[tuple(inactive_indices[idx])] = True

        return pattern

    def update_load_stats(
        self, computation_time: float, communication_volume: int, memory_usage: int
    ) -> None:
        """Update load balancing statistics.
        
        Args:
            computation_time: Time taken for computation
            communication_volume: Volume of communication
            memory_usage: Memory usage in bytes
        """
        self.load_stats["computation_times"].append(computation_time)
        self.load_stats["communication_volumes"].append(communication_volume)
        self.load_stats["memory_usage"].append(memory_usage)

        # Keep only recent history
        max_history = 50
        for key in self.load_stats:
            if len(self.load_stats[key]) > max_history:
                self.load_stats[key] = self.load_stats[key][-max_history:]


__all__ = [
    "HierarchicalSparsePatternGenerator",
]