"""

# DEPRECATED: This file is scheduled for removal in v0.4.0.\n# Please use unified_memory_pool.py instead.\n\nimport warnings\nwarnings.warn(\n    f"{__file__} is deprecated and will be removed in v0.4.0. "\n    "Please use unified_memory_pool.py instead.",\n    DeprecationWarning,\n    stacklevel=2\n)\n
Fragment-aware memory pool implementation for Phase 1.4.

This module provides advanced memory management with fragmentation analysis,
compaction strategies, and efficient allocation algorithms.
"""

import bisect
import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor

logger = logging.getLogger("dilated_attention_pytorch.fragment_aware_pool")


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""

    address: int  # Virtual address (tensor id)
    size: int  # Size in bytes
    device: torch.device
    dtype: torch.dtype
    is_free: bool = True
    allocation_time: float = 0.0
    last_access_time: float = 0.0
    access_count: int = 0
    tensor: Optional[Tensor] = None

    def __post_init__(self):
        """Initialize timestamps."""
        current_time = time.time()
        if self.allocation_time == 0.0:
            self.allocation_time = current_time
        if self.last_access_time == 0.0:
            self.last_access_time = current_time


@dataclass
class FragmentationStats:
    """Statistics about memory fragmentation."""

    total_memory: int = 0
    used_memory: int = 0
    free_memory: int = 0
    largest_free_block: int = 0
    num_free_blocks: int = 0
    num_used_blocks: int = 0
    external_fragmentation: float = 0.0
    internal_fragmentation: float = 0.0
    average_block_size: float = 0.0
    fragmentation_score: float = 0.0

    def calculate_fragmentation(self) -> None:
        """Calculate fragmentation metrics."""
        if self.free_memory > 0 and self.largest_free_block > 0:
            # External fragmentation: inability to allocate large contiguous blocks
            self.external_fragmentation = 1.0 - (
                self.largest_free_block / self.free_memory
            )

        if self.total_memory > 0:
            # Internal fragmentation: wasted space within allocated blocks
            # Estimate based on block count and average size
            total_blocks = self.num_free_blocks + self.num_used_blocks
            if total_blocks > 0:
                self.average_block_size = self.total_memory / total_blocks
                # Assume 10% internal fragmentation on average
                self.internal_fragmentation = 0.1

        # Combined fragmentation score (0-1, higher is worse)
        self.fragmentation_score = (
            0.7 * self.external_fragmentation + 0.3 * self.internal_fragmentation
        )


class FragmentAwareMemoryPool:
    """
    Advanced memory pool with fragmentation management.

    Features:
    - Sophisticated fragmentation analysis
    - Multiple compaction strategies
    - Size-bucketed allocation
    - Best-fit, first-fit, and buddy allocation algorithms
    - Automatic defragmentation triggers
    - Memory coalescing
    """

    def __init__(
        self,
        initial_size: int = 1024 * 1024 * 1024,  # 1GB default
        growth_factor: float = 2.0,
        fragmentation_threshold: float = 0.3,
        compaction_strategy: str = "best_fit",
        enable_coalescing: bool = True,
    ):
        """Initialize the fragment-aware memory pool."""
        self.initial_size = initial_size
        self.growth_factor = growth_factor
        self.fragmentation_threshold = fragmentation_threshold
        self.compaction_strategy = compaction_strategy
        self.enable_coalescing = enable_coalescing

        # Memory blocks organized by device
        self._blocks: Dict[torch.device, List[MemoryBlock]] = defaultdict(list)

        # Free lists for each device (sorted by size)
        self._free_lists: Dict[torch.device, List[MemoryBlock]] = defaultdict(list)

        # Address to block mapping for fast lookup
        self._address_map: Dict[int, MemoryBlock] = {}

        # Size buckets for efficient allocation
        self._size_buckets: Dict[torch.device, Dict[int, List[MemoryBlock]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        # Fragmentation statistics
        self._stats: Dict[torch.device, FragmentationStats] = defaultdict(
            FragmentationStats
        )

        # Lock for thread safety
        self._lock = threading.RLock()

        # Compaction in progress flag
        self._compacting = False

        # Allocation strategies
        self._allocation_strategies = {
            "first_fit": self._first_fit_allocate,
            "best_fit": self._best_fit_allocate,
            "buddy": self._buddy_allocate,
        }

        logger.info(
            f"Initialized FragmentAwareMemoryPool with {initial_size / (1024**3):.2f}GB, "
            f"strategy: {compaction_strategy}"
        )

    def allocate(
        self,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Tensor:
        """
        Allocate memory with fragment-aware allocation.

        Args:
            size: Size in bytes
            dtype: Data type
            device: Device to allocate on
            shape: Optional shape for the tensor

        Returns:
            Allocated tensor
        """
        with self._lock:
            # Check if defragmentation is needed
            if self._should_defragment(device):
                self._defragment(device)

            # Try to allocate from free list
            block = self._allocate_from_free_list(size, device)

            if block is None:
                # No suitable free block, allocate new
                block = self._allocate_new_block(size, dtype, device)

            # Create tensor from block
            if shape is None:
                # Calculate shape from size and dtype
                element_size = torch.finfo(dtype).bits // 8
                num_elements = size // element_size
                shape = (num_elements,)

            # Create or reuse tensor
            if block.tensor is None or block.tensor.shape != shape:
                block.tensor = torch.empty(shape, dtype=dtype, device=device)

            # Update block state
            block.is_free = False
            block.last_access_time = time.time()
            block.access_count += 1

            # Update statistics
            self._update_stats(device)

            return block.tensor

    def deallocate(self, tensor: Tensor) -> None:
        """
        Return memory to the pool.

        Args:
            tensor: Tensor to deallocate
        """
        with self._lock:
            # Find block by tensor id
            tensor_id = id(tensor)
            block = self._address_map.get(tensor_id)

            if block is None:
                logger.warning(f"Attempt to deallocate unknown tensor {tensor_id}")
                return

            # Mark block as free
            block.is_free = True
            block.last_access_time = time.time()

            # Add to free list
            device = tensor.device
            bisect.insort(self._free_lists[device], block, key=lambda b: b.size)

            # Add to size bucket
            bucket = self._get_size_bucket(block.size)
            self._size_buckets[device][bucket].append(block)

            # Try to coalesce adjacent free blocks
            if self.enable_coalescing:
                self._coalesce_free_blocks(device)

            # Update statistics
            self._update_stats(device)

    def _should_defragment(self, device: torch.device) -> bool:
        """Check if defragmentation is needed."""
        stats = self._stats[device]
        stats.calculate_fragmentation()

        return (
            stats.fragmentation_score > self.fragmentation_threshold
            and not self._compacting
        )

    def _defragment(self, device: torch.device) -> None:
        """Perform memory defragmentation."""
        if self._compacting:
            return

        self._compacting = True
        logger.info(f"Starting defragmentation for {device}")

        try:
            # Strategy 1: Coalesce adjacent free blocks
            self._coalesce_free_blocks(device)

            # Strategy 2: Compact memory by moving blocks
            if self.compaction_strategy == "compact":
                self._compact_memory(device)

            # Strategy 3: Reorganize blocks by size
            elif self.compaction_strategy == "reorganize":
                self._reorganize_blocks(device)

            # Update statistics
            self._update_stats(device)

            stats = self._stats[device]
            logger.info(
                f"Defragmentation complete for {device}, "
                f"fragmentation: {stats.fragmentation_score:.3f}"
            )

        finally:
            self._compacting = False

    def _coalesce_free_blocks(self, device: torch.device) -> None:
        """Merge adjacent free blocks."""
        blocks = self._blocks[device]
        if len(blocks) < 2:
            return

        # Sort blocks by address
        blocks.sort(key=lambda b: b.address)

        # Merge adjacent free blocks
        i = 0
        while i < len(blocks) - 1:
            current = blocks[i]
            next_block = blocks[i + 1]

            if current.is_free and next_block.is_free:
                # Check if blocks are adjacent in memory
                if self._are_adjacent(current, next_block):
                    # Merge blocks
                    current.size += next_block.size

                    # Remove next block
                    blocks.pop(i + 1)
                    self._free_lists[device].remove(next_block)
                    del self._address_map[next_block.address]

                    # Update size buckets
                    old_bucket = self._get_size_bucket(next_block.size)
                    self._size_buckets[device][old_bucket].remove(next_block)

                    continue

            i += 1

    def _are_adjacent(self, block1: MemoryBlock, block2: MemoryBlock) -> bool:
        """Check if two blocks are adjacent in memory."""
        # This is a simplified check - in real implementation would need
        # actual memory addresses
        return abs(block1.address - block2.address) == block1.size

    def _compact_memory(self, device: torch.device) -> None:
        """Compact memory by moving blocks to reduce fragmentation."""
        blocks = self._blocks[device]

        # Sort blocks by address
        blocks.sort(key=lambda b: b.address)

        # Move used blocks to the beginning
        write_pos = 0
        for block in blocks:
            if not block.is_free and block.tensor is not None:
                if block.address != write_pos:
                    # Move block (in real implementation, would copy tensor data)
                    block.address = write_pos
                    self._address_map[id(block.tensor)] = block

                write_pos += block.size

    def _reorganize_blocks(self, device: torch.device) -> None:
        """Reorganize blocks by size for better allocation efficiency."""
        blocks = self._blocks[device]

        # Separate used and free blocks
        used_blocks = [b for b in blocks if not b.is_free]
        free_blocks = [b for b in blocks if b.is_free]

        # Sort by size (largest first for used, smallest first for free)
        used_blocks.sort(key=lambda b: b.size, reverse=True)
        free_blocks.sort(key=lambda b: b.size)

        # Reassign addresses
        address = 0

        # Place large used blocks first
        for block in used_blocks:
            block.address = address
            if block.tensor is not None:
                self._address_map[id(block.tensor)] = block
            address += block.size

        # Then free blocks
        for block in free_blocks:
            block.address = address
            address += block.size

    def _allocate_from_free_list(
        self, size: int, device: torch.device
    ) -> Optional[MemoryBlock]:
        """Allocate from free list using configured strategy."""
        strategy = self._allocation_strategies.get(
            self.compaction_strategy, self._best_fit_allocate
        )
        return strategy(size, device)

    def _first_fit_allocate(
        self, size: int, device: torch.device
    ) -> Optional[MemoryBlock]:
        """First-fit allocation strategy."""
        free_list = self._free_lists[device]

        for block in free_list:
            if block.size >= size:
                # Remove from free list
                free_list.remove(block)

                # Remove from size bucket
                bucket = self._get_size_bucket(block.size)
                self._size_buckets[device][bucket].remove(block)

                # Split block if much larger
                if block.size > size * 2:
                    self._split_block(block, size, device)

                return block

        return None

    def _best_fit_allocate(
        self, size: int, device: torch.device
    ) -> Optional[MemoryBlock]:
        """Best-fit allocation strategy."""
        # Use size buckets for efficient search
        bucket = self._get_size_bucket(size)

        # Search in current and larger buckets
        for search_bucket in range(bucket, bucket + 5):
            bucket_blocks = self._size_buckets[device].get(search_bucket, [])

            # Find best fit in bucket
            best_block = None
            best_waste = float("inf")

            for block in bucket_blocks:
                if block.size >= size:
                    waste = block.size - size
                    if waste < best_waste:
                        best_waste = waste
                        best_block = block

            if best_block:
                # Remove from free list and buckets
                self._free_lists[device].remove(best_block)
                bucket_blocks.remove(best_block)

                # Split if necessary
                if best_block.size > size * 1.5:
                    self._split_block(best_block, size, device)

                return best_block

        return None

    def _buddy_allocate(self, size: int, device: torch.device) -> Optional[MemoryBlock]:
        """Buddy allocation strategy."""
        # Round size up to power of 2
        size = 1 << (size - 1).bit_length()

        # Find smallest power-of-2 block
        free_list = self._free_lists[device]

        for block in free_list:
            block_size = 1 << (block.size - 1).bit_length()
            if block_size >= size:
                # Remove from free list
                free_list.remove(block)

                # Split recursively until we get the right size
                while block_size > size:
                    block_size //= 2
                    self._split_block(block, block_size, device)

                return block

        return None

    def _split_block(self, block: MemoryBlock, size: int, device: torch.device) -> None:
        """Split a block into used and free portions."""
        if block.size <= size:
            return

        # Create new free block for remainder
        remainder = MemoryBlock(
            address=block.address + size,
            size=block.size - size,
            device=device,
            dtype=block.dtype,
            is_free=True,
        )

        # Update original block
        block.size = size

        # Add remainder to data structures
        self._blocks[device].append(remainder)
        bisect.insort(self._free_lists[device], remainder, key=lambda b: b.size)

        bucket = self._get_size_bucket(remainder.size)
        self._size_buckets[device][bucket].append(remainder)

    def _allocate_new_block(
        self, size: int, dtype: torch.dtype, device: torch.device
    ) -> MemoryBlock:
        """Allocate a new memory block."""
        # In real implementation, would allocate from system
        # For now, create a simulated block
        block = MemoryBlock(
            address=self._get_next_address(device),
            size=size,
            device=device,
            dtype=dtype,
            is_free=False,
        )

        self._blocks[device].append(block)
        return block

    def _get_next_address(self, device: torch.device) -> int:
        """Get next available address."""
        blocks = self._blocks[device]
        if not blocks:
            return 0

        # Find highest address
        max_block = max(blocks, key=lambda b: b.address + b.size)
        return max_block.address + max_block.size

    def _get_size_bucket(self, size: int) -> int:
        """Get size bucket index for a given size."""
        # Use logarithmic buckets
        if size <= 0:
            return 0
        return int(math.log2(size))

    def _update_stats(self, device: torch.device) -> None:
        """Update fragmentation statistics."""
        stats = self._stats[device]
        blocks = self._blocks[device]

        # Reset stats
        stats.total_memory = sum(b.size for b in blocks)
        stats.used_memory = sum(b.size for b in blocks if not b.is_free)
        stats.free_memory = stats.total_memory - stats.used_memory

        # Count blocks
        stats.num_used_blocks = sum(1 for b in blocks if not b.is_free)
        stats.num_free_blocks = sum(1 for b in blocks if b.is_free)

        # Find largest free block
        free_blocks = [b for b in blocks if b.is_free]
        if free_blocks:
            stats.largest_free_block = max(b.size for b in free_blocks)
        else:
            stats.largest_free_block = 0

        # Calculate fragmentation
        stats.calculate_fragmentation()

    def get_stats(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            if device:
                stats = self._stats[device]
                return {
                    "device": str(device),
                    "total_memory": stats.total_memory,
                    "used_memory": stats.used_memory,
                    "free_memory": stats.free_memory,
                    "fragmentation_score": stats.fragmentation_score,
                    "external_fragmentation": stats.external_fragmentation,
                    "internal_fragmentation": stats.internal_fragmentation,
                    "num_blocks": stats.num_used_blocks + stats.num_free_blocks,
                    "largest_free_block": stats.largest_free_block,
                }
            else:
                # Return stats for all devices
                all_stats = {}
                for dev, stats in self._stats.items():
                    all_stats[str(dev)] = {
                        "total_memory": stats.total_memory,
                        "used_memory": stats.used_memory,
                        "fragmentation_score": stats.fragmentation_score,
                    }
                return all_stats

    def get_fragmentation_report(self) -> str:
        """Generate detailed fragmentation report."""
        lines = ["Fragment-Aware Memory Pool Report", "=" * 40, ""]

        with self._lock:
            for device, stats in self._stats.items():
                status = (
                    "🔴 HIGH"
                    if stats.fragmentation_score > 0.5
                    else "🟡 MEDIUM"
                    if stats.fragmentation_score > 0.3
                    else "🟢 LOW"
                )

                lines.extend(
                    [
                        f"Device: {device}",
                        f"  Status: {status}",
                        f"  Total Memory: {stats.total_memory / (1024**3):.2f} GB",
                        f"  Used Memory: {stats.used_memory / (1024**3):.2f} GB ({stats.used_memory / stats.total_memory * 100:.1f}%)",
                        f"  Fragmentation Score: {stats.fragmentation_score:.3f}",
                        f"  External Fragmentation: {stats.external_fragmentation:.3f}",
                        f"  Internal Fragmentation: {stats.internal_fragmentation:.3f}",
                        f"  Number of Blocks: {stats.num_used_blocks + stats.num_free_blocks}",
                        f"  Largest Free Block: {stats.largest_free_block / (1024**2):.2f} MB",
                        "",
                    ]
                )

        return "\n".join(lines)
