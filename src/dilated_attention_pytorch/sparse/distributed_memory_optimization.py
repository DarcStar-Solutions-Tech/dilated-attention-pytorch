"""
Memory optimization utilities for distributed sparse attention.

This module contains memory management and gradient communication optimizations
used by block-sparse ring distributed attention implementations.

Classes:
    AdaptiveMemoryPool: Dynamic memory pool with GPU pressure awareness
    OptimizedGradientCommunicator: Efficient gradient bucketing and compression
    GradientCompressor: Top-k gradient compression with error feedback
"""

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


class AdaptiveMemoryPool:
    """
    Adaptive memory pool management with dynamic cleanup thresholds.

    This class provides intelligent buffer management for block-sparse distributed
    attention, with features including:
    - Dynamic threshold adjustment based on GPU memory pressure
    - LRU eviction policy for efficient memory usage
    - Hot key cache for frequent access patterns
    - Optional pinned memory for faster GPU transfers
    - Thread-safe operations with fine-grained locking

    The pool automatically adjusts its behavior based on available GPU memory:
    - When memory < 10%: Aggressive cleanup (threshold / 4)
    - When memory > 50%: Conservative cleanup (threshold * 2)
    - Normal operation: Standard threshold
    """

    def __init__(self, device: torch.device, enable_pinned: bool = True):
        """Initialize the adaptive memory pool.

        Args:
            device: Target device for buffer allocation
            enable_pinned: Whether to use pinned memory for CUDA devices
        """
        self.device = device
        self.enable_pinned = enable_pinned and device.type == "cuda"

        # Memory pools with LRU tracking
        self._pools = OrderedDict()
        self._usage_count = {}
        self._access_order = []
        self._access_lock = threading.Lock()

        # Hot key cache for frequent access patterns
        self._hot_keys_cache = OrderedDict()
        self._max_hot_keys = 50

        # Statistics for adaptive management
        self._allocation_stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_buffer(
        self, shape: tuple, dtype: torch.dtype, pinned: bool = False
    ) -> torch.Tensor:
        """Get a buffer from the pool or allocate a new one.

        Args:
            shape: Buffer shape
            dtype: Data type
            pinned: Whether to use pinned memory

        Returns:
            Tensor buffer
        """
        key = (shape, dtype, pinned and self.enable_pinned)

        with self._access_lock:
            # Check hot keys first
            if key in self._hot_keys_cache:
                self._hot_keys_cache.move_to_end(key)
                self._allocation_stats["hits"] += 1
                return self._hot_keys_cache[key]

            # Check regular pool
            if key in self._pools and self._pools[key]:
                buffer = self._pools[key].pop()
                self._usage_count[key] = self._usage_count.get(key, 0) + 1

                # Add to hot keys if frequently accessed
                if self._usage_count[key] > 10:
                    self._add_to_hot_keys(key, buffer)

                self._allocation_stats["hits"] += 1
                return buffer

            # Allocate new buffer
            self._allocation_stats["misses"] += 1

            if pinned and self.enable_pinned:
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
            else:
                buffer = torch.empty(shape, dtype=dtype, device=self.device)

            return buffer

    def return_buffer(self, buffer: torch.Tensor, pinned: bool = False) -> None:
        """Return a buffer to the pool.

        Args:
            buffer: Buffer to return
            pinned: Whether the buffer is pinned
        """
        if buffer is None:
            return

        key = (buffer.shape, buffer.dtype, pinned and self.enable_pinned)

        with self._access_lock:
            if key not in self._pools:
                self._pools[key] = []
            self._pools[key].append(buffer)

            # Track access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def clear_unused_buffers(self, threshold: int = 100) -> None:
        """Clear unused buffers with adaptive threshold.

        Args:
            threshold: Base threshold for buffer count
        """
        with self._access_lock:
            # Adjust threshold based on GPU memory pressure
            if self.device.type == "cuda":
                threshold = self._adjust_threshold_for_memory_pressure(threshold)

            # Count total buffers
            total_buffers = sum(len(buffers) for buffers in self._pools.values())

            if total_buffers <= threshold:
                return

            # LRU eviction
            buffers_to_evict = total_buffers - threshold
            evicted = 0

            for key in list(self._access_order):
                if evicted >= buffers_to_evict:
                    break

                if key in self._pools and self._pools[key]:
                    num_to_evict = min(
                        len(self._pools[key]), buffers_to_evict - evicted
                    )
                    del self._pools[key][:num_to_evict]
                    evicted += num_to_evict
                    self._allocation_stats["evictions"] += num_to_evict

                    if not self._pools[key]:
                        del self._pools[key]
                        self._access_order.remove(key)

    def _adjust_threshold_for_memory_pressure(self, base_threshold: int) -> int:
        """Adjust cleanup threshold based on GPU memory pressure.

        Args:
            base_threshold: Base threshold value

        Returns:
            Adjusted threshold
        """
        try:
            free_memory = torch.cuda.mem_get_info(self.device.index)[0]
            total_memory = torch.cuda.mem_get_info(self.device.index)[1]
            memory_free_ratio = free_memory / total_memory

            if memory_free_ratio < 0.1:  # Less than 10% free
                return base_threshold // 4  # Aggressive cleanup
            elif memory_free_ratio > 0.5:  # More than 50% free
                return base_threshold * 2  # Conservative cleanup
            else:
                return base_threshold
        except Exception:
            return base_threshold

    def _add_to_hot_keys(self, key: tuple, buffer: torch.Tensor) -> None:
        """Add frequently accessed buffer to hot keys cache.

        Args:
            key: Buffer key
            buffer: Buffer tensor
        """
        if len(self._hot_keys_cache) >= self._max_hot_keys:
            # Remove oldest
            self._hot_keys_cache.popitem(last=False)
        self._hot_keys_cache[key] = buffer

    def get_stats(self) -> Dict[str, int]:
        """Get allocation statistics.

        Returns:
            Dictionary of statistics
        """
        with self._access_lock:
            stats = self._allocation_stats.copy()
            stats["total_buffers"] = sum(
                len(buffers) for buffers in self._pools.values()
            )
            stats["hot_keys"] = len(self._hot_keys_cache)
            return stats


class OptimizedGradientCommunicator:
    """
    Optimized gradient communication with bucketing and compression.

    Features:
    - Gradient bucketing by size and count thresholds
    - Top-k gradient compression with error feedback
    - Asynchronous all-reduce operations
    - Automatic gradient hook registration
    """

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        bucket_size_mb: float = 25.0,
        bucket_count: int = 32,
        compression_ratio: float = 0.1,
        enable_compression: bool = True,
    ):
        """Initialize gradient communicator.

        Args:
            process_group: Process group for communication
            bucket_size_mb: Maximum bucket size in MB
            bucket_count: Maximum gradients per bucket
            compression_ratio: Top-k compression ratio
            enable_compression: Whether to enable compression
        """
        self.process_group = process_group
        self.bucket_size_mb = bucket_size_mb
        self.bucket_count = bucket_count
        self.compression_ratio = compression_ratio
        self.enable_compression = enable_compression

        # Gradient buckets
        self._buckets: List[List[Tensor]] = []
        self._current_bucket: List[Tensor] = []
        self._current_bucket_size = 0

        # Compression state
        self._error_feedback: Dict[str, Tensor] = {}

        # Communication handles
        self._allreduce_handles = []
        self._lock = threading.Lock()

    def register_gradients(self, model: torch.nn.Module) -> None:
        """Register gradient hooks for automatic bucketing.

        Args:
            model: Model to register hooks on
        """

        def make_hook(param_name: str):
            def hook(grad: Tensor) -> Tensor:
                self.add_gradient(grad, param_name)
                return grad

            return hook

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(make_hook(name))

    def add_gradient(self, grad: Tensor, param_name: Optional[str] = None) -> None:
        """Add gradient to current bucket.

        Args:
            grad: Gradient tensor
            param_name: Parameter name for error feedback
        """
        with self._lock:
            # Apply compression if enabled
            if self.enable_compression and param_name:
                grad = self._compress_gradient(grad, param_name)

            # Calculate gradient size
            grad_size = grad.numel() * grad.element_size() / (1024 * 1024)  # MB

            # Check if we should start a new bucket
            if (
                self._current_bucket_size + grad_size > self.bucket_size_mb
                or len(self._current_bucket) >= self.bucket_count
            ):
                self._flush_current_bucket()

            # Add to current bucket
            self._current_bucket.append(grad)
            self._current_bucket_size += grad_size

    def _compress_gradient(self, grad: Tensor, param_name: str) -> Tensor:
        """Apply top-k compression with error feedback.

        Args:
            grad: Gradient to compress
            param_name: Parameter name for error feedback

        Returns:
            Compressed gradient
        """
        # Add error feedback if available
        if param_name in self._error_feedback:
            grad = grad + self._error_feedback[param_name]

        # Flatten gradient
        original_shape = grad.shape
        grad_flat = grad.flatten()

        # Top-k selection
        k = max(1, int(grad_flat.numel() * self.compression_ratio))
        values, indices = torch.topk(grad_flat.abs(), k)

        # Create sparse gradient
        sparse_grad = torch.zeros_like(grad_flat)
        sparse_grad[indices] = grad_flat[indices]

        # Store error for next iteration
        self._error_feedback[param_name] = (grad_flat - sparse_grad).reshape(
            original_shape
        )

        return sparse_grad.reshape(original_shape)

    def _flush_current_bucket(self) -> None:
        """Flush current bucket and start all-reduce."""
        if not self._current_bucket:
            return

        # Copy bucket for async operation
        bucket = self._current_bucket.copy()
        self._buckets.append(bucket)

        # Start async all-reduce
        handle = self._allreduce_bucket_async(bucket)
        if handle is not None:
            self._allreduce_handles.append(handle)

        # Reset current bucket
        self._current_bucket = []
        self._current_bucket_size = 0

    def _allreduce_bucket_async(self, bucket: List[Tensor]) -> Optional[dist.Work]:
        """Start async all-reduce for a bucket.

        Args:
            bucket: List of gradients to all-reduce

        Returns:
            Communication handle
        """
        if not bucket:
            return None

        # Flatten all gradients in bucket
        flat_grads = torch.cat([g.flatten() for g in bucket])

        # Start async all-reduce
        handle = dist.all_reduce(flat_grads, group=self.process_group, async_op=True)

        # Store mapping for unpacking later
        handle.bucket = bucket
        handle.flat_grads = flat_grads

        return handle

    def synchronize(self) -> None:
        """Wait for all pending communications to complete."""
        with self._lock:
            # Flush any remaining gradients
            self._flush_current_bucket()

            # Wait for all handles
            for handle in self._allreduce_handles:
                handle.wait()

                # Unpack gradients
                offset = 0
                for grad in handle.bucket:
                    numel = grad.numel()
                    grad.copy_(
                        handle.flat_grads[offset : offset + numel].reshape(grad.shape)
                    )
                    offset += numel

            # Clear handles
            self._allreduce_handles.clear()
            self._buckets.clear()


class GradientCompressor:
    """
    Advanced gradient compression for distributed training.

    Supports multiple compression algorithms:
    - Top-k sparsification
    - Random-k sparsification
    - Threshold-based sparsification
    """

    def __init__(self, compression_ratio: float = 0.1, algorithm: str = "topk"):
        """Initialize gradient compressor.

        Args:
            compression_ratio: Compression ratio (fraction to keep)
            algorithm: Compression algorithm
        """
        self.compression_ratio = compression_ratio
        self.algorithm = algorithm
        self.error_feedback = {}

    def compress(self, grad: Tensor, name: str) -> Tuple[Tensor, Tensor]:
        """Compress gradient.

        Args:
            grad: Gradient to compress
            name: Parameter name

        Returns:
            Compressed values and indices
        """
        # Add error feedback
        if name in self.error_feedback:
            grad = grad + self.error_feedback[name]

        # Flatten
        shape = grad.shape
        grad_flat = grad.flatten()

        # Apply compression algorithm
        if self.algorithm == "topk":
            values, indices = self._topk_compress(grad_flat)
        elif self.algorithm == "randomk":
            values, indices = self._randomk_compress(grad_flat)
        elif self.algorithm == "threshold":
            values, indices = self._threshold_compress(grad_flat)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Store error
        sparse_grad = torch.zeros_like(grad_flat)
        sparse_grad[indices] = values
        self.error_feedback[name] = (grad_flat - sparse_grad).reshape(shape)

        return values, indices

    def _topk_compress(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        """Top-k compression."""
        k = max(1, int(grad.numel() * self.compression_ratio))
        values, indices = torch.topk(grad.abs(), k)
        return grad[indices], indices

    def _randomk_compress(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        """Random-k compression."""
        k = max(1, int(grad.numel() * self.compression_ratio))
        indices = torch.randperm(grad.numel(), device=grad.device)[:k]
        return grad[indices], indices

    def _threshold_compress(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        """Threshold-based compression."""
        threshold = torch.quantile(grad.abs(), 1 - self.compression_ratio)
        mask = grad.abs() >= threshold
        indices = torch.nonzero(mask).squeeze()
        return grad[indices], indices


__all__ = [
    "AdaptiveMemoryPool",
    "OptimizedGradientCommunicator",
    "GradientCompressor",
]
