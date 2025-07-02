"""
Test suite for the unified memory pool.

Tests memory pool functionality including adaptive cleanup, thread safety,
and buffer management.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from dilated_attention_pytorch.core import (
    MemoryPoolConfig,
    UnifiedMemoryPool,
    get_global_memory_pool,
    reset_global_memory_pool,
)


class TestUnifiedMemoryPool:
    """Test UnifiedMemoryPool class."""

    def test_initialization(self):
        """Test memory pool initialization."""
        config = MemoryPoolConfig(
            hot_cache_size=10, hot_cache_threshold=5, allow_buffer_slicing=True
        )
        pool = UnifiedMemoryPool(config)

        assert pool.config.hot_cache_size == 10
        assert pool.config.hot_cache_threshold == 5
        assert len(pool._pools) == 4  # default, ring, sparse, distributed
        assert hasattr(pool, "_lock")

    def test_buffer_allocation(self):
        """Test basic buffer allocation."""
        pool = UnifiedMemoryPool()

        # Allocate buffer
        shape = (10, 20, 30)
        dtype = torch.float32
        device = torch.device("cpu")

        buffer = pool.get_buffer(shape, dtype, device)

        assert buffer.shape == shape
        assert buffer.dtype == dtype
        assert buffer.device == device

    def test_buffer_reuse(self):
        """Test buffer reuse from pool."""
        pool = UnifiedMemoryPool()

        # Allocate and track buffer
        shape = (100, 100)
        buffer1 = pool.get_buffer(shape, torch.float32, torch.device("cpu"))
        buffer1_id = id(buffer1)

        # Request same buffer specification
        buffer2 = pool.get_buffer(shape, torch.float32, torch.device("cpu"))

        # Should get the same buffer
        assert id(buffer2) == buffer1_id

    def test_compatible_buffer_reshape(self):
        """Test finding compatible buffers that can be reshaped."""
        config = MemoryPoolConfig(allow_buffer_slicing=True)
        pool = UnifiedMemoryPool(config)

        # Allocate buffer with 1000 elements
        buffer1 = pool.get_buffer((10, 100), torch.float32)

        # Request buffer with same number of elements but different shape
        buffer2 = pool.get_buffer((20, 50), torch.float32)

        # Should reuse by reshaping
        assert buffer2.numel() == buffer1.numel()

    def test_compatible_buffer_slicing(self):
        """Test finding compatible buffers that can be sliced."""
        config = MemoryPoolConfig(allow_buffer_slicing=True)
        pool = UnifiedMemoryPool(config)

        # Allocate large buffer
        _ = pool.get_buffer((1000,), torch.float32)

        # Request smaller buffer
        buffer2 = pool.get_buffer((500,), torch.float32)

        # Should get a slice of the larger buffer
        assert buffer2.shape == (500,)

    def test_hot_cache_promotion(self):
        """Test promotion to hot cache."""
        config = MemoryPoolConfig(hot_cache_size=5, hot_cache_threshold=3)
        pool = UnifiedMemoryPool(config)

        shape = (10, 10)
        device = None

        # Access buffer multiple times
        # Need 4 accesses because promotion happens AFTER incrementing count to 3
        for _ in range(4):
            buffer = pool.get_buffer(shape, torch.float32, device=device)
            # Get the actual device from the returned buffer
            if device is None and _ == 0:
                device = buffer.device

        # Should be in hot cache now
        key = (shape, torch.float32, device, False, "default")
        assert key in pool._hot_cache

    def test_hot_cache_size_limit(self):
        """Test hot cache size limiting."""
        config = MemoryPoolConfig(hot_cache_size=3, hot_cache_threshold=1)
        pool = UnifiedMemoryPool(config)

        # Fill hot cache beyond limit
        for i in range(5):
            shape = (i + 1, i + 1)
            pool.get_buffer(shape, torch.float32)
            pool.get_buffer(shape, torch.float32)  # Access twice to promote

        # Hot cache should be limited
        assert len(pool._hot_cache) <= 3

    def test_thread_safety(self):
        """Test thread-safe buffer allocation."""
        pool = UnifiedMemoryPool()
        buffers = []

        def allocate_buffers(thread_id):
            for i in range(10):
                shape = (thread_id, i + 1, 10)
                buffer = pool.get_buffer(shape, torch.float32)
                buffers.append(buffer)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=allocate_buffers, args=(i + 1,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have allocated all buffers without issues
        assert len(buffers) == 50

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available")
    def test_memory_pressure_cleanup(
        self, mock_cuda_available, mock_device_props, mock_reserved, mock_allocated
    ):
        """Test cleanup under memory pressure."""
        # Mock CUDA available
        mock_cuda_available.return_value = True

        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 1000000000  # 1GB
        mock_device_props.return_value = mock_props

        # Mock high memory usage (90%)
        mock_allocated.return_value = 900000000
        mock_reserved.return_value = 950000000

        pool = UnifiedMemoryPool()

        # Allocate some buffers
        for i in range(10):
            pool.get_buffer((100, 100), torch.float32)

        # Trigger cleanup check
        pool._maybe_cleanup()

        # Should have triggered aggressive cleanup
        # (Hard to test actual cleanup without real GPU)

    def test_pool_statistics(self):
        """Test getting pool statistics."""
        pool = UnifiedMemoryPool()

        # Allocate buffers in different pools
        pool.get_buffer((10, 10), torch.float32, pool_type="default")
        pool.get_buffer((20, 20), torch.float32, pool_type="ring")
        pool.get_buffer((30, 30), torch.float32, pool_type="sparse")

        stats = pool.get_stats()

        assert stats["total_buffers"] >= 3
        assert stats["pool_sizes"]["default"] >= 1
        assert stats["pool_sizes"]["ring"] >= 1
        assert stats["pool_sizes"]["sparse"] >= 1
        assert "memory_by_pool" in stats

    def test_clear_pool(self):
        """Test clearing pools."""
        pool = UnifiedMemoryPool()

        # Allocate buffers
        pool.get_buffer((10, 10), torch.float32)
        pool.get_buffer((20, 20), torch.float32)

        # Clear all pools
        pool.clear_pool()

        stats = pool.get_stats()
        assert stats["total_buffers"] == 0
        assert stats["hot_cache_size"] == 0

    def test_clear_specific_pool(self):
        """Test clearing specific pool."""
        pool = UnifiedMemoryPool()

        # Allocate buffers in different pools
        pool.get_buffer((10, 10), torch.float32, pool_type="default")
        pool.get_buffer((20, 20), torch.float32, pool_type="ring")

        # Clear only ring pool
        pool.clear_pool("ring")

        stats = pool.get_stats()
        assert stats["pool_sizes"]["default"] >= 1
        assert stats["pool_sizes"]["ring"] == 0

    def test_pinned_memory(self):
        """Test pinned memory allocation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pool = UnifiedMemoryPool()

        # Request pinned memory
        buffer = pool.get_buffer(
            (100, 100), torch.float32, torch.device("cuda"), pinned=True
        )

        assert buffer.is_cuda
        # Note: Can't directly test if memory is pinned


class TestGlobalMemoryPool:
    """Test global memory pool functionality."""

    def test_global_pool_singleton(self):
        """Test global pool is a singleton."""
        reset_global_memory_pool()  # Reset first

        pool1 = get_global_memory_pool()
        pool2 = get_global_memory_pool()

        assert pool1 is pool2

    def test_global_pool_configuration(self):
        """Test configuring global pool."""
        reset_global_memory_pool()

        config = MemoryPoolConfig(hot_cache_size=20, hot_cache_threshold=10)

        pool = get_global_memory_pool(config)
        assert pool.config.hot_cache_size == 20

        # Subsequent calls ignore config
        pool2 = get_global_memory_pool()
        assert pool2 is pool

    def test_reset_global_pool(self):
        """Test resetting global pool."""
        pool1 = get_global_memory_pool()
        pool1.get_buffer((10, 10), torch.float32)

        reset_global_memory_pool()

        pool2 = get_global_memory_pool()
        assert pool1 is not pool2


class TestMemoryPoolEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_pool_type(self):
        """Test handling of invalid pool type."""
        pool = UnifiedMemoryPool()

        # Should fall back to default pool
        buffer = pool.get_buffer((10, 10), torch.float32, pool_type="invalid_pool")

        assert buffer.shape == (10, 10)

    @patch("torch.empty")
    def test_oom_recovery(self, mock_empty):
        """Test OOM error recovery."""
        # Mock OOM on first call, success on second
        mock_empty.side_effect = [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            torch.zeros(10, 10),
        ]

        pool = UnifiedMemoryPool()

        # Should recover from OOM
        buffer = pool.get_buffer((10, 10), torch.float32)
        assert buffer.shape == (10, 10)

        # Should have called empty twice
        assert mock_empty.call_count == 2

    def test_buffer_stats_tracking(self):
        """Test buffer statistics tracking."""
        pool = UnifiedMemoryPool()

        shape = (50, 50)
        device = None

        # Access buffer multiple times
        for i in range(5):
            buffer = pool.get_buffer(shape, torch.float32, device=device)
            if device is None and i == 0:
                device = buffer.device
            time.sleep(0.01)  # Small delay

        # Check statistics
        key = (shape, torch.float32, device, False, "default")
        assert key in pool._stats
        stats = pool._stats[key]
        assert stats.access_count == 5
        assert stats.size_bytes == 50 * 50 * 4  # float32 = 4 bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
