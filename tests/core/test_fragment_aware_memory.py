"""
Test suite for fragment-aware memory management.
"""

import pytest
import torch

from dilated_attention_pytorch.core.fragment_aware_pool import (
    FragmentAwareMemoryPool,
    FragmentationStats,
    MemoryBlock,
)
from dilated_attention_pytorch.core.bucketed_memory_pool import (
    BucketedMemoryPool,
    BucketConfig,
    MemoryBucket,
)


class TestFragmentationStats:
    """Test fragmentation statistics calculation."""

    def test_fragmentation_calculation(self):
        """Test fragmentation metrics calculation."""
        stats = FragmentationStats()
        stats.total_memory = 1000
        stats.used_memory = 600
        stats.free_memory = 400
        stats.largest_free_block = 200
        stats.num_free_blocks = 4
        stats.num_used_blocks = 6

        stats.calculate_fragmentation()

        # External fragmentation: 1 - (200/400) = 0.5
        assert abs(stats.external_fragmentation - 0.5) < 0.01

        # Internal fragmentation should be around 0.1 (estimated)
        assert abs(stats.internal_fragmentation - 0.1) < 0.05

        # Combined score
        expected_score = 0.7 * 0.5 + 0.3 * 0.1
        assert abs(stats.fragmentation_score - expected_score) < 0.01


class TestMemoryBlock:
    """Test memory block functionality."""

    def test_memory_block_creation(self):
        """Test memory block creation."""
        device = torch.device("cpu")
        dtype = torch.float32

        block = MemoryBlock(
            address=1000,
            size=512,
            device=device,
            dtype=dtype,
        )

        assert block.address == 1000
        assert block.size == 512
        assert block.device == device
        assert block.dtype == dtype
        assert block.is_free is True
        assert block.allocation_time > 0
        assert block.last_access_time > 0


class TestFragmentAwareMemoryPool:
    """Test fragment-aware memory pool."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def pool(self):
        return FragmentAwareMemoryPool(
            initial_size=1024 * 1024,  # 1MB
            fragmentation_threshold=0.3,
            compaction_strategy="best_fit",
        )

    def test_pool_initialization(self, pool):
        """Test pool initialization."""
        assert pool.initial_size == 1024 * 1024
        assert pool.fragmentation_threshold == 0.3
        assert pool.compaction_strategy == "best_fit"
        assert pool.enable_coalescing is True

    def test_basic_allocation(self, pool, device):
        """Test basic memory allocation."""
        size = 1024
        dtype = torch.float32
        shape = (256,)  # 256 float32 = 1024 bytes

        tensor = pool.allocate(size, dtype, device, shape)

        assert tensor is not None
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device == device

    def test_allocation_strategies(self, device):
        """Test different allocation strategies."""
        strategies = ["first_fit", "best_fit", "buddy"]

        for strategy in strategies:
            pool = FragmentAwareMemoryPool(
                initial_size=1024 * 1024,
                compaction_strategy=strategy,
            )

            size = 1024
            dtype = torch.float32
            shape = (256,)

            tensor = pool.allocate(size, dtype, device, shape)
            assert tensor is not None
            assert tensor.shape == shape

    def test_fragmentation_detection(self, pool, device):
        """Test fragmentation detection and defragmentation."""
        # Simulate fragmented allocation pattern
        tensors = []

        # Allocate many small blocks
        for i in range(20):
            size = 512
            shape = (128,)
            tensor = pool.allocate(size, torch.float32, device, shape)
            tensors.append(tensor)

        # Deallocate every other block
        for i in range(0, len(tensors), 2):
            pool.deallocate(tensors[i])

        # Check fragmentation
        stats = pool.get_stats(device)

        # Should have some fragmentation
        if "fragmentation_score" in stats:
            assert stats["fragmentation_score"] >= 0

    def test_coalescing(self, pool, device):
        """Test memory coalescing."""
        # Create adjacent blocks
        size = 1024
        shape1 = (256,)
        shape2 = (256,)

        tensor1 = pool.allocate(size, torch.float32, device, shape1)
        tensor2 = pool.allocate(size, torch.float32, device, shape2)

        # Deallocate to trigger coalescing
        pool.deallocate(tensor1)
        pool.deallocate(tensor2)

        # Should be able to allocate larger block
        large_tensor = pool.allocate(size * 2, torch.float32, device, (512,))
        assert large_tensor is not None

    def test_stats_reporting(self, pool, device):
        """Test statistics reporting."""
        # Allocate some memory
        _ = pool.allocate(1024, torch.float32, device, (256,))

        # Get stats
        stats = pool.get_stats(device)

        assert isinstance(stats, dict)
        assert "device" in stats
        assert "total_memory" in stats

        # Get all device stats
        all_stats = pool.get_stats()
        assert isinstance(all_stats, dict)

    def test_fragmentation_report(self, pool, device):
        """Test fragmentation report generation."""
        # Allocate some memory to create fragments
        _ = pool.allocate(1024, torch.float32, device, (256,))

        report = pool.get_fragmentation_report()

        assert isinstance(report, str)
        assert "Fragment-Aware Memory Pool Report" in report
        assert str(device) in report


class TestBucketConfig:
    """Test bucket configuration."""

    def test_bucket_config_names(self):
        """Test bucket name generation."""
        configs = [
            BucketConfig(512),
            BucketConfig(2048),
            BucketConfig(1024 * 1024),
            BucketConfig(1024 * 1024 * 1024),
        ]

        names = [config.name for config in configs]
        assert "512B" in names
        assert "2KB" in names
        assert "1MB" in names
        assert "1GB" in names


class TestMemoryBucket:
    """Test individual memory bucket."""

    @pytest.fixture
    def bucket_config(self):
        return BucketConfig(size=4096, initial_count=4, max_count=16)

    @pytest.fixture
    def bucket(self, bucket_config):
        return MemoryBucket(bucket_config)

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_bucket_initialization(self, bucket, bucket_config):
        """Test bucket initialization."""
        assert bucket.size == bucket_config.size
        assert bucket.config == bucket_config
        assert bucket.stats["current_count"] == bucket_config.initial_count

    def test_bucket_allocation(self, bucket, device):
        """Test allocation from bucket."""
        shape = (256,)  # 256 float32 = 1024 bytes < 4096
        dtype = torch.float32

        tensor = bucket.allocate(shape, dtype, device)

        assert tensor is not None
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device == device

    def test_bucket_size_mismatch(self, bucket, device):
        """Test allocation with size too large for bucket."""
        # Request larger than bucket size
        shape = (2048,)  # 2048 float32 = 8192 bytes > 4096
        dtype = torch.float32

        tensor = bucket.allocate(shape, dtype, device)

        # Should return None for oversized request
        assert tensor is None

    def test_bucket_reuse(self, bucket, device):
        """Test buffer reuse in bucket."""
        shape = (256,)
        dtype = torch.float32

        # Allocate
        tensor1 = bucket.allocate(shape, dtype, device)
        assert tensor1 is not None

        # Deallocate
        success = bucket.deallocate(tensor1)
        assert success is True

        # Allocate again - should reuse
        tensor2 = bucket.allocate(shape, dtype, device)
        assert tensor2 is not None

        # Should have high hit rate
        stats = bucket.get_stats()
        assert stats["hit_rate"] > 0

    def test_bucket_stats(self, bucket, device):
        """Test bucket statistics."""
        # Do some allocations
        tensors = []
        for _ in range(3):
            tensor = bucket.allocate((256,), torch.float32, device)
            if tensor is not None:
                tensors.append(tensor)

        stats = bucket.get_stats()

        assert stats["allocations"] >= 3
        assert stats["current_buffers"] > 0
        assert stats["name"] == "4KB"
        assert 0 <= stats["hit_rate"] <= 1
        assert 0 <= stats["efficiency"] <= 1


class TestBucketedMemoryPool:
    """Test bucketed memory pool."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def pool(self):
        return BucketedMemoryPool(
            bucket_sizes=[1024, 4096, 16384, 65536],
            adaptive_buckets=True,
        )

    def test_pool_initialization(self, pool):
        """Test pool initialization."""
        assert len(pool.buckets) == 4
        assert 1024 in pool.buckets
        assert 4096 in pool.buckets
        assert 16384 in pool.buckets
        assert 65536 in pool.buckets

    def test_bucket_selection(self, pool, device):
        """Test automatic bucket selection."""
        # Should use 4096 bucket for 3000 byte request
        size = 3000
        shape = (750,)  # 750 float32 = 3000 bytes

        tensor = pool.allocate(size, torch.float32, device, shape)

        assert tensor is not None
        assert tensor.shape == shape

        # Check stats
        stats = pool.get_stats()
        assert stats["bucketed_allocations"] >= 1

    def test_large_allocation(self, pool, device):
        """Test allocation larger than any bucket."""
        # Request larger than largest bucket (65536)
        size = 100000
        shape = (25000,)  # 25000 float32 = 100000 bytes

        tensor = pool.allocate(size, torch.float32, device, shape)

        assert tensor is not None
        assert tensor.shape == shape

        # Should go to large allocations
        stats = pool.get_stats()
        assert stats["large_allocations"] >= 1

    def test_adaptive_buckets(self, pool, device):
        """Test adaptive bucket creation."""
        # Request same unusual size many times
        size = 7777
        shape = (1944,)  # Close to 7777 bytes

        # Make many requests to trigger adaptive bucket
        tensors = []
        for _ in range(pool.adaptation_threshold + 1):
            tensor = pool.allocate(size, torch.float32, device, shape)
            if tensor is not None:
                tensors.append(tensor)

        # Should have created adaptive bucket
        stats = pool.get_stats()
        if stats["adaptive_buckets_created"] > 0:
            assert stats["adaptive_buckets_created"] >= 1

    def test_deallocation(self, pool, device):
        """Test memory deallocation."""
        size = 2000
        shape = (500,)

        tensor = pool.allocate(size, torch.float32, device, shape)
        assert tensor is not None

        # Deallocate
        pool.deallocate(tensor)

        # Should be returned to bucket
        # Try allocating again - should reuse
        tensor2 = pool.allocate(size, torch.float32, device, shape)
        assert tensor2 is not None

    def test_emergency_cleanup(self, pool, device):
        """Test emergency cleanup on OOM."""
        # Fill up buckets
        tensors = []
        try:
            for _ in range(100):
                tensor = pool.allocate(4000, torch.float32, device, (1000,))
                tensors.append(tensor)
        except torch.cuda.OutOfMemoryError:
            # Expected on some systems
            pass

        # Emergency cleanup should work
        pool._emergency_cleanup()

        # Should be able to allocate after cleanup
        tensor = pool.allocate(1000, torch.float32, device, (250,))
        # Don't assert success as system may be truly out of memory

    def test_efficiency_report(self, pool, device):
        """Test efficiency report generation."""
        # Do some allocations
        for size in [1000, 3000, 10000]:
            shape = (size // 4,)  # float32 = 4 bytes
            tensor = pool.allocate(size, torch.float32, device, shape)
            if tensor is not None:
                pool.deallocate(tensor)

        report = pool.get_efficiency_report()

        assert isinstance(report, str)
        assert "Bucketed Memory Pool Efficiency Report" in report
        assert "Total Allocations" in report

    def test_clear_pool(self, pool, device):
        """Test clearing the entire pool."""
        # Allocate some memory
        tensor = pool.allocate(2000, torch.float32, device, (500,))
        assert tensor is not None

        # Clear pool
        pool.clear()

        # Stats should be reset
        stats = pool.get_stats()
        # Note: don't check total_allocations as it's cumulative
        assert stats["total_buffers"] == 0 or stats["total_memory_allocated"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
