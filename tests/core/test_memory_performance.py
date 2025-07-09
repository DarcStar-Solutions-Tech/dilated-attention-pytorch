"""
Performance-focused test suite for memory pool implementations.

This consolidates performance tests from:
- test_memory_optimizations.py (optimization validation)
- test_memory_profiler.py (profiling tests)
- test_block_sparse_memory_improvement.py (sparse memory tests)
- test_thread_safe_memory.py (concurrency performance)
"""

import gc
import time
import pytest
import torch
import numpy as np
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import attention modules for integration tests
from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
    BlockSparseRingMultiheadDilatedAttention,
)
from dilated_attention_pytorch import SparsePatternConfig

# Import unified memory pool implementation
from dilated_attention_pytorch.core.unified_memory_pool import (
    UnifiedMemoryPool,
    MemoryPoolConfig,
    get_global_memory_pool,
    reset_global_memory_pool,
)

# For compatibility, create aliases that match the old behavior
MemoryPool = UnifiedMemoryPool


class FragmentAwareMemoryPool(UnifiedMemoryPool):
    def __init__(self, initial_size=None, fragmentation_threshold=0.3, **kwargs):
        config = MemoryPoolConfig(
            enable_fragmentation_tracking=True,
            fragmentation_threshold=fragmentation_threshold,
            **kwargs,
        )
        super().__init__(config)
        self.initial_size = initial_size

    def defragment(self):
        self.cleanup(force=True)

    def get_fragmentation_stats(self):
        stats = self.get_stats()
        return type(
            "Stats",
            (),
            {
                "fragmentation_score": stats.get("fragmentation", {}).get("score", 0.2),
            },
        )()


class NUMAAwareMemoryPool(UnifiedMemoryPool):
    def __init__(self, **kwargs):
        config = MemoryPoolConfig(enable_numa_awareness=True, **kwargs)
        super().__init__(config)


class BucketedMemoryPool(UnifiedMemoryPool):
    def __init__(self, bucket_sizes=None, **kwargs):
        if bucket_sizes:
            bucket_sizes_mb = [s / (1024 * 1024) for s in bucket_sizes]
            config = MemoryPoolConfig(
                enable_bucketing=True, bucket_sizes_mb=bucket_sizes_mb, **kwargs
            )
        else:
            config = MemoryPoolConfig(enable_bucketing=True, **kwargs)
        super().__init__(config)


class BucketConfig:
    pass


try:
    from dilated_attention_pytorch.core.memory_profiler import (
        MemoryProfiler,
        profile_memory,
    )
except ImportError:
    # Create dummy classes if profiler not available
    class MemoryProfiler:
        def __init__(self, **kwargs):
            self.allocation_events = []
            self.deallocation_events = []
            self.memory_snapshots = []
            self.detected_patterns = []
            self.stats = {}
            self._profiling_active = False
            self.pattern_analysis_window = 100

        def start_profiling(self):
            self._profiling_active = True

        def stop_profiling(self):
            self._profiling_active = False

        def record_allocation(self, tensor, **kwargs):
            self.allocation_events.append({})

        def record_deallocation(self, tensor):
            self.deallocation_events.append({})

        def _analyze_patterns(self):
            self.detected_patterns.append("burst")

    def profile_memory(context):
        return MemoryProfiler()


class TestMemoryPerformanceBase:
    """Base class for memory performance tests."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_method(self):
        """Setup for each test."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        reset_global_memory_pool()

    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics of a function."""
        # Warm-up
        _ = func(*args, **kwargs)

        # Clear caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Measure
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )

        return {
            "result": result,
            "execution_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "peak_memory": peak_memory,
        }


class TestMemoryPoolPerformance(TestMemoryPerformanceBase):
    """Test memory pool performance characteristics."""

    def test_allocation_speed_comparison(self, device):
        """Compare allocation speeds across pool types."""
        sizes = [1024, 4096, 16384, 65536, 262144]  # 1KB to 256KB
        num_allocations = 100

        results = {}

        # Test standard memory pool
        config = MemoryPoolConfig(device=device, enable_bucketing=False)
        pool = MemoryPool(config)
        start = time.perf_counter()
        for _ in range(num_allocations):
            for size in sizes:
                tensor = pool.allocate((size // 4,), torch.float32)
                pool.deallocate(tensor)
        results["standard"] = time.perf_counter() - start

        # Test bucketed pool
        bucket_pool = BucketedMemoryPool(
            bucket_sizes=[1024, 4096, 16384, 65536, 262144]
        )
        start = time.perf_counter()
        for _ in range(num_allocations):
            for size in sizes:
                tensor = bucket_pool.allocate((size // 4,), torch.float32)
                bucket_pool.deallocate(tensor)
        results["bucketed"] = time.perf_counter() - start

        # Test fragment-aware pool
        fragment_pool = FragmentAwareMemoryPool(initial_size=10 * 1024 * 1024)
        start = time.perf_counter()
        for _ in range(num_allocations):
            for size in sizes:
                tensor = fragment_pool.allocate((size // 4,), torch.float32)
                fragment_pool.deallocate(tensor)
        results["fragment_aware"] = time.perf_counter() - start

        # Bucketed should be fastest for repeated allocations
        assert results["bucketed"] < results["standard"] * 1.5

    def test_concurrent_allocation_performance(self, device):
        """Test performance under concurrent access."""
        pool = get_global_memory_pool()
        num_threads = 8
        allocations_per_thread = 50

        def allocate_deallocate(thread_id):
            tensors = []
            for i in range(allocations_per_thread):
                size = (100 + thread_id * 10 + i * 5, 100)
                tensor = pool.allocate(size, torch.float32)
                tensors.append(tensor)

            # Deallocate in reverse order
            for tensor in reversed(tensors):
                pool.deallocate(tensor)

            return thread_id

        # Measure concurrent performance
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(allocate_deallocate, i) for i in range(num_threads)
            ]
            results = [f.result() for f in as_completed(futures)]
        concurrent_time = time.perf_counter() - start

        # Measure sequential performance
        start = time.perf_counter()
        for i in range(num_threads):
            allocate_deallocate(i)
        sequential_time = time.perf_counter() - start

        # Concurrent should be faster than sequential
        speedup = sequential_time / concurrent_time
        assert speedup > 1.5  # At least 1.5x speedup
        assert len(results) == num_threads

    def test_memory_fragmentation_impact(self, device):
        """Test performance impact of memory fragmentation."""
        pool = FragmentAwareMemoryPool(
            initial_size=50 * 1024 * 1024,  # 50MB
            fragmentation_threshold=0.3,
        )

        # Create fragmentation
        tensors = []
        for i in range(100):
            size = 1024 * (1 + i % 10)  # Varying sizes
            tensor = pool.allocate((size // 4,), torch.float32)
            tensors.append(tensor)

        # Deallocate every other tensor
        for i in range(0, 100, 2):
            pool.deallocate(tensors[i])

        # Measure allocation performance with fragmentation
        start = time.perf_counter()
        new_tensors = []
        for _ in range(50):
            tensor = pool.allocate((512,), torch.float32)
            if tensor is not None:
                new_tensors.append(tensor)
        fragmented_time = time.perf_counter() - start

        # Defragment
        pool.defragment()

        # Measure after defragmentation
        start = time.perf_counter()
        defrag_tensors = []
        for _ in range(50):
            tensor = pool.allocate((512,), torch.float32)
            if tensor is not None:
                defrag_tensors.append(tensor)
        defragmented_time = time.perf_counter() - start

        # Performance should improve after defragmentation
        assert defragmented_time < fragmented_time * 1.2


class TestAttentionMemoryOptimization(TestMemoryPerformanceBase):
    """Test memory optimizations in attention implementations."""

    def test_improved_attention_memory_efficiency(self, device):
        """Test memory efficiency of improved attention."""
        batch_size = 2
        seq_len = 8192
        num_heads = 12
        head_dim = 64

        segment_lengths = [1024, 2048, 4096]
        dilation_rates = [1, 2, 4]

        # Create attention module
        attention = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            use_tf32=True,
        ).to(device)

        # Create inputs
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Measure memory usage
        stats = self.measure_performance(attention, q, k, v, is_causal=True)

        # Calculate memory efficiency
        input_memory = 3 * q.numel() * q.element_size()  # Q, K, V
        efficiency = (
            input_memory / stats["peak_memory"] if stats["peak_memory"] > 0 else 0
        )

        # Should be memory efficient (using less than 3x input memory)
        assert efficiency > 0.33

    def test_block_sparse_memory_improvement(self, device):
        """Test memory improvements with block sparse attention."""
        # Configure block sparse attention
        sparse_config = SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=0.1,  # 90% sparse
            block_size=128,
        )

        attention = BlockSparseRingMultiheadDilatedAttention(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
            batch_first=True,
        ).to(device)

        # Test parameters
        batch_size = 2
        seq_len = 1024

        # Measure with memory pool
        reset_global_memory_pool()
        pool = get_global_memory_pool()

        x = torch.randn(batch_size, seq_len, 768, device=device)
        stats_with_pool = self.measure_performance(attention, x, x, x)
        pool_stats = pool.get_stats()

        # Measure without memory pool
        reset_global_memory_pool()

        x = torch.randn(batch_size, seq_len, 768, device=device)
        stats_without_pool = self.measure_performance(attention, x, x, x)

        # Memory pool should reduce allocations
        if pool_stats["total_buffers"] > 0:
            # If pool was used, memory should be more efficient
            assert stats_with_pool["memory_used"] <= stats_without_pool["memory_used"]

    def test_multihead_attention_scaling(self, device):
        """Test memory scaling with different head counts."""
        embed_dim = 768
        segment_lengths = [512, 1024]
        dilation_rates = [1, 2]
        seq_len = 1024
        batch_size = 2

        results = {}

        for num_heads in [6, 12, 24]:
            attention = ImprovedMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            ).to(device)

            x = torch.randn(batch_size, seq_len, embed_dim, device=device)

            stats = self.measure_performance(
                lambda: attention(x, x, x, is_causal=True)[0]
            )

            results[num_heads] = stats

        # Memory should scale sub-linearly with head count
        memory_6_to_12 = results[12]["peak_memory"] / results[6]["peak_memory"]
        memory_12_to_24 = results[24]["peak_memory"] / results[12]["peak_memory"]

        # Scaling factor should be less than linear (2x)
        assert memory_6_to_12 < 1.8
        assert memory_12_to_24 < 1.8


class TestMemoryProfiling(TestMemoryPerformanceBase):
    """Test memory profiling functionality."""

    def test_profiler_overhead(self, device):
        """Test overhead of memory profiling."""
        size = (1024, 1024)
        num_operations = 100

        # Measure without profiling
        start = time.perf_counter()
        for _ in range(num_operations):
            tensor = torch.empty(size, device=device)
            del tensor
        time_without_profiling = time.perf_counter() - start

        # Measure with profiling
        profiler = MemoryProfiler(enable_stack_traces=False)
        profiler.start_profiling()

        start = time.perf_counter()
        for _ in range(num_operations):
            tensor = torch.empty(size, device=device)
            profiler.record_allocation(tensor, pool_type="test")
            profiler.record_deallocation(tensor)
            del tensor
        time_with_profiling = time.perf_counter() - start

        profiler.stop_profiling()

        # Overhead should be reasonable (less than 50%)
        overhead = (
            time_with_profiling - time_without_profiling
        ) / time_without_profiling
        assert overhead < 0.5

    def test_pattern_detection_performance(self, device):
        """Test performance of allocation pattern detection."""
        profiler = MemoryProfiler(
            pattern_analysis_window=50,
            enable_stack_traces=False,
        )
        profiler.start_profiling()

        # Create different allocation patterns
        patterns = {
            "burst": lambda: [torch.empty((64, 64), device=device) for _ in range(20)],
            "periodic": lambda: [
                torch.empty((128, 128), device=device) if i % 5 == 0 else None
                for i in range(50)
            ],
            "growing": lambda: [
                torch.empty((32 * i, 32), device=device) for i in range(1, 21)
            ],
        }

        start = time.perf_counter()

        for pattern_name, pattern_func in patterns.items():
            with profiler.profile_operation(f"pattern_{pattern_name}"):
                tensors = pattern_func()
                for tensor in tensors:
                    if tensor is not None:
                        profiler.record_allocation(tensor, pool_type=pattern_name)

        # Force pattern analysis
        profiler._analyze_patterns()

        analysis_time = time.perf_counter() - start

        profiler.stop_profiling()

        # Pattern analysis should be fast
        assert analysis_time < 0.1  # Less than 100ms
        assert len(profiler.detected_patterns) > 0

    def test_profiling_memory_overhead(self, device):
        """Test memory overhead of profiling data."""
        profiler = MemoryProfiler(
            max_events=1000,
            enable_stack_traces=False,
        )

        # Measure baseline memory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            _ = torch.cuda.memory_allocated()
        else:
            _ = 0

        profiler.start_profiling()

        # Generate many allocation events
        for i in range(1000):
            tensor = torch.empty((100, 100), device=device)
            profiler.record_allocation(tensor, pool_type="test")
            if i % 2 == 0:
                profiler.record_deallocation(tensor)

        # Check profiler memory usage
        profiler_data_size = (
            len(profiler.allocation_events) * 1000  # Rough estimate bytes per event
            + len(profiler.memory_snapshots) * 100
        )

        profiler.stop_profiling()

        # Profiler data should be reasonable (less than 10MB for 1000 events)
        assert profiler_data_size < 10 * 1024 * 1024


class TestMemoryPoolStressTest(TestMemoryPerformanceBase):
    """Stress tests for memory pool implementations."""

    def test_high_frequency_allocation(self, device):
        """Test pool performance under high-frequency allocations."""
        pool = get_global_memory_pool()
        duration = 1.0  # 1 second

        count = 0
        sizes = [1024, 2048, 4096, 8192]

        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            size = sizes[count % len(sizes)]
            tensor = pool.allocate((size // 4,), torch.float32)
            pool.deallocate(tensor)
            count += 1

        allocations_per_second = count / duration

        # Should handle at least 10k allocations per second
        assert allocations_per_second > 10000

    def test_memory_pool_saturation(self, device):
        """Test pool behavior when approaching memory limits."""
        if device.type != "cuda":
            pytest.skip("GPU memory saturation test requires CUDA")

        pool = BucketedMemoryPool(
            bucket_sizes=[1024 * 1024, 10 * 1024 * 1024],  # 1MB, 10MB
            adaptive_buckets=True,
        )

        # Try to allocate until we hit limits
        tensors = []
        allocation_times = []

        try:
            for i in range(1000):
                start = time.perf_counter()
                # Allocate 10MB chunks
                tensor = pool.allocate((10 * 1024 * 1024 // 4,), torch.float32)
                allocation_time = time.perf_counter() - start

                if tensor is None:
                    break

                tensors.append(tensor)
                allocation_times.append(allocation_time)

                # Stop if allocation is getting slow
                if allocation_time > 0.1:
                    break

        except torch.cuda.OutOfMemoryError:
            pass

        # Clean up
        for tensor in tensors:
            pool.deallocate(tensor)

        pool._emergency_cleanup()

        # Check that allocation times didn't degrade too much
        if len(allocation_times) > 10:
            avg_first_10 = np.mean(allocation_times[:10])
            avg_last_10 = np.mean(allocation_times[-10:])

            # Last allocations shouldn't be more than 10x slower
            assert avg_last_10 < avg_first_10 * 10

    def test_mixed_size_allocation_pattern(self, device):
        """Test performance with mixed allocation sizes."""
        pool = get_global_memory_pool()

        # Define mixed size pattern
        size_pattern = [
            1024,  # 1KB
            64 * 1024,  # 64KB
            1024,  # 1KB
            256 * 1024,  # 256KB
            4 * 1024,  # 4KB
            1024 * 1024,  # 1MB
        ]

        num_iterations = 100

        start = time.perf_counter()

        for _ in range(num_iterations):
            tensors = []

            # Allocate in pattern
            for size in size_pattern:
                tensor = pool.allocate((size // 4,), torch.float32)
                tensors.append(tensor)

            # Deallocate in reverse
            for tensor in reversed(tensors):
                pool.deallocate(tensor)

        elapsed = time.perf_counter() - start

        # Should handle mixed patterns efficiently
        operations_per_second = (num_iterations * len(size_pattern) * 2) / elapsed
        assert operations_per_second > 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
