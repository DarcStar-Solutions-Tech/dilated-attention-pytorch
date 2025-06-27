#!/usr/bin/env python3
"""
Stress tests for memory pool implementations.

These tests verify memory pools handle extreme conditions:
- High concurrency
- Memory exhaustion  
- Rapid allocation/deallocation
- Large buffer sizes
- Long-running operations
"""

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch

from dilated_attention_pytorch.core.memory_pool import (
    UnifiedMemoryPool,
    get_global_memory_pool,
    MemoryPoolConfig,
)


class TestMemoryPoolStress:
    """Stress tests for memory pool implementations."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_concurrent_allocation_stress(self, device):
        """Test memory pool under high concurrent allocation load."""
        config = MemoryPoolConfig(max_pool_size_mb=100)
        pool = UnifiedMemoryPool(config=config)
        
        num_threads = 10
        allocations_per_thread = 20
        errors = []
        successful_allocs = []
        
        def allocate_buffers(thread_id: int):
            """Allocate and use buffers concurrently."""
            try:
                thread_buffers = []
                for i in range(allocations_per_thread):
                    # Random buffer sizes
                    size = torch.randint(100, 10000, (1,)).item()
                    buffer = pool.get_buffer(
                        shape=(size,),
                        dtype=torch.float32,
                        device=device,
                    )
                    
                    # Simulate work
                    buffer.fill_(float(thread_id))
                    thread_buffers.append(buffer)
                
                successful_allocs.append((thread_id, len(thread_buffers)))
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Launch concurrent threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(allocate_buffers, i)
                for i in range(num_threads)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # Verify results
        assert len(errors) == 0, f"Errors during concurrent allocation: {errors}"
        assert len(successful_allocs) == num_threads
        total_allocs = sum(count for _, count in successful_allocs)
        assert total_allocs == num_threads * allocations_per_thread

    def test_rapid_allocation_deallocation(self, device):
        """Test rapid allocation and deallocation cycles."""
        config = MemoryPoolConfig(max_pool_size_mb=50)
        pool = UnifiedMemoryPool(config=config)
        
        num_cycles = 100
        buffers_per_cycle = 5
        
        for cycle in range(num_cycles):
            cycle_buffers = []
            
            # Rapid allocation
            for i in range(buffers_per_cycle):
                buffer = pool.get_buffer(
                    shape=(1000, 100),
                    dtype=torch.float16,
                    device=device,
                )
                cycle_buffers.append(buffer)
            
            # Verify buffers are valid
            for buffer in cycle_buffers:
                assert buffer.shape == (1000, 100)
                assert buffer.dtype == torch.float16
            
            # Clear references to allow reuse
            cycle_buffers.clear()
        
        # Pool should maintain reasonable size
        total_buffers = sum(len(p) for p in pool._pools.values())
        assert total_buffers <= 100  # Should reuse buffers

    def test_variable_size_allocation_patterns(self, device):
        """Test allocation patterns with highly variable sizes."""
        pool = get_global_memory_pool()
        
        # Different size categories
        size_patterns = [
            (10,),           # Tiny
            (100, 100),      # Small
            (1000, 1000),    # Medium
            (100, 100, 100), # 3D Medium
            (10, 10, 10, 10),  # 4D Small
        ]
        
        allocations = []
        
        # Allocate in mixed patterns
        for i in range(100):
            pattern_idx = i % len(size_patterns)
            shape = size_patterns[pattern_idx]
            
            buffer = pool.get_buffer(
                shape=shape,
                dtype=torch.float32 if i % 2 == 0 else torch.float16,
                device=device,
            )
            allocations.append((shape, buffer))
        
        # Verify all allocations succeeded
        assert len(allocations) == 100
        
        # Check shape diversity is maintained
        unique_shapes = set(shape for shape, _ in allocations)
        assert len(unique_shapes) == len(size_patterns)

    def test_long_running_stability(self, device):
        """Test memory pool stability over extended operations."""
        config = MemoryPoolConfig(max_pool_size_mb=30)
        pool = UnifiedMemoryPool(config=config)
        
        start_time = time.time()
        duration = 2.0  # 2 seconds of continuous operation
        iteration = 0
        allocation_times = []
        
        while time.time() - start_time < duration:
            iteration += 1
            
            # Allocation phase
            alloc_start = time.time()
            buffers = []
            for i in range(5):
                buffer = pool.get_buffer(
                    shape=(100, 100),
                    dtype=torch.float32,
                    device=device,
                )
                buffers.append(buffer)
            allocation_times.append(time.time() - alloc_start)
            
            # Work simulation
            for buffer in buffers:
                buffer.fill_(iteration)
            
            # Clear references
            buffers.clear()
        
        # Verify stable performance
        avg_alloc_time = sum(allocation_times) / len(allocation_times)
        
        # Times should remain reasonable
        assert avg_alloc_time < 0.1  # 100ms average
        
        # Pool should not have grown unbounded
        total_buffers = sum(len(p) for p in pool._pools.values())
        assert total_buffers <= 50

    def test_dtype_variety_stress(self, device):
        """Test handling of various dtypes under stress."""
        config = MemoryPoolConfig(max_pool_size_mb=40)
        pool = UnifiedMemoryPool(config=config)
        
        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.uint8,
        ]
        
        if device.type == "cuda" and torch.cuda.is_available():
            # Only add bfloat16 if device supports it
            try:
                test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=device)
                dtypes.append(torch.bfloat16)
            except:
                pass
        
        allocations = []
        
        # Allocate buffers of all dtypes
        for i in range(len(dtypes) * 10):
            dtype = dtypes[i % len(dtypes)]
            buffer = pool.get_buffer(
                shape=(100, 100),
                dtype=dtype,
                device=device,
            )
            allocations.append((dtype, buffer))
            
            # Verify dtype is correct
            assert buffer.dtype == dtype
        
        # Check all dtypes were handled
        allocated_dtypes = set(dtype for dtype, _ in allocations)
        assert len(allocated_dtypes) == len(dtypes)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        device = torch.device("cuda:0")
        config = MemoryPoolConfig(max_pool_size_mb=100)
        pool = UnifiedMemoryPool(config=config)
        
        # Allocate large buffers to create memory pressure
        large_buffers = []
        try:
            for i in range(20):
                # Allocate 10MB buffers
                buffer = pool.get_buffer(
                    shape=(2500000,),  # ~10MB with float32
                    dtype=torch.float32,
                    device=device,
                )
                large_buffers.append(buffer)
                
                # Force the buffer to be allocated
                buffer.fill_(1.0)
                
        except torch.cuda.OutOfMemoryError:
            # Expected under memory pressure
            pass
        
        # Check pool handled memory pressure gracefully
        stats = pool.get_stats()
        assert stats["total_buffers"] > 0
        
        # Clear references and cleanup
        large_buffers.clear()
        pool._aggressive_cleanup()
        torch.cuda.empty_cache()

    def test_thread_local_patterns(self, device):
        """Test thread-local buffer usage patterns."""
        pool = get_global_memory_pool()
        
        thread_results = {}
        thread_buffers = {}
        
        def thread_work(thread_id: int):
            """Simulate thread-local buffer usage."""
            local_buffers = []
            
            # Each thread uses same shape repeatedly
            shape = (100 * (thread_id + 1), 100)
            
            for i in range(20):
                buffer = pool.get_buffer(
                    shape=shape,
                    dtype=torch.float32,
                    device=device,
                    pool_type="default",
                )
                local_buffers.append(buffer)
            
            # Store buffer pointers to check reuse
            thread_buffers[thread_id] = [b.data_ptr() for b in local_buffers]
            thread_results[thread_id] = len(set(thread_buffers[thread_id]))
        
        # Run threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_work, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have reused buffers
        for thread_id, unique_buffers in thread_results.items():
            # Should have reused some buffers (not all unique)
            assert unique_buffers < 20, f"Thread {thread_id} didn't reuse buffers"


class TestGlobalMemoryPoolStress:
    """Stress tests for the global memory pool singleton."""

    def test_concurrent_global_access(self):
        """Test concurrent access to global memory pool."""
        pool = get_global_memory_pool()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        errors = []
        
        def access_global_pool(thread_id: int):
            """Access global pool from multiple threads."""
            try:
                for i in range(50):
                    buffer = pool.get_buffer(
                        shape=(100,),
                        dtype=torch.float32,
                        device=device,
                    )
                    buffer.fill_(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Launch many threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=access_global_pool, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Errors in global pool access: {errors}"

    def test_global_pool_cleanup(self):
        """Test global pool cleanup behavior."""
        pool = get_global_memory_pool()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Allocate many buffers
        buffers = []
        for i in range(50):
            buffer = pool.get_buffer(
                shape=(1000, 1000),
                dtype=torch.float32,
                device=device,
            )
            buffers.append(buffer)
        
        # Get initial stats
        initial_stats = pool.get_stats()
        initial_count = initial_stats["total_buffers"]
        
        # Clear references
        buffers.clear()
        gc.collect()
        
        # Aggressive cleanup
        pool._aggressive_cleanup()
        
        # Check buffers were cleaned
        final_stats = pool.get_stats()
        final_count = final_stats["total_buffers"]
        
        # Should have fewer buffers after cleanup
        assert final_count < initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])