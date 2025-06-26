#!/usr/bin/env python3
"""
Thread safety tests for dilated attention implementations.

Tests concurrent access patterns, race conditions, and synchronization.
"""

import random
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseMemoryPool,
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
    SparsePatternGenerator,
)
from dilated_attention_pytorch.ring_dilated_attention import (
    RingAttentionMemoryPool,
    RingDilatedAttention,
)


class ThreadSafetyTester:
    """Utility class for thread safety testing."""

    def __init__(self):
        self.errors: list[Exception] = []
        self.results: dict[int, Any] = {}
        self.lock = threading.Lock()

    def record_error(self, thread_id: int, error: Exception):
        """Thread-safe error recording."""
        with self.lock:
            self.errors.append((thread_id, error))

    def record_result(self, thread_id: int, result: Any):
        """Thread-safe result recording."""
        with self.lock:
            self.results[thread_id] = result

    def run_concurrent_test(self, func: Callable, num_threads: int, *args, **kwargs):
        """Run function concurrently in multiple threads."""
        threads = []

        for i in range(num_threads):

            def worker(thread_id=i):
                try:
                    result = func(thread_id, *args, **kwargs)
                    self.record_result(thread_id, result)
                except Exception as e:
                    self.record_error(thread_id, e)

            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        return self.results, self.errors


class TestMemoryPoolThreadSafety:
    """Test thread safety of memory pools."""

    def test_concurrent_buffer_allocation(self):
        """Test concurrent buffer allocation and deallocation."""
        device = torch.device("cpu")
        pool = RingAttentionMemoryPool(device, max_pool_size=50)
        tester = ThreadSafetyTester()

        def allocate_and_release(thread_id: int):
            """Allocate and release buffers randomly."""
            allocated = []

            for i in range(20):
                # Randomly allocate or release
                if random.random() > 0.3 or not allocated:
                    # Allocate
                    shape = (random.randint(10, 100), random.randint(10, 100))
                    buffer = pool.get_buffer(shape, torch.float32, f"thread_{thread_id}_buf_{i}")
                    allocated.append(buffer)
                # Simulate releasing by clearing
                elif random.random() > 0.5:
                    pool.clear_unused_buffers(threshold=random.randint(1, 10))

                # Small delay to increase chance of race conditions
                time.sleep(0.0001)

            return len(allocated)

        results, errors = tester.run_concurrent_test(allocate_and_release, num_threads=10)

        # Should have no errors
        assert len(errors) == 0, f"Thread safety violations: {errors}"

        # All threads should have completed successfully
        assert len(results) == 10

        # Pool should still be in valid state
        assert len(pool._pools) <= pool.max_pool_size

    def test_hot_cache_concurrent_access(self):
        """Test concurrent access to hot cache."""
        device = torch.device("cpu")
        pool = RingAttentionMemoryPool(device, max_pool_size=20, max_cache_size=5)
        tester = ThreadSafetyTester()

        def access_hot_patterns(thread_id: int):
            """Access same keys to trigger hot cache."""
            buffers = []

            # All threads access same set of keys
            for i in range(5):
                for j in range(10):
                    key = f"hot_key_{i}"  # Same key across threads
                    buffer = pool.get_buffer((50, 50), torch.float32, key)
                    buffers.append(buffer)

                    # Occasionally access unique key
                    if random.random() > 0.8:
                        unique_key = f"thread_{thread_id}_unique_{j}"
                        buffer = pool.get_buffer((30, 30), torch.float32, unique_key)
                        buffers.append(buffer)

            return len(buffers)

        results, errors = tester.run_concurrent_test(access_hot_patterns, num_threads=8)

        assert len(errors) == 0
        assert len(pool._hot_keys_cache) <= pool.max_cache_size

    def test_lru_eviction_race_condition(self):
        """Test LRU eviction under concurrent access."""
        device = torch.device("cpu")
        pool = RingAttentionMemoryPool(device, max_pool_size=5)
        tester = ThreadSafetyTester()

        def compete_for_buffers(thread_id: int):
            """Multiple threads compete for limited buffer slots."""
            successes = 0

            for i in range(50):
                try:
                    # Each thread uses unique keys
                    key = f"thread_{thread_id}_iter_{i}"
                    buffer = pool.get_buffer((100,), torch.float32, key)

                    # Simulate some work
                    buffer.fill_(thread_id)

                    # Verify buffer wasn't corrupted
                    if buffer.sum().item() == thread_id * 100:
                        successes += 1

                    # Random delay
                    time.sleep(random.uniform(0.0001, 0.001))

                except Exception:
                    # Eviction might cause issues
                    pass

            return successes

        results, errors = tester.run_concurrent_test(compete_for_buffers, num_threads=10)

        # Should handle eviction gracefully
        assert len(errors) == 0
        assert all(successes > 0 for successes in results.values())


class TestPatternGeneratorThreadSafety:
    """Test thread safety of sparse pattern generation."""

    def test_concurrent_pattern_generation(self):
        """Test concurrent pattern generation and caching."""
        config = SparsePatternConfig(pattern_type="dilated_sparse")
        generator = SparsePatternGenerator(config, max_cache_size=10)
        tester = ThreadSafetyTester()

        def generate_patterns(thread_id: int):
            """Generate patterns with overlapping parameters."""
            patterns = []

            # Mix of unique and shared parameters
            seq_lengths = [512, 1024, 2048] if thread_id % 2 == 0 else [768, 1536, 3072]

            for _ in range(10):
                seq_len = random.choice(seq_lengths)
                num_heads = random.choice([4, 8, 12])

                pattern = generator.create_pattern(seq_len, num_heads)
                patterns.append(pattern)

                # Verify pattern is valid
                assert pattern.dtype == torch.bool
                assert pattern.dim() == 2

            return len(patterns)

        results, errors = tester.run_concurrent_test(generate_patterns, num_threads=6)

        assert len(errors) == 0
        assert len(generator.pattern_cache) <= generator.max_cache_size

        # Verify cache entries are valid
        for key, pattern in generator.pattern_cache.items():
            assert isinstance(pattern, torch.Tensor)
            assert pattern.dtype == torch.bool


class TestAttentionModuleThreadSafety:
    """Test thread safety of attention modules."""

    def test_concurrent_forward_passes(self):
        """Test concurrent forward passes through attention."""
        attention = RingDilatedAttention(
            segment_lengths=[256, 512], dilation_rates=[1, 2], block_size=128
        )
        tester = ThreadSafetyTester()

        def run_forward(thread_id: int):
            """Run forward passes with different batch sizes."""
            outputs = []

            for i in range(5):
                batch_size = thread_id % 3 + 1
                seq_len = 512
                num_heads = 8
                head_dim = 64

                q = torch.randn(batch_size, seq_len, num_heads, head_dim)
                k = torch.randn(batch_size, seq_len, num_heads, head_dim)
                v = torch.randn(batch_size, seq_len, num_heads, head_dim)

                output = attention(q, k, v)
                outputs.append(output)

                # Verify output shape
                assert output.shape == q.shape

            return len(outputs)

        results, errors = tester.run_concurrent_test(run_forward, num_threads=4)

        assert len(errors) == 0
        assert all(count == 5 for count in results.values())

    def test_memory_pool_sharing(self):
        """Test multiple attention modules sharing memory pool."""
        device = torch.device("cpu")
        shared_pool = RingAttentionMemoryPool(device, max_pool_size=30)

        # Create multiple attention modules sharing the pool
        attentions = []
        for i in range(3):
            attention = BlockSparseRingDilatedAttention(
                segment_lengths=[256], dilation_rates=[1], enable_memory_pool=True
            )
            # Inject shared pool
            attention.memory_pool = BlockSparseMemoryPool()
            attention._memory_pool = shared_pool
            attentions.append(attention)

        tester = ThreadSafetyTester()

        def use_attention(thread_id: int):
            """Different threads use different attention instances."""
            attention = attentions[thread_id % len(attentions)]

            for i in range(10):
                q = torch.randn(1, 256, 4, 32)
                k = torch.randn(1, 256, 4, 32)
                v = torch.randn(1, 256, 4, 32)

                output = attention(q, k, v)
                assert output is not None

            return True

        results, errors = tester.run_concurrent_test(use_attention, num_threads=9)

        assert len(errors) == 0


class TestWeakReferenceCleanup:
    """Test cleanup with weak references."""

    def test_memory_pool_cleanup(self):
        """Test memory pool cleanup when attention modules are deleted."""
        device = torch.device("cpu")
        pool = RingAttentionMemoryPool(device)

        # Track pool with weak reference
        pool_ref = weakref.ref(pool)

        # Create and delete attention module
        attention = RingDilatedAttention(segment_lengths=[128], dilation_rates=[1])
        attention._memory_pool = pool

        # Use the attention to allocate buffers
        q = torch.randn(1, 128, 4, 32)
        k = torch.randn(1, 128, 4, 32)
        v = torch.randn(1, 128, 4, 32)
        _ = attention(q, k, v)

        # Delete attention
        del attention
        del pool

        # Force garbage collection
        import gc

        gc.collect()

        # Pool should be cleaned up
        assert pool_ref() is None


class TestDeadlockPrevention:
    """Test prevention of deadlocks."""

    def test_no_deadlock_on_nested_locks(self):
        """Test that nested locking doesn't cause deadlock."""
        device = torch.device("cpu")
        pool = RingAttentionMemoryPool(device)

        def thread1():
            """Thread 1: Get buffer then clear."""
            for _ in range(100):
                buffer = pool.get_buffer((50, 50), torch.float32, "key1")
                pool.clear_unused_buffers()
                time.sleep(0.0001)

        def thread2():
            """Thread 2: Clear then get buffer."""
            for _ in range(100):
                pool.clear_unused_buffers()
                buffer = pool.get_buffer((60, 60), torch.float32, "key2")
                time.sleep(0.0001)

        # Run with timeout to detect deadlock
        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)

        t1.start()
        t2.start()

        # Wait with timeout
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # Threads should complete (not deadlock)
        assert not t1.is_alive()
        assert not t2.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
