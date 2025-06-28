#!/usr/bin/env python3
"""
Edge case and stress tests for memory pool integration in attention modules.

This test suite covers:
1. Extreme sequence lengths
2. Concurrent access patterns
3. Memory pressure scenarios
4. Dynamic shape changes
5. Error recovery
"""

import gc
import pytest
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.improved_dilated_attention_v2 import (
    ImprovedDilatedAttentionV2,
)
from dilated_attention_pytorch.core.attention_buffer_manager import BufferType
from dilated_attention_pytorch.core.memory_pool import reset_global_memory_pool


class TestMemoryPoolEdgeCases:
    """Test edge cases for memory pool integration."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear pools
        reset_global_memory_pool()

        yield

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_extreme_sequence_lengths(self):
        """Test with very long and very short sequences."""
        device = torch.device("cuda")

        # Test configurations
        configs = [
            (1, 16, 4, 32),  # Very short sequence
            (1, 8192, 8, 64),  # Long sequence
            (1, 16384, 4, 32),  # Very long sequence
        ]

        for batch_size, seq_len, num_heads, head_dim in configs:
            # Skip if too large for available memory
            required_memory = (
                batch_size * seq_len * num_heads * head_dim * 4 * 3
            )  # Approximate
            available_memory = torch.cuda.get_device_properties(0).total_memory

            if required_memory > available_memory * 0.8:
                continue

            try:
                # Create attention with memory pool
                attention = ImprovedDilatedAttention(
                    segment_lengths=[min(256, seq_len), min(512, seq_len), seq_len],
                    dilation_rates=[1, 2, 4],
                    enable_memory_pool=True,
                    lightweight_pool=True,
                )

                # Create inputs
                shape = (batch_size, seq_len, num_heads, head_dim)
                q = torch.randn(shape, device=device, dtype=torch.float16)
                k = torch.randn(shape, device=device, dtype=torch.float16)
                v = torch.randn(shape, device=device, dtype=torch.float16)

                # Forward pass
                output = attention(q, k, v)
                assert output.shape == shape

                # Cleanup
                del attention, q, k, v, output
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                # Expected for very large sequences
                torch.cuda.empty_cache()
                continue

    def test_dynamic_shape_changes(self):
        """Test attention with rapidly changing input shapes."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create attention with buffer manager
        attention = ImprovedDilatedAttentionV2(
            segment_lengths=[128, 256, 512],
            dilation_rates=[1, 2, 4],
            enable_buffer_manager=True,
            enable_buffer_reuse=True,
        )

        # Test with varying shapes (all divisible by 512)
        shapes = [
            (2, 512, 8, 64),
            (4, 1024, 8, 64),
            (1, 512, 8, 64),
            (3, 1024, 8, 64),
            (2, 1536, 8, 64),
        ]

        for i, shape in enumerate(shapes * 3):  # Repeat to test reuse
            q = torch.randn(shape, device=device)
            k = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)

            output = attention(q, k, v)
            assert output.shape == shape

        # Check buffer statistics
        if hasattr(attention, "get_buffer_stats"):
            stats = attention.get_buffer_stats()
            # Should have allocated buffers for different shapes
            assert stats.get("cache_misses", 0) > 0
            # With 5 shapes repeated 3 times, we expect 15 allocations
            assert stats["buffer_types"][BufferType.OUTPUT]["allocations"] == 15

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_concurrent_access(self):
        """Test concurrent access to memory pool from multiple threads."""
        device = torch.device("cuda")
        num_threads = 4
        iterations_per_thread = 10

        # Create shared attention module
        attention = ImprovedDilatedAttention(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
            lightweight_pool=True,
        ).to(device)

        # Thread function
        def worker(thread_id: int) -> List[float]:
            times = []
            for i in range(iterations_per_thread):
                shape = (1, 512, 8, 64)
                q = torch.randn(shape, device=device)
                k = torch.randn(shape, device=device)
                v = torch.randn(shape, device=device)

                # Time forward pass
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                _ = attention(q, k, v)
                end.record()

                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            return times

        # Run concurrent threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # Verify all threads completed successfully
        assert len(results) == num_threads
        for thread_times in results:
            assert len(thread_times) == iterations_per_thread

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_memory_pressure_scenario(self):
        """Test behavior under memory pressure."""
        device = torch.device("cuda")

        # Allocate large tensors to create memory pressure
        pressure_tensors = []
        try:
            # Allocate 70% of available memory
            available_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            to_allocate = int(
                (available_memory * 0.7 - allocated_memory) / 4
            )  # float32

            if to_allocate > 0:
                pressure_tensor = torch.zeros(to_allocate, device=device)
                pressure_tensors.append(pressure_tensor)

        except torch.cuda.OutOfMemoryError:
            pass

        # Now try to use attention under memory pressure
        try:
            attention = ImprovedDilatedAttentionV2(
                segment_lengths=[128, 256],
                dilation_rates=[1, 2],
                enable_buffer_manager=True,
                enable_buffer_reuse=True,
            )

            # Small inputs due to memory pressure
            shape = (1, 256, 4, 32)
            q = torch.randn(shape, device=device, dtype=torch.float16)
            k = torch.randn(shape, device=device, dtype=torch.float16)
            v = torch.randn(shape, device=device, dtype=torch.float16)

            # Should still work
            output = attention(q, k, v)
            assert output.shape == shape

            # Buffer manager should adapt
            if hasattr(attention, "get_buffer_stats"):
                stats = attention.get_buffer_stats()
                # May have fewer cached buffers due to pressure
                assert "cached_buffers" in stats

        finally:
            # Release pressure
            pressure_tensors.clear()
            torch.cuda.empty_cache()

    def test_error_recovery(self):
        """Test recovery from allocation failures."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create attention with very large segment lengths
        attention = ImprovedDilatedAttention(
            segment_lengths=[10000, 20000, 40000],  # Unrealistic sizes
            dilation_rates=[1, 2, 4],
            enable_memory_pool=True,
        )

        # Try with inputs that don't match segment requirements
        shape = (1, 100, 4, 32)  # Too small for segments
        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        # Should handle gracefully
        output = attention(q, k, v)
        assert output.shape == shape

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_mixed_precision_pool_usage(self, dtype):
        """Test memory pool with different data types."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Skip float64 on older GPUs
        if dtype == torch.float64 and device.type == "cuda":
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 6:
                pytest.skip("Float64 not well supported on older GPUs")

        attention = ImprovedDilatedAttentionV2(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            enable_buffer_manager=True,
            enable_buffer_reuse=True,
        )

        # Multiple forward passes with same dtype
        shape = (2, 256, 8, 64)
        for i in range(5):
            q = torch.randn(shape, device=device, dtype=dtype)
            k = torch.randn(shape, device=device, dtype=dtype)
            v = torch.randn(shape, device=device, dtype=dtype)

            output = attention(q, k, v)
            assert output.dtype == dtype
            assert output.shape == shape

        # Check buffer reuse
        if hasattr(attention, "get_buffer_stats"):
            stats = attention.get_buffer_stats()
            # Should have good cache hit rate for same dtype
            if i > 0:  # After first iteration
                assert stats.get("cache_hit_rate", 0) > 0.5

    def test_attention_with_masks(self):
        """Test memory pool behavior with attention masks."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64

        attention = ImprovedDilatedAttention(
            segment_lengths=[128, 256, 512],
            dilation_rates=[1, 2, 4],
            enable_memory_pool=True,
        )

        # Create inputs
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        # Test with causal mask
        output_causal = attention(q, k, v, is_causal=True)
        assert output_causal.shape == shape

        # Test with custom attention mask
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.tril(mask)  # Lower triangular
        output_masked = attention(q, k, v, attention_mask=mask)
        assert output_masked.shape == shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_memory_pool_with_gradient_accumulation(self):
        """Test memory pool behavior during gradient accumulation."""
        device = torch.device("cuda")
        accumulation_steps = 4

        attention = ImprovedDilatedAttention(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
            lightweight_pool=True,
        ).to(device)

        # Enable gradient computation
        attention.train()

        # Dummy loss function
        criterion = nn.MSELoss()

        # Gradient accumulation loop
        for step in range(accumulation_steps):
            shape = (1, 256, 8, 64)
            q = torch.randn(shape, device=device, requires_grad=True)
            k = torch.randn(shape, device=device, requires_grad=True)
            v = torch.randn(shape, device=device, requires_grad=True)
            target = torch.randn(shape, device=device)

            # Forward pass
            output = attention(q, k, v)
            loss = criterion(output, target) / accumulation_steps

            # Backward pass (accumulate gradients)
            loss.backward()

        # Memory pool should handle gradient tensors properly
        if hasattr(attention, "memory_pool"):
            pool = attention.memory_pool
            # Pool should still be functional after gradient accumulation
            assert hasattr(pool, "allocate")


class TestBufferManagerStress:
    """Stress tests for attention buffer manager."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_buffer_manager_memory_limits(self):
        """Test buffer manager behavior at memory limits."""
        device = torch.device("cuda")

        # Create attention with aggressive caching
        attention = ImprovedDilatedAttentionV2(
            segment_lengths=[512, 1024, 2048],
            dilation_rates=[1, 2, 4],
            enable_buffer_manager=True,
            enable_buffer_reuse=True,
            enable_preallocation=True,
        )

        # Pre-allocate many different sizes to stress cache
        sizes = [
            (2, 512, 8, 64),
            (4, 1024, 8, 64),
            (1, 2048, 8, 64),
            (3, 768, 8, 64),
            (2, 1536, 8, 64),
        ]

        for size in sizes:
            if hasattr(attention, "buffer_manager") and attention.buffer_manager:
                attention.buffer_manager.preallocate_buffers(
                    size[0], size[1], size[2], size[3], device
                )

        # Now run with different sizes to test cache limits
        for _ in range(20):
            for batch, seq_len, heads, dim in sizes:
                shape = (batch, seq_len, heads, dim)
                q = torch.randn(shape, device=device, dtype=torch.float16)
                k = torch.randn(shape, device=device, dtype=torch.float16)
                v = torch.randn(shape, device=device, dtype=torch.float16)

                output = attention(q, k, v)
                assert output.shape == shape

        # Check that cache is bounded
        if hasattr(attention, "get_buffer_stats"):
            stats = attention.get_buffer_stats()
            # Cache should not grow unbounded
            assert stats.get("cached_buffers", 0) < 100  # Reasonable limit

    def test_buffer_type_consistency(self):
        """Test consistent buffer type usage across forward passes."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        attention = ImprovedDilatedAttentionV2(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            enable_buffer_manager=True,
            enable_buffer_reuse=True,
        )

        shape = (2, 256, 8, 64)

        # Track buffer allocations per type
        buffer_types_used = set()

        for i in range(10):
            q = torch.randn(shape, device=device)
            k = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)

            _ = attention(q, k, v)

            if hasattr(attention, "get_buffer_stats"):
                stats = attention.get_buffer_stats()
                if "buffer_types" in stats:
                    for buffer_type, type_stats in stats["buffer_types"].items():
                        if type_stats.get("allocations", 0) > 0:
                            buffer_types_used.add(buffer_type)

        # Should consistently use same buffer types
        assert len(buffer_types_used) > 0, "No buffer types recorded"
        assert "OUTPUT" in str(buffer_types_used), "Should use OUTPUT buffers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
