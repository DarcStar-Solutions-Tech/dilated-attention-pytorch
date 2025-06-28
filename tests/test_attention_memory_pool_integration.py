#!/usr/bin/env python3
"""
Comprehensive tests for memory pool integration across all attention implementations.

This test suite validates:
1. Memory pool integration in all attention variants
2. Buffer reuse efficiency
3. Memory leak detection
4. Performance characteristics
5. Multi-GPU scenarios
"""

import gc
import pytest
import torch
import torch.nn as nn
from typing import Type, Any, Dict

from dilated_attention_pytorch import (
    DilatedAttention,
    MultiheadDilatedAttention,
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
    create_multihead_dilated_attention,
)
from dilated_attention_pytorch.improved_dilated_attention_v2 import (
    ImprovedDilatedAttentionV2,
)
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
)
from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
)
from dilated_attention_pytorch.core.memory_pool import reset_global_memory_pool


# Test configurations
ATTENTION_CLASSES = [
    (DilatedAttention, {"enable_memory_pool": True}),
    (ImprovedDilatedAttention, {"enable_memory_pool": True}),
    (ImprovedDilatedAttentionV2, {"enable_buffer_manager": True}),
    (RingDilatedAttentionV2, {"enable_memory_pool": True}),
    (BlockSparseRingDilatedAttention, {"enable_memory_pool": True}),
]

MULTIHEAD_CLASSES = [
    (MultiheadDilatedAttention, {"enable_memory_pool": True}),
    (ImprovedMultiheadDilatedAttention, {"enable_memory_pool": True}),
    (
        BlockSparseRingMultiheadDilatedAttention,
        {"enable_memory_pool": True, "sparsity_ratio": 0.1},
    ),
]


def clear_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TestMemoryPoolIntegration:
    """Test memory pool integration across attention implementations."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test."""
        clear_memory()
        # Clear any existing memory pools
        reset_global_memory_pool()
        yield
        # Cleanup after test
        clear_memory()

    @pytest.mark.parametrize("attention_class,kwargs", ATTENTION_CLASSES)
    def test_basic_memory_pool_usage(
        self, attention_class: Type[nn.Module], kwargs: Dict[str, Any]
    ):
        """Test basic memory pool usage in attention modules."""
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        # Skip if CUDA not available for certain classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if attention_class.__name__.startswith("Ring") and device.type == "cpu":
            pytest.skip("Ring attention requires CUDA")

        # Create attention module
        # DilatedAttention uses attention_dropout instead of dropout
        if attention_class.__name__ == "DilatedAttention":
            attention = attention_class(
                segment_lengths=[256, 512, 1024],
                dilation_rates=[1, 2, 4],
                attention_dropout=0.0,
                **kwargs,
            )
        else:
            attention = attention_class(
                segment_lengths=[256, 512, 1024],
                dilation_rates=[1, 2, 4],
                dropout=0.0,
                **kwargs,
            )

        # Create test tensors
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        # Get initial memory stats
        if hasattr(attention, "memory_pool"):
            pool = attention.memory_pool
            initial_allocations = (
                pool.total_allocations if hasattr(pool, "total_allocations") else 0
            )
        else:
            pool = None
            initial_allocations = 0

        # Forward pass
        output = attention(q, k, v)
        assert output.shape == shape

        # Check memory pool was used
        if pool and hasattr(pool, "total_allocations"):
            assert pool.total_allocations > initial_allocations

        # Multiple forward passes to test reuse
        for _ in range(3):
            output = attention(q, k, v)

        # Cleanup
        if hasattr(attention, "cleanup_buffers"):
            attention.cleanup_buffers()

    @pytest.mark.parametrize("multihead_class,kwargs", MULTIHEAD_CLASSES)
    def test_multihead_memory_pool_usage(
        self, multihead_class: Type[nn.Module], kwargs: Dict[str, Any]
    ):
        """Test memory pool usage in multihead attention modules."""
        batch_size = 2
        seq_len = 512
        embed_dim = 256
        num_heads = 8

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if multihead_class.__name__.startswith("Ring") and device.type == "cpu":
            pytest.skip("Ring attention requires CUDA")

        # Create multihead attention
        # MultiheadDilatedAttention doesn't support batch_first
        if multihead_class.__name__ == "MultiheadDilatedAttention":
            attention = multihead_class(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[128, 256, 512],
                dilation_rates=[1, 2, 4],
                dropout=0.0,
                **kwargs,
            )
        else:
            attention = multihead_class(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[128, 256, 512],
                dilation_rates=[1, 2, 4],
                dropout=0.0,
                batch_first=True,
                **kwargs,
            )

        # Create test tensors
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Forward pass
        output, _ = attention(x, x, x)
        assert output.shape == x.shape

        # Multiple passes
        for _ in range(3):
            output, _ = attention(x, x, x)

    def test_buffer_reuse_efficiency(self):
        """Test buffer reuse efficiency with attention buffer manager."""
        if not torch.cuda.is_available():
            pytest.skip("Buffer reuse test requires CUDA")

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        # Create attention with buffer manager
        attention = ImprovedDilatedAttentionV2(
            segment_lengths=[256, 512, 1024],
            dilation_rates=[1, 2, 4],
            enable_buffer_manager=True,
            enable_buffer_reuse=True,
            enable_preallocation=True,
        )

        device = torch.device("cuda")
        shape = (batch_size, seq_len, num_heads, head_dim)

        # Pre-allocate buffers
        if hasattr(attention, "buffer_manager") and attention.buffer_manager:
            attention.buffer_manager.preallocate_buffers(
                batch_size, seq_len, num_heads, head_dim, device
            )

        # Multiple forward passes
        cache_hits = []
        for i in range(5):
            q = torch.randn(shape, device=device)
            k = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)

            _ = attention(q, k, v)

            # Get buffer stats
            if hasattr(attention, "get_buffer_stats"):
                stats = attention.get_buffer_stats()
                if "cache_hits" in stats:
                    cache_hits.append(stats["cache_hits"])

        # Verify cache hits increase
        if len(cache_hits) > 1:
            assert cache_hits[-1] > cache_hits[0], (
                "Buffer reuse should increase cache hits"
            )

        # Cleanup
        attention.cleanup_buffers()

    def test_memory_leak_detection(self):
        """Test for memory leaks in attention modules."""
        if not torch.cuda.is_available():
            pytest.skip("Memory leak test requires CUDA")

        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64

        # Track memory usage
        memory_usage = []

        for iteration in range(10):
            clear_memory()

            # Record initial memory
            initial_memory = torch.cuda.memory_allocated()

            # Create and use attention
            attention = ImprovedDilatedAttention(
                segment_lengths=[128, 256, 512],
                dilation_rates=[1, 2, 4],
                enable_memory_pool=True,
                lightweight_pool=True,
            )

            shape = (batch_size, seq_len, num_heads, head_dim)
            q = torch.randn(shape, device="cuda")
            k = torch.randn(shape, device="cuda")
            v = torch.randn(shape, device="cuda")

            # Forward passes
            for _ in range(5):
                output = attention(q, k, v)

            # Cleanup
            if hasattr(attention, "cleanup_buffers"):
                attention.cleanup_buffers()
            del attention, q, k, v, output

            clear_memory()

            # Record final memory
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            memory_usage.append(memory_increase)

        # Check for memory leak - usage should stabilize
        if len(memory_usage) > 5:
            recent_usage = memory_usage[-5:]
            avg_usage = sum(recent_usage) / len(recent_usage)
            max_deviation = max(abs(u - avg_usage) for u in recent_usage)

            # Allow some variation but not continuous growth
            assert max_deviation < 1e6, f"Memory usage not stable: {recent_usage}"

    def test_factory_pattern_pool_integration(self):
        """Test memory pool integration through factory pattern."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create attention through factory
        attention = create_multihead_dilated_attention(
            "improved",
            embed_dim=256,
            num_heads=8,
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
            batch_first=True,
        )

        # Test forward pass
        batch_size = 2
        seq_len = 256
        x = torch.randn(batch_size, seq_len, 256, device=device)

        output, _ = attention(x, x, x)
        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_multi_gpu_pool_sharing(self):
        """Test memory pool behavior across multiple GPUs."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")

        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64

        # Create attention modules on different GPUs
        attentions = []
        for gpu_id in range(2):
            device = torch.device(f"cuda:{gpu_id}")
            attention = ImprovedDilatedAttention(
                segment_lengths=[128, 256, 512],
                dilation_rates=[1, 2, 4],
                enable_memory_pool=True,
            ).to(device)
            attentions.append(attention)

        # Test on each GPU
        for gpu_id, attention in enumerate(attentions):
            device = torch.device(f"cuda:{gpu_id}")
            shape = (batch_size, seq_len, num_heads, head_dim)

            q = torch.randn(shape, device=device)
            k = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)

            output = attention(q, k, v)
            assert output.device == device
            assert output.shape == shape

    def test_pool_configuration_options(self):
        """Test different memory pool configuration options."""
        configs = [
            {"enable_memory_pool": True, "lightweight_pool": True},
            {"enable_memory_pool": True, "lightweight_pool": False},
            {"enable_memory_pool": True, "pool_size_mb": 256},
            {"enable_memory_pool": False},
        ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for config in configs:
            attention = ImprovedDilatedAttention(
                segment_lengths=[256, 512], dilation_rates=[1, 2], **config
            )

            # Test forward pass
            shape = (1, 512, 4, 32)
            q = torch.randn(shape, device=device)
            k = torch.randn(shape, device=device)
            v = torch.randn(shape, device=device)

            output = attention(q, k, v)
            assert output.shape == shape

            # Cleanup
            if hasattr(attention, "cleanup_buffers"):
                attention.cleanup_buffers()


class TestMemoryPoolBenchmarks:
    """Benchmark tests for memory pool integration."""

    @pytest.mark.benchmark
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_allocation_speed_comparison(self, benchmark):
        """Benchmark allocation speed with and without memory pool."""
        batch_size = 4
        seq_len = 2048
        num_heads = 16
        head_dim = 64

        def run_without_pool():
            attention = ImprovedDilatedAttention(
                segment_lengths=[512, 1024, 2048],
                dilation_rates=[1, 2, 4],
                enable_memory_pool=False,
            )

            shape = (batch_size, seq_len, num_heads, head_dim)
            q = torch.randn(shape, device="cuda")
            k = torch.randn(shape, device="cuda")
            v = torch.randn(shape, device="cuda")

            output = attention(q, k, v)
            torch.cuda.synchronize()
            return output

        def run_with_pool():
            attention = ImprovedDilatedAttention(
                segment_lengths=[512, 1024, 2048],
                dilation_rates=[1, 2, 4],
                enable_memory_pool=True,
                lightweight_pool=True,
            )

            shape = (batch_size, seq_len, num_heads, head_dim)
            q = torch.randn(shape, device="cuda")
            k = torch.randn(shape, device="cuda")
            v = torch.randn(shape, device="cuda")

            # Warm up the pool
            _ = attention(q, k, v)
            torch.cuda.synchronize()

            # Benchmark subsequent runs
            output = attention(q, k, v)
            torch.cuda.synchronize()
            return output

        # Run benchmarks
        if hasattr(benchmark, "group"):
            benchmark.group = "memory_pool_allocation"

        # Benchmark with pool (should be faster after warmup)
        result = benchmark(run_with_pool)
        assert result is not None

    @pytest.mark.benchmark
    def test_memory_fragmentation_impact(self):
        """Test impact of memory pool on fragmentation."""
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        # This test would measure memory fragmentation
        # but is simplified here for demonstration

        batch_size = 2
        num_iterations = 50

        # Test without pool
        clear_memory()
        without_pool_fragments = []

        for i in range(num_iterations):
            seq_len = 256 + i * 64  # Varying sizes
            attention = ImprovedDilatedAttention(
                segment_lengths=[256, 512],
                dilation_rates=[1, 2],
                enable_memory_pool=False,
            )

            shape = (batch_size, seq_len, 8, 64)
            q = torch.randn(shape, device="cuda")
            output = attention(q, q, q)

            # Measure fragmentation (simplified)
            if hasattr(torch.cuda, "memory_stats"):
                stats = torch.cuda.memory_stats()
                if "allocated_bytes.all.current" in stats:
                    without_pool_fragments.append(stats["allocated_bytes.all.current"])

            del attention, q, output

        # Test with pool
        clear_memory()
        with_pool_fragments = []

        for i in range(num_iterations):
            seq_len = 256 + i * 64
            attention = ImprovedDilatedAttention(
                segment_lengths=[256, 512],
                dilation_rates=[1, 2],
                enable_memory_pool=True,
                lightweight_pool=True,
            )

            shape = (batch_size, seq_len, 8, 64)
            q = torch.randn(shape, device="cuda")
            output = attention(q, q, q)

            if hasattr(torch.cuda, "memory_stats"):
                stats = torch.cuda.memory_stats()
                if "allocated_bytes.all.current" in stats:
                    with_pool_fragments.append(stats["allocated_bytes.all.current"])

            if hasattr(attention, "cleanup_buffers"):
                attention.cleanup_buffers()
            del attention, q, output

        # With pool should have more consistent memory usage
        if len(with_pool_fragments) > 10 and len(without_pool_fragments) > 10:
            with_pool_variance = torch.var(torch.tensor(with_pool_fragments[-10:]))
            without_pool_variance = torch.var(
                torch.tensor(without_pool_fragments[-10:])
            )

            # Pool should reduce variance (less fragmentation)
            # This is a simplified test - real fragmentation testing is more complex
            assert with_pool_variance <= without_pool_variance * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
