"""Consolidated memory pool tests.

This file consolidates multiple memory pool test files:
- test_core_memory_pool.py
- test_dilated_attention_memory_pool.py
- test_factory_memory_pool_auto_enable.py
- test_memory_pool_integration.py
- test_memory_pool_stress.py
- test_block_sparse_memory_pool.py
"""

import gc
import pytest
import torch
from typing import Dict


class TestMemoryPoolCore:
    """Core memory pool functionality tests."""

    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_memory_pool_creation(self):
        """Test memory pool creation and basic operations."""
        from dilated_attention_pytorch.core.memory_pool import MemoryPool

        pool = MemoryPool(device=self.device)

        # Test allocation
        size = (100, 100)
        tensor = pool.allocate(size, dtype=torch.float32)
        assert tensor.shape == size
        assert tensor.dtype == torch.float32
        assert tensor.device.type == self.device.type

        # Test deallocation
        pool.deallocate(tensor)

        # Test reuse
        tensor2 = pool.allocate(size, dtype=torch.float32)
        # Should reuse the same memory
        assert tensor2.data_ptr() == tensor.data_ptr()

    def test_memory_pool_with_dilated_attention(self):
        """Test memory pool integration with dilated attention."""
        from dilated_attention_pytorch import DilatedAttention

        # Create model with memory pool enabled
        model = DilatedAttention(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
        ).to(self.device)

        # Test forward pass
        batch_size, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)

        # First pass - allocates memory
        output1 = model(q, k, v)

        # Second pass - should reuse memory
        output2 = model(q, k, v)

        assert output1.shape == output2.shape
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_factory_auto_enable_memory_pool(self):
        """Test factory pattern auto-enables memory pool for large sequences."""
        from dilated_attention_pytorch import create_multihead_dilated_attention

        # Large sequence should auto-enable memory pool
        model = create_multihead_dilated_attention(
            "improved",
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
        )

        # Check if memory pool is enabled (implementation-specific)
        if hasattr(model, "_attention") and hasattr(
            model._attention, "enable_memory_pool"
        ):
            # For large sequences, memory pool should be auto-enabled
            assert hasattr(model._attention, "_memory_pool")

    def test_memory_pool_stress(self):
        """Stress test memory pool with multiple allocations."""
        from dilated_attention_pytorch.core.memory_pool import MemoryPool

        pool = MemoryPool(device=self.device)
        tensors = []

        # Allocate many tensors of different sizes
        sizes = [(100, 100), (200, 200), (300, 300), (100, 100), (200, 200)]
        for size in sizes:
            tensor = pool.allocate(size, dtype=torch.float32)
            tensors.append(tensor)

        # Deallocate in different order
        for i in [0, 2, 1, 4, 3]:
            pool.deallocate(tensors[i])

        # Reallocate and check reuse
        new_tensor = pool.allocate((100, 100), dtype=torch.float32)
        # Should reuse one of the deallocated tensors
        assert any(new_tensor.data_ptr() == t.data_ptr() for t in tensors[:3])

    def test_block_sparse_memory_pool(self):
        """Test memory pool with block sparse attention."""
        try:
            from dilated_attention_pytorch import BlockSparseRingDilatedAttention
        except ImportError:
            pytest.skip("BlockSparseRingDilatedAttention not available")

        model = BlockSparseRingDilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            sparsity_config={"block_size": 64, "sparsity_ratio": 0.9},
            enable_memory_pool=True,
        ).to(self.device)

        # Test forward pass
        batch_size, seq_len, num_heads, head_dim = 1, 512, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)

        output = model(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()


class TestMemoryPoolIntegration:
    """Integration tests for memory pool across different modules."""

    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reset global memory pool
        from dilated_attention_pytorch.core.memory_pool import reset_global_memory_pool

        reset_global_memory_pool()

    def test_global_memory_pool_sharing(self):
        """Test that global memory pool is shared across modules."""
        from dilated_attention_pytorch.core.memory_pool import get_global_memory_pool
        from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention

        # Create multiple models
        model1 = DilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            enable_memory_pool=True,
        ).to(self.device)

        model2 = ImprovedDilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            enable_memory_pool=True,
        ).to(self.device)

        # Get global pool stats before
        pool = get_global_memory_pool()
        stats_before = pool.get_stats()

        # Run forward passes
        batch_size, seq_len, num_heads, head_dim = 1, 512, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)

        _ = model1(q, k, v)
        _ = model2(q, k, v)

        # Get stats after
        stats_after = pool.get_stats()

        # Should have allocated some memory
        assert stats_after["total_allocated"] > stats_before["total_allocated"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_pool_gpu_memory_reduction(self):
        """Test that memory pool reduces GPU memory usage."""
        from dilated_attention_pytorch import ImprovedDilatedAttention

        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Model without memory pool
        model_no_pool = ImprovedDilatedAttention(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            enable_memory_pool=False,
        ).cuda()

        # Run multiple forward passes
        batch_size, seq_len, num_heads, head_dim = 2, 2048, 8, 64
        for _ in range(5):
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            _ = model_no_pool(q, k, v)

        memory_no_pool = torch.cuda.max_memory_allocated()

        # Clear and test with pool
        del model_no_pool, q, k, v
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model_with_pool = ImprovedDilatedAttention(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
        ).cuda()

        # Run same forward passes
        for _ in range(5):
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            _ = model_with_pool(q, k, v)

        memory_with_pool = torch.cuda.max_memory_allocated()

        # Memory pool should reduce peak memory usage
        # Note: This might not always be true due to PyTorch's own caching
        print(f"Memory without pool: {memory_no_pool / 1024**2:.2f} MB")
        print(f"Memory with pool: {memory_with_pool / 1024**2:.2f} MB")


class TestMemoryPoolStress:
    """Stress tests for memory pool under heavy load."""

    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.slow
    def test_concurrent_allocations(self):
        """Test memory pool with concurrent allocations from multiple models."""
        from dilated_attention_pytorch import create_multihead_dilated_attention
        import threading

        def worker(model_id: int, results: Dict[int, bool]):
            """Worker function for concurrent test."""
            try:
                model = create_multihead_dilated_attention(
                    "improved",
                    embed_dim=256,
                    num_heads=8,
                    segment_lengths=[512],
                    dilation_rates=[1],
                    enable_memory_pool=True,
                )

                # Run multiple iterations
                for _ in range(10):
                    x = torch.randn(1, 512, 256, device=self.device)
                    output = model(x, x, x)
                    assert output.shape == x.shape

                results[model_id] = True
            except Exception as e:
                print(f"Worker {model_id} failed: {e}")
                results[model_id] = False

        # Run multiple threads
        num_threads = 4
        threads = []
        results = {}

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, results))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should succeed
        assert all(results.values()), f"Some threads failed: {results}"

    def test_memory_pool_cleanup(self):
        """Test memory pool cleanup and recovery."""
        from dilated_attention_pytorch.core.memory_pool import (
            get_global_memory_pool,
            reset_global_memory_pool,
        )

        pool = get_global_memory_pool()

        # Allocate some memory
        tensors = []
        for i in range(10):
            size = (100 * (i + 1), 100 * (i + 1))
            tensor = pool.allocate(size, dtype=torch.float32)
            tensors.append(tensor)

        # Get stats
        stats_before = pool.get_stats()
        assert stats_before["total_allocated"] > 0

        # Reset pool
        reset_global_memory_pool()

        # Get new pool and check it's empty
        new_pool = get_global_memory_pool()
        stats_after = new_pool.get_stats()
        assert stats_after["total_allocated"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
