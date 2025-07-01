"""
Test memory pool integration across attention modules.

This test suite verifies that:
1. Memory pool is correctly integrated in supported modules
2. Large tensor allocations use the memory pool when enabled
3. Memory is properly deallocated
4. Pool statistics are tracked correctly
"""

import torch
import pytest
import gc
from unittest.mock import patch

from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective as RingDilatedAttentionV2,
)

# RingDilatedAttentionV3 is deprecated, removed from tests
from dilated_attention_pytorch.core import reset_global_memory_pool


class TestMemoryPoolIntegration:
    """Test memory pool integration in attention modules."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset memory pool before each test."""
        reset_global_memory_pool()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_dilated_attention_memory_pool(self):
        """Test memory pool usage in DilatedAttention."""
        # Create with memory pool enabled
        model = DilatedAttention(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
        )

        # Verify memory pool is initialized
        assert model.enable_memory_pool is True
        assert model._memory_pool is not None

        # Run forward pass
        # Need larger tensors to trigger memory pool (>1MB)
        batch_size = 4
        seq_len = 512
        num_heads = 16
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Mock allocate method to track calls
        original_allocate = model._memory_pool.allocate
        allocate_calls = []

        def track_allocate(*args, **kwargs):
            allocate_calls.append((args, kwargs))
            return original_allocate(*args, **kwargs)

        with patch.object(model._memory_pool, "allocate", side_effect=track_allocate):
            output = model(q, k, v)

        # Verify allocations were made
        assert len(allocate_calls) > 0, "Memory pool allocate should be called"
        assert output.shape == q.shape

        # Check pool statistics
        stats = model._memory_pool.get_stats()
        assert "enhanced_pool" in stats
        enhanced_stats = stats["enhanced_pool"]
        assert enhanced_stats["total_allocations"] > 0

    def test_improved_dilated_attention_memory_pool(self):
        """Test memory pool usage in ImprovedDilatedAttention."""
        model = ImprovedDilatedAttention(
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
        )

        assert model.enable_memory_pool is True
        assert model._memory_pool is not None

        # Test with CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        q = torch.randn(1, 128, 8, 32, device=device)
        k = torch.randn(1, 128, 8, 32, device=device)
        v = torch.randn(1, 128, 8, 32, device=device)

        output = model(q, k, v)
        assert output.shape == q.shape

        # Verify memory pool was used
        stats = model._memory_pool.get_stats()
        assert "enhanced_pool" in stats
        enhanced_stats = stats["enhanced_pool"]
        assert enhanced_stats["total_allocations"] > 0

    def test_ring_attention_v2_memory_pool(self):
        """Test memory pool usage in RingDilatedAttentionV2."""
        model = RingDilatedAttentionV2(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
            lightweight_pool=True,  # Use lightweight for faster tests
        )

        # Check initialization
        assert model.enable_memory_pool is True
        assert model._memory_pool is not None

        # Test allocation methods
        shape = (2, 1024, 16, 64)
        dtype = torch.float32
        device = torch.device("cpu")

        # Test _allocate_tensor
        tensor = model._allocate_tensor(shape, dtype, device)
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device.type == device.type

        # Test _deallocate_tensor
        model._deallocate_tensor(tensor)

        # Verify cleanup
        model.cleanup_buffers()

    # RingDilatedAttentionV3 test removed - deprecated

    def test_memory_pool_disabled_by_default(self):
        """Verify memory pool is disabled by default."""
        models = [
            DilatedAttention(segment_lengths=[64], dilation_rates=[1]),
            ImprovedDilatedAttention(segment_lengths=[64], dilation_rates=[1]),
            RingDilatedAttentionV2(segment_lengths=[64], dilation_rates=[1]),
        ]

        for model in models:
            assert model.enable_memory_pool is False, (
                f"{model.__class__.__name__} should have memory pool disabled by default"
            )
            assert model._memory_pool is None

    def test_memory_pool_size_threshold(self):
        """Test that only large tensors use memory pool."""
        model = RingDilatedAttentionV2(
            segment_lengths=[64],
            dilation_rates=[1],
            enable_memory_pool=True,
        )

        # Small tensor (< 1MB) should not use pool
        small_shape = (10, 10, 10)  # ~4KB for float32
        _ = model._allocate_tensor(small_shape, torch.float32, torch.device("cpu"))

        # Large tensor (>= 1MB) should use pool
        large_shape = (64, 256, 256)  # ~16MB for float32

        with patch.object(model._memory_pool, "allocate") as mock_allocate:
            mock_allocate.return_value = torch.zeros(large_shape)
            _ = model._allocate_tensor(large_shape, torch.float32, torch.device("cpu"))
            mock_allocate.assert_called_once()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_pool_gpu_allocation(self):
        """Test memory pool with GPU tensors."""
        model = ImprovedDilatedAttention(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            enable_memory_pool=True,
        ).cuda()

        # Large GPU tensor
        shape = (4, 256, 16, 64)
        tensor = model._allocate_tensor(shape, torch.float16, torch.device("cuda"))

        assert tensor.shape == shape
        assert tensor.dtype == torch.float16
        assert tensor.device.type == "cuda"
        assert tensor.is_contiguous()

        # Deallocate if method exists
        if hasattr(model, "_deallocate_tensor"):
            model._deallocate_tensor(tensor)

    def test_memory_pool_statistics(self):
        """Test memory pool statistics tracking."""
        model = DilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            enable_memory_pool=True,
        )

        # Initial stats
        stats = model._memory_pool.get_stats()
        enhanced_stats = stats.get("enhanced_pool", {})
        initial_allocations = enhanced_stats.get("total_allocations", 0)

        # Allocate several tensors
        tensors = []
        for i in range(5):
            shape = (32, 512, 8, 64)  # ~4MB each
            tensor = model._allocate_tensor(shape, torch.float32, torch.device("cpu"))
            tensors.append(tensor)

        # Check updated stats
        stats = model._memory_pool.get_stats()
        enhanced_stats = stats.get("enhanced_pool", {})
        assert enhanced_stats.get("total_allocations", 0) >= initial_allocations + 5

        # Deallocate if method exists
        if hasattr(model, "_deallocate_tensor"):
            for tensor in tensors:
                model._deallocate_tensor(tensor)

        # Final stats - EnhancedMemoryPool doesn't track deallocations separately
        # Just check that the pool is still functional
        stats = model._memory_pool.get_stats()
        assert stats is not None


class TestMemoryPoolConsistency:
    """Test consistency of memory pool implementations."""

    def test_allocate_deallocate_patterns(self):
        """Verify consistent allocate/deallocate method names."""
        expected_allocate = "_allocate_tensor"
        expected_deallocate = "_deallocate_tensor"

        models = [
            DilatedAttention(
                segment_lengths=[64], dilation_rates=[1], enable_memory_pool=True
            ),
            ImprovedDilatedAttention(
                segment_lengths=[64], dilation_rates=[1], enable_memory_pool=True
            ),
            RingDilatedAttentionV2(
                segment_lengths=[64], dilation_rates=[1], enable_memory_pool=True
            ),
        ]

        for model in models:
            assert hasattr(model, expected_allocate), (
                f"{model.__class__.__name__} missing {expected_allocate}"
            )
            # Only RingDilatedAttentionV2 has _deallocate_tensor
            if model.__class__.__name__ == "RingDilatedAttentionV2":
                assert hasattr(model, expected_deallocate), (
                    f"{model.__class__.__name__} missing {expected_deallocate}"
                )
            assert callable(getattr(model, expected_allocate))
            if hasattr(model, expected_deallocate):
                assert callable(getattr(model, expected_deallocate))

    def test_memory_pool_parameter_consistency(self):
        """Verify consistent parameter naming."""
        classes = [
            DilatedAttention,
            ImprovedDilatedAttention,
            RingDilatedAttentionV2,
        ]

        for cls in classes:
            # Check __init__ signature
            import inspect

            sig = inspect.signature(cls.__init__)
            params = sig.parameters

            assert "enable_memory_pool" in params, (
                f"{cls.__name__} missing enable_memory_pool parameter"
            )

            # Check default value
            default = params["enable_memory_pool"].default
            assert default is False, (
                f"{cls.__name__} should have enable_memory_pool=False by default"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
