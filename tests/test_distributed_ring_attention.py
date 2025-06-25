#!/usr/bin/env python3
"""
Comprehensive tests for distributed Ring Attention implementations.

Tests cover:
- Distributed initialization and setup
- Ring communication patterns
- Memory pool thread safety
- Error recovery mechanisms
- Edge cases and boundary conditions
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import threading
import time
import os
from typing import List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock

from dilated_attention_pytorch.ring_dilated_attention import (
    RingDilatedAttention,
    RingAttentionMemoryPool,
)
from dilated_attention_pytorch.ring_distributed_dilated_attention import (
    RingDistributedDilatedAttention,
)
from dilated_attention_pytorch.block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
    DistributedSparseConfig,
)


class TestRingAttentionMemoryPool:
    """Test memory pool thread safety and limits."""
    
    def test_memory_pool_size_limits(self):
        """Test that memory pool enforces size limits."""
        device = torch.device('cpu')
        pool = RingAttentionMemoryPool(device, max_pool_size=5, max_cache_size=2)
        
        # Fill pool to limit
        buffers = []
        for i in range(10):
            buffer = pool.get_buffer((100,), torch.float32, f"key_{i}")
            buffers.append(buffer)
        
        # Check pool doesn't exceed limit
        assert len(pool._pools) <= 5
        assert len(pool._hot_keys_cache) <= 2
    
    def test_memory_pool_thread_safety(self):
        """Test concurrent access to memory pool."""
        device = torch.device('cpu')
        pool = RingAttentionMemoryPool(device, max_pool_size=10)
        
        errors = []
        buffers_acquired = []
        
        def worker(worker_id: int, iterations: int):
            try:
                for i in range(iterations):
                    # Get buffer
                    buffer = pool.get_buffer(
                        (100, 100), 
                        torch.float32, 
                        f"worker_{worker_id}_iter_{i}"
                    )
                    buffers_acquired.append(buffer)
                    
                    # Simulate work
                    time.sleep(0.001)
                    
                    # Occasionally trigger cleanup
                    if i % 5 == 0:
                        pool.clear_unused_buffers(threshold=1)
                        
            except Exception as e:
                errors.append((worker_id, e))
        
        # Launch multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i, 20))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(buffers_acquired) == 100  # 5 workers * 20 iterations
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        device = torch.device('cpu')
        pool = RingAttentionMemoryPool(device, max_pool_size=3)
        
        # Access buffers in order
        buffer1 = pool.get_buffer((10,), torch.float32, "buffer1")
        buffer2 = pool.get_buffer((20,), torch.float32, "buffer2")
        buffer3 = pool.get_buffer((30,), torch.float32, "buffer3")
        
        # Access buffer1 again (moves to end)
        buffer1_again = pool.get_buffer((10,), torch.float32, "buffer1")
        assert buffer1_again is buffer1  # Should get same buffer
        
        # Add new buffer - should evict buffer2 (LRU)
        buffer4 = pool.get_buffer((40,), torch.float32, "buffer4")
        
        # Check buffer2 was evicted
        assert len(pool._pools) == 3
        assert ((20,), torch.float32, "buffer2", False) not in pool._pools


class TestDistributedRingAttention:
    """Test distributed Ring Attention functionality."""
    
    @pytest.fixture
    def mock_distributed_env(self):
        """Mock distributed environment for testing."""
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=4), \
             patch('torch.distributed.get_rank', return_value=0):
            yield
    
    def test_ring_size_validation(self, mock_distributed_env):
        """Test ring size validation in distributed mode."""
        # Should work with ring_size=4 (matches world_size)
        attention = RingDilatedAttention(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            ring_size=4
        )
        assert attention.ring_size == 4
        
        # Should raise error if ring_size > world_size
        with pytest.raises(ValueError, match="ring_size.*cannot exceed world_size"):
            RingDilatedAttention(
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                ring_size=8
            )
    
    def test_single_gpu_fallback(self):
        """Test behavior when distributed is not initialized."""
        with patch('torch.distributed.is_initialized', return_value=False):
            attention = RingDilatedAttention(
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                ring_size=4  # Should be ignored
            )
            
            # Should fall back to single GPU
            assert attention.ring_size == 1
            assert attention.rank == 0
    
    @pytest.mark.parametrize("error_type", [
        torch.distributed.DistBackendError,
        RuntimeError,
        torch.cuda.OutOfMemoryError,
    ])
    def test_communication_error_handling(self, mock_distributed_env, error_type):
        """Test error handling during ring communication."""
        attention = RingDilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
        )
        
        # Mock communication failure
        with patch.object(attention, '_ring_communicate_kv', side_effect=error_type("Test error")):
            # Create test tensors
            batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim)
            
            # Should raise the error (not silently fail)
            with pytest.raises(error_type):
                attention(q, k, v)


class TestBlockSparseDistributed:
    """Test distributed block-sparse attention."""
    
    def test_distributed_sparse_config_validation(self):
        """Test validation of distributed sparse configuration."""
        # Valid config
        config = DistributedSparseConfig(
            hierarchical_stages=3,
            inter_node_sparsity=0.01,
            gradient_compression_ratio=0.1
        )
        
        # Should work
        attention = BlockSparseRingDistributedDilatedAttention(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            distributed_config=config
        )
        
        # Invalid gradient compression ratio
        with pytest.raises(ValueError, match="gradient_compression_ratio"):
            bad_config = DistributedSparseConfig(
                gradient_compression_ratio=1.5  # > 1.0
            )
            BlockSparseRingDistributedDilatedAttention(
                segment_lengths=[512],
                dilation_rates=[1],
                distributed_config=bad_config
            )


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_forward_error_cleanup(self):
        """Test resource cleanup on forward pass errors."""
        attention = BlockSparseRingDistributedDilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            enable_memory_pool=True
        )
        
        # Track allocated buffers
        initial_pool_size = len(attention.memory_pool.pool) if attention.memory_pool else 0
        
        # Force an error during forward pass
        with patch.object(attention, '_compute_sparse_attention', 
                         side_effect=RuntimeError("Test error")):
            
            batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim)
            
            # Should handle error gracefully
            with pytest.raises(RuntimeError, match="Test error"):
                attention(q, k, v)
        
        # Check buffers were returned to pool
        if attention.memory_pool:
            final_pool_size = len(attention.memory_pool.pool)
            # Pool size should not have grown (buffers returned)
            assert final_pool_size <= initial_pool_size + 1
    
    def test_distributed_error_recovery(self):
        """Test error recovery in distributed operations."""
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=4), \
             patch('torch.distributed.get_rank', return_value=0):
            
            attention = RingDistributedDilatedAttention(
                segment_lengths=[512],
                dilation_rates=[1],
                enable_deepspeed=False  # Avoid DeepSpeed complexity
            )
            
            # Mock communication failure
            mock_handle = MagicMock()
            mock_handle.wait.side_effect = RuntimeError("Communication failed")
            
            with patch('torch.distributed.isend', return_value=mock_handle):
                # Should handle gracefully
                k_block = torch.randn(128, 8, 64)
                v_block = torch.randn(128, 8, 64)
                
                with pytest.raises(RuntimeError, match="Ring communication failed"):
                    attention._ring_communicate_kv(k_block, v_block, ring_step=1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        attention = RingDilatedAttention(
            segment_lengths=[128],
            dilation_rates=[1]
        )
        
        # Zero sequence length should fail validation
        q = torch.randn(2, 0, 8, 64)
        k = torch.randn(2, 0, 8, 64)
        v = torch.randn(2, 0, 8, 64)
        
        with pytest.raises(ValueError, match="Sequence length.*must be divisible"):
            attention(q, k, v)
    
    def test_single_head(self):
        """Test with single attention head."""
        attention = RingDilatedAttention(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2]
        )
        
        # Single head should work
        q = torch.randn(2, 256, 1, 64)
        k = torch.randn(2, 256, 1, 64)
        v = torch.randn(2, 256, 1, 64)
        
        output = attention(q, k, v)
        assert output.shape == (2, 256, 1, 64)
    
    def test_extreme_sequence_lengths(self):
        """Test with very long sequences."""
        attention = RingDilatedAttention(
            segment_lengths=[4096],
            dilation_rates=[1],
            block_size=512
        )
        
        # Long sequence (but still valid)
        batch_size = 1
        seq_len = 16384  # 4x segment length
        num_heads = 4
        head_dim = 32
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        output = attention(q, k, v)
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)


class TestMemoryLimits:
    """Test memory limit enforcement."""
    
    def test_pattern_cache_limits(self):
        """Test pattern cache size limits."""
        from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
            SparsePatternGenerator, SparsePatternConfig
        )
        
        config = SparsePatternConfig(pattern_type='local_window')
        generator = SparsePatternGenerator(config, max_cache_size=3)
        
        # Generate patterns to fill cache
        patterns = []
        for seq_len in [512, 1024, 2048, 4096]:
            pattern = generator.create_pattern(seq_len, num_heads=8)
            patterns.append(pattern)
        
        # Cache should not exceed limit
        assert len(generator.pattern_cache) <= 3
        
        # Oldest patterns should be evicted
        assert (512, 8, 'local_window', config.sparsity_ratio) not in generator.pattern_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDASpecific:
    """Tests specific to CUDA functionality."""
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        device = torch.device('cuda:0')
        pool = RingAttentionMemoryPool(device, max_pool_size=10)
        
        # Allocate large buffers to create memory pressure
        large_buffers = []
        try:
            for i in range(5):
                # Allocate 100MB buffers
                buffer = pool.get_buffer((25_000_000,), torch.float32, f"large_{i}")
                large_buffers.append(buffer)
        except torch.cuda.OutOfMemoryError:
            # Expected under memory pressure
            pass
        
        # Clear unused buffers should be more aggressive under pressure
        pool.clear_unused_buffers()
        
        # Pool should have cleared buffers
        assert len(pool._pools) < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])