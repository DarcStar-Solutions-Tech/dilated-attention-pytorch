#!/usr/bin/env python3
"""
Comprehensive tests for Block-Sparse Ring Attention implementations.

Tests cover functionality, performance, quality, and integration
of all block-sparse attention variants.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Tuple, Optional

# Import implementations to test
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
    SparsePatternGenerator,
    ContentAdaptiveSparsity
)
from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
    FusedQKVProjection,
    create_block_sparse_multihead_attention,
    create_adaptive_sparse_multihead_attention
)
from dilated_attention_pytorch.block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
    DistributedSparseConfig,
    DistributedSparsePattern,
    HierarchicalSparsePatternGenerator
)
from dilated_attention_pytorch.utils.sparse_pattern_utils import (
    PatternType,
    PatternConfig,
    SparsePatternGenerator as UtilsSparsePatternGenerator,
    PatternQualityAnalyzer,
    PatternOptimizer,
    analyze_pattern_statistics
)

# Test configurations
TEST_CONFIGS = {
    'small': {
        'batch_size': 2,
        'seq_len': 1024,
        'num_heads': 4,
        'head_dim': 32,
        'embed_dim': 128
    },
    'medium': {
        'batch_size': 4,
        'seq_len': 4096,
        'num_heads': 8,
        'head_dim': 64,
        'embed_dim': 512
    },
    'large': {
        'batch_size': 2,
        'seq_len': 16384,
        'num_heads': 16,
        'head_dim': 64,
        'embed_dim': 1024
    }
}

SPARSITY_RATIOS = [0.1, 0.25, 0.5, 0.75]
PATTERN_TYPES = ['local_window', 'dilated_sparse', 'global_local']


class TestSparsePatternGeneration:
    """Test sparse pattern generation and utilities"""
    
    @pytest.mark.parametrize("pattern_type", PATTERN_TYPES)
    @pytest.mark.parametrize("sparsity_ratio", [0.25, 0.5])
    def test_pattern_generation(self, pattern_type, sparsity_ratio):
        """Test basic pattern generation"""
        config = SparsePatternConfig(
            pattern_type=pattern_type,
            sparsity_ratio=sparsity_ratio,
            block_size=128
        )
        
        generator = SparsePatternGenerator(config)
        seq_len = 2048
        num_heads = 8
        
        pattern = generator.create_pattern(seq_len, num_heads)
        
        # Check shape
        num_blocks = seq_len // config.block_size
        assert pattern.shape == (num_blocks, num_blocks)
        assert pattern.dtype == torch.bool
        
        # Check sparsity ratio (allow 10% tolerance)
        actual_sparsity = pattern.float().mean().item()
        assert abs(actual_sparsity - sparsity_ratio) < 0.1
        
    def test_pattern_caching(self):
        """Test pattern caching functionality"""
        config = SparsePatternConfig(pattern_type='dilated_sparse', sparsity_ratio=0.25)
        generator = SparsePatternGenerator(config)
        
        seq_len = 1024
        num_heads = 4
        
        # Generate pattern twice
        pattern1 = generator.create_pattern(seq_len, num_heads)
        pattern2 = generator.create_pattern(seq_len, num_heads)
        
        # Should be identical due to caching
        assert torch.equal(pattern1, pattern2)
        
        # Check cache contains the pattern
        assert len(generator.pattern_cache) > 0
        
    def test_adaptive_sparsity_learning(self):
        """Test content-adaptive sparsity learning"""
        head_dim = 64
        block_size = 128
        
        adaptive = ContentAdaptiveSparsity(head_dim, block_size)
        
        # Create test inputs
        batch = 2
        seq_len = 1024
        num_heads = 8
        num_blocks = seq_len // block_size
        
        q = torch.randn(batch, seq_len, num_heads, head_dim)
        k = torch.randn(batch, seq_len, num_heads, head_dim)
        
        # Predict importance
        pattern = adaptive.predict_block_importance(q, k, sparsity_ratio=0.25)
        
        # Check output shape and properties
        assert pattern.shape == (batch, num_heads, num_blocks, num_blocks)
        assert pattern.dtype == torch.bool
        
        # Check sparsity
        actual_sparsity = pattern.float().mean().item()
        assert 0.2 <= actual_sparsity <= 0.3  # Around 25% with tolerance


class TestBlockSparseRingDilatedAttention:
    """Test core block-sparse ring attention implementation"""
    
    @pytest.mark.parametrize("config_name", ['small', 'medium'])
    @pytest.mark.parametrize("sparsity_ratio", [0.25, 0.5])
    def test_forward_pass(self, config_name, sparsity_ratio):
        """Test basic forward pass functionality"""
        config = TEST_CONFIGS[config_name]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sparse_config = SparsePatternConfig(
            pattern_type='dilated_sparse',
            sparsity_ratio=sparsity_ratio,
            block_size=128
        )
        
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
            device=device
        )
        
        # Create test inputs
        batch = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Forward pass
        output = attention(q, k, v)
        
        # Check output shape and properties
        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert output.device == device
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_attention_weights_return(self):
        """Test returning attention weights"""
        config = TEST_CONFIGS['small']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sparse_config = SparsePatternConfig(pattern_type='local_window', sparsity_ratio=0.5)
        
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Forward pass with attention weights
        output, attention_weights = attention(q, k, v, return_attention_weights=True)
        
        # Check shapes
        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert attention_weights.shape == (batch, num_heads, seq_len, seq_len)
        
        # Check attention weights properties
        assert torch.all(attention_weights >= 0)  # Non-negative
        # Note: Sparse attention weights won't sum to 1 for each row
        
    def test_causal_masking(self):
        """Test causal masking functionality"""
        config = TEST_CONFIGS['small']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sparse_config = SparsePatternConfig(pattern_type='local_window', sparsity_ratio=0.3)
        
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            sparse_config=sparse_config,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = 512  # Smaller for easier testing
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Test causal and non-causal
        output_causal = attention(q, k, v, is_causal=True)
        output_non_causal = attention(q, k, v, is_causal=False)
        
        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal, atol=1e-6)
        
    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        config = TEST_CONFIGS['small']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sparse_config = SparsePatternConfig(pattern_type='dilated_sparse', sparsity_ratio=0.25)
        
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[512],
            dilation_rates=[1],
            sparse_config=sparse_config,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Multiple forward passes
        for _ in range(3):
            _ = attention(q, k, v)
            
        # Check performance stats
        stats = attention.get_performance_stats()
        
        assert stats['total_forwards'] >= 3
        assert len(stats['sparse_ratio_history']) >= 3
        assert stats['avg_sparsity'] > 0
        assert stats['avg_speedup'] > 1


class TestBlockSparseRingMultiheadDilatedAttention:
    """Test multihead block-sparse attention implementation"""
    
    @pytest.mark.parametrize("config_name", ['small', 'medium'])
    def test_multihead_forward_pass(self, config_name):
        """Test multihead attention forward pass"""
        config = TEST_CONFIGS[config_name]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        attention = create_block_sparse_multihead_attention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            sparsity_ratio=0.25,
            pattern_type='dilated_sparse',
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        embed_dim = config['embed_dim']
        
        # Test input (batch_first=True by default)
        query = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Self-attention
        output, attention_weights = attention(query, need_weights=True)
        
        # Check output
        assert output.shape == (batch, seq_len, embed_dim)
        assert attention_weights.shape == (batch, seq_len, seq_len)  # Averaged over heads
        assert not torch.isnan(output).any()
        
    def test_multihead_compatibility(self):
        """Test compatibility with nn.MultiheadAttention interface"""
        config = TEST_CONFIGS['small']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create both implementations
        sparse_attention = create_block_sparse_multihead_attention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            sparsity_ratio=0.8,  # High sparsity for more similar results
            device=device
        )
        
        dense_attention = torch.nn.MultiheadAttention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            batch_first=True,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = 512  # Smaller for comparison
        embed_dim = config['embed_dim']
        
        query = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Forward passes
        sparse_output, sparse_weights = sparse_attention(query, need_weights=True)
        dense_output, dense_weights = dense_attention(query, query, query, need_weights=True)
        
        # Shapes should match
        assert sparse_output.shape == dense_output.shape
        assert sparse_weights.shape == dense_weights.shape
        
        # Outputs should be different but in reasonable range
        mse = F.mse_loss(sparse_output, dense_output)
        assert mse < 1.0  # Reasonable approximation
        
    def test_fused_qkv_projection(self):
        """Test fused QKV projection"""
        embed_dim = 256
        num_heads = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        projection = FusedQKVProjection(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device
        )
        
        batch = 2
        seq_len = 1024
        
        x = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Forward pass
        q, k, v = projection(x)
        
        # Check shapes
        head_dim = embed_dim // num_heads
        expected_shape = (batch, seq_len, num_heads, head_dim)
        
        assert q.shape == expected_shape
        assert k.shape == expected_shape
        assert v.shape == expected_shape
        
        # Test output projection
        attention_output = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        output = projection.project_output(attention_output)
        
        assert output.shape == (batch, seq_len, embed_dim)
        
    def test_adaptive_sparse_creation(self):
        """Test adaptive sparse attention creation"""
        config = TEST_CONFIGS['small']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        attention = create_adaptive_sparse_multihead_attention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        embed_dim = config['embed_dim']
        
        query = torch.randn(batch, seq_len, embed_dim, device=device)
        
        # Forward pass
        output, _ = attention(query, need_weights=True)
        
        assert output.shape == (batch, seq_len, embed_dim)
        assert not torch.isnan(output).any()


class TestBlockSparseAdvancedDistributedAttention:
    """Test advanced distributed block-sparse attention"""
    
    def test_hierarchical_pattern_generation(self):
        """Test hierarchical sparse pattern generation"""
        config = DistributedSparseConfig(
            pattern_type=DistributedSparsePattern.HIERARCHICAL,
            sparsity_ratio=0.25,
            local_sparsity=0.4,
            global_sparsity=0.1,
            inter_node_sparsity=0.05
        )
        
        # Mock distributed setup
        world_size = 8
        rank = 0
        
        generator = HierarchicalSparsePatternGenerator(config, world_size, rank)
        
        seq_len = 2048
        num_heads = 8
        
        patterns = generator.create_hierarchical_pattern(seq_len, num_heads)
        
        # Check pattern structure
        assert 'local' in patterns
        assert 'global' in patterns
        assert 'inter_node' in patterns
        
        num_blocks = seq_len // config.block_size
        expected_shape = (num_heads, num_blocks, num_blocks)
        
        for pattern_name, pattern in patterns.items():
            assert pattern.shape == expected_shape
            assert pattern.dtype == torch.bool
            
        # Check sparsity levels (local > global > inter_node)
        local_sparsity = patterns['local'].float().mean().item()
        global_sparsity = patterns['global'].float().mean().item()
        inter_node_sparsity = patterns['inter_node'].float().mean().item()
        
        assert local_sparsity > global_sparsity > inter_node_sparsity
        
    def test_load_balancing(self):
        """Test load balancing functionality"""
        config = DistributedSparseConfig(
            enable_load_balancing=True,
            load_balance_threshold=0.15
        )
        
        world_size = 4
        rank = 0
        
        generator = HierarchicalSparsePatternGenerator(config, world_size, rank)
        
        # Simulate load imbalance
        generator.update_load_stats(
            computation_time=1.5,  # High computation time
            communication_volume=1000000,
            memory_usage=8000000000
        )
        
        seq_len = 1024
        num_heads = 4
        
        patterns = generator.create_hierarchical_pattern(seq_len, num_heads)
        
        # Pattern should be adjusted for load balancing
        assert all(isinstance(pattern, torch.Tensor) for pattern in patterns.values())
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed_attention_forward(self):
        """Test distributed attention forward pass"""
        config = TEST_CONFIGS['small']
        device = torch.device('cuda')
        
        distributed_config = DistributedSparseConfig(
            pattern_type=DistributedSparsePattern.HIERARCHICAL,
            sparsity_ratio=0.25
        )
        
        # Note: This test simulates distributed setup without actual multi-GPU
        attention = BlockSparseRingDistributedDilatedAttention(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            distributed_config=distributed_config,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Forward pass
        output = attention(q, k, v)
        
        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()
        
        # Check performance metrics
        metrics = attention.get_performance_metrics()
        assert metrics['world_size'] >= 1
        assert metrics['rank'] >= 0


class TestSparsePatternUtils:
    """Test sparse pattern utilities"""
    
    @pytest.mark.parametrize("pattern_type", [PatternType.LOCAL_WINDOW, PatternType.DILATED_SPARSE, PatternType.RANDOM])
    def test_utils_pattern_generation(self, pattern_type):
        """Test pattern generation using utilities"""
        config = PatternConfig(
            pattern_type=pattern_type,
            sparsity_ratio=0.3,
            block_size=128
        )
        
        generator = UtilsSparsePatternGenerator(config)
        
        seq_len = 1024
        num_heads = 4
        
        pattern = generator.generate_pattern(seq_len, num_heads)
        
        num_blocks = seq_len // config.block_size
        assert pattern.shape == (num_heads, num_blocks, num_blocks)
        
        # Check sparsity
        actual_sparsity = pattern.float().mean().item()
        assert abs(actual_sparsity - 0.3) < 0.1
        
    def test_pattern_quality_analysis(self):
        """Test pattern quality analysis"""
        analyzer = PatternQualityAnalyzer()
        
        # Create test pattern and reference attention
        num_heads = 4
        num_blocks = 16
        
        sparse_pattern = torch.rand(num_heads, num_blocks, num_blocks) > 0.7
        reference_attention = torch.softmax(torch.randn(1, num_heads, num_blocks*8, num_blocks*8), dim=-1)
        
        metrics = analyzer.analyze_pattern_quality(sparse_pattern, reference_attention)
        
        # Check metrics are reasonable
        assert 0 <= metrics.coverage_ratio <= 1
        assert 0 <= metrics.locality_score <= 1
        assert 0 <= metrics.global_connectivity <= 1
        assert metrics.efficiency_score >= 0
        assert 0 <= metrics.compression_ratio <= 1
        assert metrics.approximation_error >= 0
        
    def test_pattern_optimization(self):
        """Test pattern optimization"""
        optimizer = PatternOptimizer(quality_threshold=0.9)
        
        # Create suboptimal pattern
        num_heads = 2
        num_blocks = 8
        initial_pattern = torch.rand(num_heads, num_blocks, num_blocks) > 0.9  # Very sparse
        
        # Optimize pattern
        optimized_pattern = optimizer.optimize_pattern(initial_pattern, max_iterations=3)
        
        assert optimized_pattern.shape == initial_pattern.shape
        assert optimized_pattern.dtype == torch.bool
        
        # Optimized pattern should generally have higher density
        initial_density = initial_pattern.float().mean()
        optimized_density = optimized_pattern.float().mean()
        assert optimized_density >= initial_density
        
    def test_pattern_statistics(self):
        """Test pattern statistics calculation"""
        # Create test pattern
        pattern = torch.rand(4, 16, 16) > 0.7
        
        stats = pattern_statistics(pattern)
        
        # Check required statistics
        assert 'sparsity_ratio' in stats
        assert 'density_ratio' in stats
        assert 'total_elements' in stats
        assert 'active_elements' in stats
        assert 'avg_row_density' in stats
        assert 'avg_col_density' in stats
        assert 'diagonal_density' in stats
        
        # Check consistency
        assert abs(stats['sparsity_ratio'] + stats['density_ratio'] - 1.0) < 1e-6
        assert stats['total_elements'] == pattern.numel()
        assert stats['active_elements'] == pattern.sum().item()


class TestPerformanceComparison:
    """Performance comparison tests"""
    
    @pytest.mark.parametrize("sparsity_ratio", [0.1, 0.25, 0.5])
    def test_speedup_measurement(self, sparsity_ratio):
        """Test actual speedup measurement"""
        config = TEST_CONFIGS['medium']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create sparse attention
        sparse_config = SparsePatternConfig(
            pattern_type='dilated_sparse',
            sparsity_ratio=sparsity_ratio
        )
        
        sparse_attention = BlockSparseRingDilatedAttention(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Warmup
        for _ in range(3):
            _ = sparse_attention(q, k, v)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Time sparse attention
        start_time = time.time()
        for _ in range(10):
            output = sparse_attention(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sparse_time = time.time() - start_time
        
        # Theoretical speedup
        theoretical_speedup = 1.0 / sparsity_ratio
        
        print(f"Sparsity: {sparsity_ratio:.2f}, "
              f"Time: {sparse_time:.4f}s, "
              f"Theoretical speedup: {theoretical_speedup:.1f}x")
        
        # Basic sanity checks
        assert sparse_time > 0
        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()
        
    def test_memory_usage(self):
        """Test memory usage reduction"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
            
        config = TEST_CONFIGS['large']
        device = torch.device('cuda')
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        sparse_config = SparsePatternConfig(
            pattern_type='dilated_sparse',
            sparsity_ratio=0.1  # Very sparse
        )
        
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
            device=device
        )
        
        batch = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
        
        # Forward pass
        output = attention(q, k, v)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        print(f"Memory used: {memory_used / 1024**2:.1f} MB")
        
        # Should use reasonable memory
        assert memory_used > 0
        assert output.shape == (batch, seq_len, num_heads, head_dim)


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "pattern":
            pytest.main(["-v", "TestSparsePatternGeneration"])
        elif test_category == "basic":
            pytest.main(["-v", "TestBlockSparseRingDilatedAttention"])
        elif test_category == "multihead":
            pytest.main(["-v", "TestBlockSparseRingMultiheadDilatedAttention"])
        elif test_category == "distributed":
            pytest.main(["-v", "TestBlockSparseAdvancedDistributedAttention"])
        elif test_category == "utils":
            pytest.main(["-v", "TestSparsePatternUtils"])
        elif test_category == "performance":
            pytest.main(["-v", "TestPerformanceComparison"])
        else:
            print("Unknown test category. Available: pattern, basic, multihead, distributed, utils, performance")
    else:
        # Run all tests
        pytest.main(["-v", __file__])