"""
Tests for dynamic segment selection functionality.
"""

import pytest
import torch

from dilated_attention_pytorch.utils.dynamic_segment_selector import (
    DynamicSegmentSelector,
    SegmentSelectionConfig,
    MemoryEstimator,
    HardwareAnalyzer,
    SequenceAnalyzer,
)


class TestMemoryEstimator:
    """Test memory estimation functions."""

    def test_segment_memory_estimation(self):
        """Test memory estimation for single segment."""
        # Test with typical configuration
        memory_gb = MemoryEstimator.estimate_segment_memory(
            batch_size=2,
            segment_length=2048,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
        )

        # Should be reasonable (few GB)
        assert 0.1 < memory_gb < 10.0

        # Test scaling with batch size
        memory_2x_batch = MemoryEstimator.estimate_segment_memory(
            batch_size=4,
            segment_length=2048,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
        )
        assert abs(memory_2x_batch - 2 * memory_gb) < 0.1

        # Test scaling with segment length (quadratic due to attention)
        memory_2x_seg = MemoryEstimator.estimate_segment_memory(
            batch_size=2,
            segment_length=4096,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
        )
        # Should be roughly 4x due to quadratic attention
        assert memory_2x_seg > 3 * memory_gb

    def test_total_memory_estimation(self):
        """Test peak memory estimation for multiple segments."""
        segment_lengths = [1024, 2048, 4096]

        peak_memory = MemoryEstimator.estimate_total_memory(
            segment_lengths=segment_lengths,
            batch_size=2,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
        )

        # Peak should be from largest segment
        largest_segment_memory = MemoryEstimator.estimate_segment_memory(
            batch_size=2,
            segment_length=4096,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
        )

        assert abs(peak_memory - largest_segment_memory) < 0.001


class TestHardwareAnalyzer:
    """Test hardware analysis functions."""

    def test_gpu_info(self):
        """Test GPU information retrieval."""
        gpu_info = HardwareAnalyzer.get_gpu_info()

        assert "name" in gpu_info
        assert "compute_capability" in gpu_info
        assert "total_memory_gb" in gpu_info
        assert "available_memory_gb" in gpu_info

        if torch.cuda.is_available():
            assert gpu_info["total_memory_gb"] > 0
            assert gpu_info["compute_capability"][0] >= 3

    def test_optimal_block_size(self):
        """Test optimal block size selection."""
        config = SegmentSelectionConfig()

        # Test known GPU models
        gpu_info_h100 = {"name": "NVIDIA H100", "compute_capability": (9, 0)}
        block_size = HardwareAnalyzer.get_optimal_block_size(gpu_info_h100, config)
        assert block_size == 256

        gpu_info_a100 = {"name": "NVIDIA A100", "compute_capability": (8, 0)}
        block_size = HardwareAnalyzer.get_optimal_block_size(gpu_info_a100, config)
        assert block_size == 128

        gpu_info_v100 = {"name": "Tesla V100", "compute_capability": (7, 0)}
        block_size = HardwareAnalyzer.get_optimal_block_size(gpu_info_v100, config)
        assert block_size == 64


class TestSequenceAnalyzer:
    """Test sequence analysis functions."""

    def test_sequence_distribution_analysis(self):
        """Test basic sequence analysis."""
        # Short sequence
        analysis = SequenceAnalyzer.analyze_sequence_distribution(1024)
        assert analysis["total_length"] == 1024
        assert analysis["is_power_of_2"] is True
        assert analysis["suggested_base_segment"] == 512

        # Medium sequence
        analysis = SequenceAnalyzer.analyze_sequence_distribution(16384)
        assert analysis["suggested_base_segment"] == 2048

        # Long sequence
        analysis = SequenceAnalyzer.analyze_sequence_distribution(65536)
        assert analysis["suggested_base_segment"] == 4096

        # Non-power-of-2
        analysis = SequenceAnalyzer.analyze_sequence_distribution(10000)
        assert analysis["is_power_of_2"] is False


class TestDynamicSegmentSelector:
    """Test the main dynamic segment selector."""

    @pytest.fixture
    def selector(self):
        """Create a selector with test configuration."""
        config = SegmentSelectionConfig(
            min_segment_size=512, max_segment_size=8192, memory_safety_factor=0.8
        )
        return DynamicSegmentSelector(config)

    def test_basic_segment_selection(self, selector):
        """Test basic segment selection."""
        segments, dilation_rates = selector.select_segments(
            sequence_length=8192, batch_size=2, num_heads=8, head_dim=64
        )

        assert len(segments) > 0
        assert len(segments) == len(dilation_rates)
        assert all(s >= selector.config.min_segment_size for s in segments)
        assert all(s <= selector.config.max_segment_size for s in segments)

        # Check divisibility
        assert 8192 % max(segments) == 0

    def test_memory_constrained_selection(self, selector):
        """Test selection with limited memory."""
        # Simulate very limited memory
        selector.config.memory_safety_factor = 0.1
        selector.config.min_free_memory_gb = 0.0

        segments, _ = selector.select_segments(
            sequence_length=32768,
            batch_size=8,  # Large batch
            num_heads=16,
            head_dim=128,
        )

        # Should select smaller segments due to memory constraints
        assert max(segments) <= 4096

    def test_cache_functionality(self, selector):
        """Test caching of segment configurations."""
        # First call
        segments1, _ = selector.select_segments(
            sequence_length=4096, batch_size=2, num_heads=8, head_dim=64
        )

        # Second call with same params should use cache
        segments2, _ = selector.select_segments(
            sequence_length=4096, batch_size=2, num_heads=8, head_dim=64
        )

        assert segments1 == segments2

        # Force refresh should give potentially different result
        segments3, _ = selector.select_segments(
            sequence_length=4096,
            batch_size=2,
            num_heads=8,
            head_dim=64,
            force_refresh=True,
        )

        # Clear cache
        selector.clear_cache()
        assert len(selector._cache) == 0

    def test_power_of_2_preference(self, selector):
        """Test power-of-2 segment sizes."""
        selector.config.prefer_power_of_2 = True

        segments, _ = selector.select_segments(
            sequence_length=10000,  # Non-power-of-2
            batch_size=2,
            num_heads=8,
            head_dim=64,
        )

        # All segments should be powers of 2
        for seg in segments:
            assert (seg & (seg - 1)) == 0  # Check if power of 2

    def test_geometric_progression(self, selector):
        """Test geometric progression of segments."""
        segments, dilation_rates = selector.select_segments(
            sequence_length=16384, batch_size=2, num_heads=8, head_dim=64
        )

        # Check if segments follow some progression (geometric or uniform)
        if len(segments) > 1:
            ratios = [segments[i + 1] / segments[i] for i in range(len(segments) - 1)]
            # Either uniform (all ratios ~1.0) or geometric progression
            is_uniform = all(0.95 <= r <= 1.05 for r in ratios)
            is_geometric = all(1.4 <= r <= 3.1 for r in ratios)
            assert is_uniform or is_geometric

        # Dilation rates should increase
        for i in range(1, len(dilation_rates)):
            assert dilation_rates[i] >= dilation_rates[i - 1]

    def test_edge_cases(self, selector):
        """Test edge cases."""
        # Very short sequence
        segments, _ = selector.select_segments(
            sequence_length=512, batch_size=1, num_heads=4, head_dim=32
        )
        assert len(segments) >= 1
        assert segments[0] == 512

        # Single segment case
        selector.config.max_num_segments = 1
        segments, _ = selector.select_segments(
            sequence_length=4096, batch_size=2, num_heads=8, head_dim=64
        )
        assert len(segments) == 1

    def test_content_boundaries(self, selector):
        """Test with content boundaries."""
        # Simulate natural boundaries in text
        content_boundaries = [1024, 2048, 3072, 4096]

        segments, _ = selector.select_segments(
            sequence_length=4096,
            batch_size=2,
            num_heads=8,
            head_dim=64,
            content_boundaries=content_boundaries,
        )

        # Segments should respect the boundaries somewhat
        assert len(segments) > 0

    def test_utility_functions(self, selector):
        """Test utility functions."""
        # Test alignment
        assert selector._align_to_multiple(100, 64) == 128
        assert selector._align_to_multiple(64, 64) == 64

        # Test nearest power of 2
        assert selector._nearest_power_of_2(100) == 128
        assert selector._nearest_power_of_2(200) == 256
        assert selector._nearest_power_of_2(256) == 256
        assert selector._nearest_power_of_2(257) == 256  # Closer to 256 than 512


class TestIntegration:
    """Integration tests with actual attention modules."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_with_dynamic_attention(self):
        """Test integration with DynamicDilatedAttention."""
        from dilated_attention_pytorch import DynamicDilatedAttention

        # Create dynamic attention
        attention = DynamicDilatedAttention(min_segment_size=512, max_segment_size=4096)

        # Test with different sequence lengths
        for seq_len in [1024, 2048, 4096]:
            q = torch.randn(2, seq_len, 8, 64, device="cuda")
            k = torch.randn(2, seq_len, 8, 64, device="cuda")
            v = torch.randn(2, seq_len, 8, 64, device="cuda")

            output = attention(q, k, v)
            assert output.shape == q.shape

            # Check configuration was set
            segments, rates = attention.get_current_configuration()
            assert len(segments) > 0
            assert seq_len % max(segments) == 0
