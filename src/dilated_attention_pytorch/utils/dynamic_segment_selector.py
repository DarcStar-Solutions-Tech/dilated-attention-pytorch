"""
Dynamic Segment Selector for Dilated Attention.

This module provides intelligent selection of segment lengths based on:
- Available GPU memory
- Sequence characteristics
- Hardware capabilities
- Runtime conditions
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch


@dataclass
class SegmentSelectionConfig:
    """Configuration for dynamic segment selection."""

    # Segment size bounds
    min_segment_size: int = 512
    max_segment_size: int = 65536

    # Memory thresholds
    memory_safety_factor: float = 0.8  # Use only 80% of available memory
    min_free_memory_gb: float = 0.5  # Keep at least 500MB free

    # Segment generation parameters
    base_segment_size: int = 2048
    geometric_ratio: float = 2.0
    max_num_segments: int = 5

    # Hardware-specific settings
    prefer_power_of_2: bool = True
    align_to_block_size: bool = True
    gpu_block_sizes: Dict[str, int] = None

    def __post_init__(self):
        if self.gpu_block_sizes is None:
            self.gpu_block_sizes = {
                "H100": 256,
                "H200": 256,
                "A100": 128,
                "A10": 128,
                "V100": 64,
                "default": 64,
            }


class MemoryEstimator:
    """Estimates memory requirements for different segment configurations."""

    @staticmethod
    def estimate_segment_memory(
        batch_size: int,
        segment_length: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """
        Estimate memory usage for processing a single segment.

        Returns:
            Memory usage in GB
        """
        # Bytes per element
        bytes_per_element = torch.finfo(dtype).bits // 8

        # Memory for Q, K, V tensors (3x for each)
        qkv_memory = (
            3 * batch_size * segment_length * num_heads * head_dim * bytes_per_element
        )

        # Memory for attention scores
        scores_memory = (
            batch_size * num_heads * segment_length * segment_length * bytes_per_element
        )

        # Memory for output
        output_memory = (
            batch_size * segment_length * num_heads * head_dim * bytes_per_element
        )

        # Add overhead for temporary buffers (20%)
        total_memory = (qkv_memory + scores_memory + output_memory) * 1.2

        return total_memory / 1e9  # Convert to GB

    @staticmethod
    def estimate_total_memory(
        segment_lengths: List[int],
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """Estimate peak memory for all segments."""
        # Peak memory is the maximum across all segments
        # (since segments are processed sequentially)
        peak_memory = 0
        for seg_len in segment_lengths:
            seg_memory = MemoryEstimator.estimate_segment_memory(
                batch_size, seg_len, num_heads, head_dim, dtype
            )
            peak_memory = max(peak_memory, seg_memory)

        return peak_memory


class HardwareAnalyzer:
    """Analyzes hardware capabilities for optimal segment selection."""

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get current GPU information."""
        if not torch.cuda.is_available():
            return {
                "name": "CPU",
                "compute_capability": (0, 0),
                "total_memory_gb": 0,
                "available_memory_gb": 0,
            }

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        # Get memory info
        total_memory = props.total_memory / 1e9
        free_memory, _ = torch.cuda.mem_get_info(device)
        available_memory = free_memory / 1e9

        return {
            "name": props.name,
            "compute_capability": (props.major, props.minor),
            "total_memory_gb": total_memory,
            "available_memory_gb": available_memory,
            "multiprocessor_count": props.multi_processor_count,
        }

    @staticmethod
    def get_optimal_block_size(
        gpu_info: Dict[str, Any], config: SegmentSelectionConfig
    ) -> int:
        """Determine optimal block size for the GPU."""
        gpu_name = gpu_info["name"]

        # Check known GPU models
        for model, block_size in config.gpu_block_sizes.items():
            if model in gpu_name:
                return block_size

        # Fallback based on compute capability
        major, minor = gpu_info["compute_capability"]
        if major >= 9:  # Hopper and newer
            return 256
        elif major >= 8:  # Ampere
            return 128
        elif major >= 7:  # Volta/Turing
            return 64
        else:
            return 32


class SequenceAnalyzer:
    """Analyzes sequence characteristics for segment selection."""

    @staticmethod
    def analyze_sequence_distribution(
        sequence_length: int, content_boundaries: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze sequence characteristics.

        Args:
            sequence_length: Total sequence length
            content_boundaries: Optional list of natural boundaries in the sequence

        Returns:
            Analysis results including suggested segment sizes
        """
        analysis = {
            "total_length": sequence_length,
            "is_power_of_2": (sequence_length & (sequence_length - 1)) == 0,
            "suggested_base_segment": None,
            "natural_boundaries": content_boundaries or [],
        }

        # Determine suggested base segment size
        if sequence_length <= 2048:
            analysis["suggested_base_segment"] = min(512, sequence_length)
        elif sequence_length <= 8192:
            analysis["suggested_base_segment"] = 1024
        elif sequence_length <= 32768:
            analysis["suggested_base_segment"] = 2048
        else:
            analysis["suggested_base_segment"] = 4096

        return analysis


class DynamicSegmentSelector:
    """Main class for dynamic segment selection."""

    def __init__(self, config: Optional[SegmentSelectionConfig] = None):
        self.config = config or SegmentSelectionConfig()
        self.memory_estimator = MemoryEstimator()
        self.hardware_analyzer = HardwareAnalyzer()
        self.sequence_analyzer = SequenceAnalyzer()

        # Cache for common configurations
        self._cache: Dict[Tuple[int, int, int], List[int]] = {}

    def select_segments(
        self,
        sequence_length: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        content_boundaries: Optional[List[int]] = None,
        force_refresh: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """
        Select optimal segment lengths and corresponding dilation rates.

        Args:
            sequence_length: Total sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: Data type for computation
            content_boundaries: Optional natural boundaries in sequence
            force_refresh: Force recomputation instead of using cache

        Returns:
            Tuple of (segment_lengths, dilation_rates)
        """
        # Check cache first
        cache_key = (sequence_length, batch_size, num_heads)
        if not force_refresh and cache_key in self._cache:
            segment_lengths = self._cache[cache_key]
            return segment_lengths, self._compute_dilation_rates(segment_lengths)

        # Get hardware info
        gpu_info = self.hardware_analyzer.get_gpu_info()
        available_memory = gpu_info["available_memory_gb"]

        # Apply safety factor
        safe_memory = available_memory * self.config.memory_safety_factor
        safe_memory = max(0, safe_memory - self.config.min_free_memory_gb)

        # Analyze sequence
        seq_analysis = self.sequence_analyzer.analyze_sequence_distribution(
            sequence_length, content_boundaries
        )

        # Select segments based on multiple factors
        segment_lengths = self._select_segments_multi_factor(
            sequence_length=sequence_length,
            safe_memory_gb=safe_memory,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            gpu_info=gpu_info,
            seq_analysis=seq_analysis,
        )

        # Cache the result
        self._cache[cache_key] = segment_lengths

        # Compute corresponding dilation rates
        dilation_rates = self._compute_dilation_rates(segment_lengths)

        return segment_lengths, dilation_rates

    def _select_segments_multi_factor(
        self,
        sequence_length: int,
        safe_memory_gb: float,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        gpu_info: Dict[str, Any],
        seq_analysis: Dict[str, Any],
    ) -> List[int]:
        """Select segments considering multiple factors."""
        # Start with base segment from sequence analysis
        base_segment = (
            seq_analysis["suggested_base_segment"] or self.config.base_segment_size
        )

        # Align to GPU block size if requested
        if self.config.align_to_block_size:
            block_size = self.hardware_analyzer.get_optimal_block_size(
                gpu_info, self.config
            )
            base_segment = self._align_to_multiple(base_segment, block_size)

        # Ensure power of 2 if requested
        if self.config.prefer_power_of_2:
            base_segment = self._nearest_power_of_2(base_segment)

        # Generate candidate segment sequences
        candidates = self._generate_candidate_sequences(base_segment, sequence_length)

        # Select best candidate based on memory constraints
        best_segments = None
        best_score = float("-inf")

        for candidate in candidates:
            # Check if it fits in memory
            peak_memory = self.memory_estimator.estimate_total_memory(
                candidate, batch_size, num_heads, head_dim, dtype
            )

            if peak_memory <= safe_memory_gb:
                # Score based on coverage and efficiency
                score = self._score_segment_sequence(candidate, sequence_length)
                if score > best_score:
                    best_score = score
                    best_segments = candidate

        # Fallback if no candidate fits
        if best_segments is None:
            # Use minimal configuration
            min_segment = min(self.config.min_segment_size, sequence_length)
            best_segments = [min_segment]

        return best_segments

    def _generate_candidate_sequences(
        self, base_segment: int, sequence_length: int
    ) -> List[List[int]]:
        """Generate candidate segment sequences."""
        candidates = []

        # Try different geometric sequences
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            segments = []
            current = base_segment

            for _ in range(self.config.max_num_segments):
                if current > sequence_length:
                    break
                if current > self.config.max_segment_size:
                    break

                segments.append(int(current))
                current = int(current * ratio)

                # Ensure power of 2 if needed
                if self.config.prefer_power_of_2:
                    current = self._nearest_power_of_2(current)

            if segments:
                candidates.append(segments)

        # Also try a uniform sequence
        num_segments = min(4, sequence_length // base_segment)
        if num_segments > 0:
            uniform_size = self._nearest_power_of_2(sequence_length // num_segments)
            uniform_segments = [uniform_size] * num_segments
            candidates.append(uniform_segments)

        return candidates

    def _score_segment_sequence(
        self, segments: List[int], sequence_length: int
    ) -> float:
        """Score a segment sequence based on coverage and efficiency."""
        # Coverage score: how much of the sequence is covered
        max_segment = max(segments)
        coverage = min(1.0, max_segment / sequence_length)

        # Diversity score: having multiple scales is good
        diversity = len(set(segments)) / len(segments)

        # Efficiency score: prefer fewer, larger segments
        efficiency = 1.0 / len(segments)

        # Combine scores
        score = coverage * 0.5 + diversity * 0.3 + efficiency * 0.2

        return score

    def _compute_dilation_rates(self, segment_lengths: List[int]) -> List[int]:
        """Compute dilation rates for given segment lengths."""
        if not segment_lengths:
            return []

        # Use increasing dilation rates
        base_rate = 1
        rates = []

        for i, seg_len in enumerate(segment_lengths):
            # Dilation rate increases with segment size
            rate = base_rate * (2**i)
            rates.append(rate)

        return rates

    @staticmethod
    def _align_to_multiple(value: int, multiple: int) -> int:
        """Align value to the nearest multiple."""
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _nearest_power_of_2(value: int) -> int:
        """Find the nearest power of 2."""
        if value <= 0:
            return 1

        # If already a power of 2
        if (value & (value - 1)) == 0:
            return value

        # Find the nearest power of 2
        power = 1
        while power < value:
            power *= 2

        # Check if previous power is closer
        if power - value > value - power // 2:
            return power // 2

        return power

    def clear_cache(self):
        """Clear the segment selection cache."""
        self._cache.clear()
