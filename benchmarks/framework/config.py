"""Configuration classes for benchmark framework."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class BenchmarkConfig:
    """Standard benchmark configuration."""

    # Test dimensions
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])
    seq_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 4096])
    num_heads_list: List[int] = field(default_factory=lambda: [8, 16])
    head_dim: int = 64

    # Dilated attention specific
    segment_lengths: List[List[int]] = field(
        default_factory=lambda: [[512, 1024], [1024, 2048], [2048, 4096]]
    )
    dilation_rates: List[List[int]] = field(
        default_factory=lambda: [[1, 2], [1, 2], [1, 2]]
    )

    # Measurement settings
    warmup_steps: int = 3
    benchmark_steps: int = 10

    # Device settings
    device: str = "cuda"
    use_fp16: bool = True

    # Memory pool settings
    use_memory_pool: bool = True
    memory_pool_size: Optional[int] = None  # Auto-detect if None

    # Pattern cache settings
    use_pattern_cache: bool = True
    pattern_cache_size: int = 100

    # Output settings
    save_plots: bool = True
    save_csv: bool = True
    verbose: bool = True


@dataclass
class DistributedConfig:
    """Configuration for distributed benchmarks."""

    world_sizes: List[int] = field(default_factory=lambda: [2, 4, 8])
    backend: str = "nccl"
    init_method: str = "env://"
    master_addr: str = "localhost"
    master_port: str = "29500"


@dataclass
class ExtremeSequenceConfig:
    """Configuration for extreme sequence length benchmarks."""

    seq_lengths: List[int] = field(
        default_factory=lambda: [8192, 16384, 32768, 65536, 131072, 262144]
    )
    batch_size: int = 1  # Usually 1 for extreme lengths
    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = False
    max_memory_gb: Optional[float] = None  # Auto-detect if None


@dataclass
class BenchmarkSuite:
    """Configuration for a suite of benchmarks."""

    name: str
    description: str
    configs: List[BenchmarkConfig]
    implementations: List[str] = field(
        default_factory=lambda: [
            "standard",
            "improved",
            "distributed",
            "ring",
            "block_sparse",
        ]
    )

    # Comparison settings
    baseline_implementation: str = "standard"

    # Test categories
    test_memory_efficiency: bool = True
    test_compute_efficiency: bool = True
    test_scalability: bool = True

    @classmethod
    def quick_test(cls) -> "BenchmarkSuite":
        """Create a quick test suite for validation."""
        return cls(
            name="quick_test",
            description="Quick validation benchmark",
            configs=[
                BenchmarkConfig(
                    batch_sizes=[1],
                    seq_lengths=[1024],
                    num_heads_list=[8],
                    warmup_steps=1,
                    benchmark_steps=3,
                )
            ],
        )

    @classmethod
    def standard_suite(cls) -> "BenchmarkSuite":
        """Create standard benchmark suite."""
        return cls(
            name="standard",
            description="Standard benchmark suite covering common configurations",
            configs=[
                BenchmarkConfig(
                    batch_sizes=[1, 2, 4],
                    seq_lengths=[1024, 2048, 4096],
                    num_heads_list=[8, 16],
                )
            ],
        )

    @classmethod
    def memory_efficiency_suite(cls) -> "BenchmarkSuite":
        """Create memory efficiency focused suite."""
        return cls(
            name="memory_efficiency",
            description="Memory efficiency benchmarks with large sequences",
            configs=[
                BenchmarkConfig(
                    batch_sizes=[1],
                    seq_lengths=[4096, 8192, 16384, 32768],
                    num_heads_list=[8],
                    benchmark_steps=5,  # Fewer steps for memory tests
                )
            ],
        )

    @classmethod
    def extreme_suite(cls) -> "BenchmarkSuite":
        """Create extreme sequence length suite."""
        extreme_config = ExtremeSequenceConfig()
        return cls(
            name="extreme",
            description="Extreme sequence length benchmarks",
            configs=[
                BenchmarkConfig(
                    batch_sizes=[1],
                    seq_lengths=extreme_config.seq_lengths,
                    num_heads_list=[8],
                    warmup_steps=1,
                    benchmark_steps=3,
                )
            ],
            implementations=[
                "improved",
                "ring",
                "block_sparse",
            ],  # Only memory-efficient ones
        )


def get_segment_dilation_configs(
    seq_length: int,
) -> List[Tuple[List[int], List[int]]]:
    """Get appropriate segment lengths and dilation rates for a sequence length.

    Args:
        seq_length: Target sequence length

    Returns:
        List of (segment_lengths, dilation_rates) tuples
    """
    configs = []

    # Single segment (no dilation)
    if seq_length >= 512:
        configs.append(([seq_length], [1]))

    # Two segments
    if seq_length >= 1024:
        configs.append(([seq_length // 2, seq_length // 2], [1, 2]))

    # Three segments
    if seq_length >= 2048:
        third = seq_length // 3
        # Round to nearest power of 2 for better performance
        third = 2 ** round(torch.log2(torch.tensor(float(third))).item())
        if third * 3 <= seq_length:
            configs.append(([third, third, third], [1, 2, 4]))

    # Four segments (for very long sequences)
    if seq_length >= 8192:
        quarter = seq_length // 4
        quarter = 2 ** round(torch.log2(torch.tensor(float(quarter))).item())
        if quarter * 4 <= seq_length:
            configs.append(([quarter, quarter, quarter, quarter], [1, 2, 4, 8]))

    return configs
