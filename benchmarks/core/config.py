"""Benchmark configuration system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch
import yaml
import json
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Test parameters
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])
    sequence_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 4096])
    num_heads: List[int] = field(default_factory=lambda: [8, 12, 16])
    embed_dims: List[int] = field(default_factory=lambda: [512, 768, 1024])

    # Dilated attention specific
    segment_lengths: List[List[int]] = field(
        default_factory=lambda: [[512, 1024], [1024, 2048], [2048, 4096]]
    )
    dilation_rates: List[List[int]] = field(
        default_factory=lambda: [[1, 2], [1, 2], [1, 2]]
    )

    # Sparse patterns
    sparsity_ratios: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    block_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    pattern_types: List[str] = field(
        default_factory=lambda: ["local_window", "dilated_sparse", "global_local"]
    )

    # Runtime parameters
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42

    # Output options
    output_format: str = "table"  # table, csv, json
    save_results: bool = True
    results_dir: str = "benchmark_results"
    plot_results: bool = False

    # Distributed options
    distributed: bool = False
    world_size: Optional[int] = None
    backend: str = "nccl"

    # Memory options
    measure_memory: bool = True
    memory_efficient: bool = True
    gradient_checkpointing: bool = False

    # Implementation selection
    implementations: List[str] = field(
        default_factory=lambda: ["standard", "improved", "ring", "block_sparse"]
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BenchmarkConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "batch_sizes": self.batch_sizes,
            "sequence_lengths": self.sequence_lengths,
            "num_heads": self.num_heads,
            "embed_dims": self.embed_dims,
            "segment_lengths": self.segment_lengths,
            "dilation_rates": self.dilation_rates,
            "sparsity_ratios": self.sparsity_ratios,
            "block_sizes": self.block_sizes,
            "pattern_types": self.pattern_types,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "output_format": self.output_format,
            "save_results": self.save_results,
            "results_dir": self.results_dir,
            "plot_results": self.plot_results,
            "distributed": self.distributed,
            "world_size": self.world_size,
            "backend": self.backend,
            "measure_memory": self.measure_memory,
            "memory_efficient": self.memory_efficient,
            "gradient_checkpointing": self.gradient_checkpointing,
            "implementations": self.implementations,
        }

    def get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device)

    def get_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)


@dataclass
class BenchmarkPreset:
    """Predefined benchmark configurations."""

    QUICK = BenchmarkConfig(
        batch_sizes=[1],
        sequence_lengths=[1024],
        num_heads=[8],
        embed_dims=[512],
        segment_lengths=[[512, 1024]],
        dilation_rates=[[1, 2]],
        warmup_iterations=1,
        benchmark_iterations=3,
    )

    STANDARD = BenchmarkConfig(
        batch_sizes=[1, 2],
        sequence_lengths=[2048, 4096],
        num_heads=[8, 12],
        embed_dims=[768],
        segment_lengths=[[1024, 2048], [2048, 4096]],
        dilation_rates=[[1, 2], [1, 2]],
    )

    COMPREHENSIVE = BenchmarkConfig(
        batch_sizes=[1, 2, 4],
        sequence_lengths=[1024, 2048, 4096, 8192],
        num_heads=[8, 12, 16],
        embed_dims=[512, 768, 1024],
        segment_lengths=[[512, 1024, 2048], [1024, 2048, 4096], [2048, 4096, 8192]],
        dilation_rates=[[1, 2, 4], [1, 2, 4], [1, 2, 4]],
        implementations=["standard", "improved", "ring", "block_sparse", "hilbert"],
    )

    MEMORY_FOCUSED = BenchmarkConfig(
        batch_sizes=[1],
        sequence_lengths=[8192, 16384, 32768, 65536],
        num_heads=[8],
        embed_dims=[768],
        measure_memory=True,
        memory_efficient=True,
        benchmark_iterations=1,  # Less iterations for memory tests
    )

    DISTRIBUTED = BenchmarkConfig(
        distributed=True,
        world_size=4,
        batch_sizes=[2, 4],
        sequence_lengths=[16384, 32768],
        implementations=["ring", "distributed"],
    )

    SPARSE_FOCUSED = BenchmarkConfig(
        implementations=["block_sparse", "block_sparse_ring"],
        sparsity_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        block_sizes=[64, 128, 256],
        pattern_types=["local_window", "dilated_sparse", "global_local", "adaptive"],
    )

    @classmethod
    def get_preset(cls, name: str) -> BenchmarkConfig:
        """Get a preset configuration by name."""
        presets = {
            "quick": cls.QUICK,
            "standard": cls.STANDARD,
            "comprehensive": cls.COMPREHENSIVE,
            "memory": cls.MEMORY_FOCUSED,
            "distributed": cls.DISTRIBUTED,
            "sparse": cls.SPARSE_FOCUSED,
        }

        if name not in presets:
            raise ValueError(
                f"Unknown preset: {name}. Available: {list(presets.keys())}"
            )

        return presets[name]
