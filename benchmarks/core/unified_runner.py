"""Unified benchmark runner for all attention implementations."""

import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from dilated_attention_pytorch import (
    ImprovedMultiheadDilatedAttention,
    MultiheadDilatedAttention,
    BlockSparseRingMultiheadDilatedAttention,
    create_multihead_dilated_attention,
    SparsePatternConfig,
)

from .base_benchmark import BaseBenchmark
from .config import BenchmarkConfig
from .output_manager import OutputManager
from .utils.memory import get_memory_stats
from .utils.timing import CUDATimer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    implementation: str
    batch_size: int
    seq_len: int
    num_heads: int
    embed_dim: int

    # Timing
    forward_time_ms: float
    backward_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None

    # Memory
    peak_memory_mb: Optional[float] = None
    allocated_memory_mb: Optional[float] = None

    # Throughput
    tokens_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None

    # Additional info
    segment_lengths: Optional[List[int]] = None
    dilation_rates: Optional[List[int]] = None
    sparsity_ratio: Optional[float] = None
    pattern_type: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class UnifiedBenchmarkRunner(BaseBenchmark):
    """Unified runner for all benchmark types."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        super().__init__(
            device=config.get_device(),
            dtype=config.get_dtype(),
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
        )
        self.config = config
        self.output_manager = OutputManager(config)
        self.results: List[BenchmarkResult] = []

    def create_attention_module(
        self,
        implementation: str,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        **kwargs,
    ) -> torch.nn.Module:
        """Create attention module based on implementation name.

        Args:
            implementation: Implementation name
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            segment_lengths: Segment lengths for dilated attention
            dilation_rates: Dilation rates
            **kwargs: Additional arguments

        Returns:
            Attention module
        """
        # Extract sparse config if provided
        sparse_config = None
        if "sparsity_ratio" in kwargs and "pattern_type" in kwargs:
            sparse_config = SparsePatternConfig(
                pattern_type=kwargs["pattern_type"],
                sparsity_ratio=kwargs["sparsity_ratio"],
                block_size=kwargs.get("block_size", 128),
            )

        # Map implementation names to modules
        impl_map = {
            "standard": lambda: MultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                batch_first=True,
            ),
            "improved": lambda: ImprovedMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                batch_first=True,
            ),
            "ring": lambda: create_multihead_dilated_attention(
                "ring",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                batch_first=True,
            ),
            "block_sparse": lambda: BlockSparseRingMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                sparse_config=sparse_config,
                dropout=0.0,
                batch_first=True,
            ),
            "hilbert": lambda: create_multihead_dilated_attention(
                "hilbert",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                batch_first=True,
            ),
            "distributed": lambda: create_multihead_dilated_attention(
                "distributed",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                batch_first=True,
            ),
        }

        if implementation not in impl_map:
            raise ValueError(f"Unknown implementation: {implementation}")

        module = impl_map[implementation]()
        return module.to(self.device).to(self.dtype)

    def benchmark_single_configuration(
        self,
        implementation: str,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        embed_dim: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a single configuration.

        Args:
            implementation: Implementation name
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of heads
            embed_dim: Embedding dimension
            segment_lengths: Segment lengths
            dilation_rates: Dilation rates
            **kwargs: Additional arguments

        Returns:
            Benchmark result
        """
        result = BenchmarkResult(
            implementation=implementation,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            embed_dim=embed_dim,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        )

        try:
            # Create module
            module = self.create_attention_module(
                implementation=implementation,
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                **kwargs,
            )

            # Create input tensors
            x = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
            )
            x.requires_grad_(True)

            # Warmup
            for _ in range(self.warmup_iterations):
                with torch.no_grad():
                    _ = module(x, x, x)

            self.cleanup_memory()

            # Forward timing
            timer = CUDATimer(self.device)
            forward_times = []

            for _ in range(self.benchmark_iterations):
                timer.start()
                output, _ = module(x, x, x, is_causal=True)
                timer.end()
                forward_times.append(timer.elapsed_time())

            result.forward_time_ms = sum(forward_times) / len(forward_times)

            # Backward timing (if requested)
            if self.config.measure_memory:
                timer.start()
                loss = output.sum()
                loss.backward()
                timer.end()
                result.backward_time_ms = timer.elapsed_time()
                result.total_time_ms = result.forward_time_ms + result.backward_time_ms

            # Memory stats
            if self.config.measure_memory and self.device.type == "cuda":
                memory_stats = get_memory_stats(self.device)
                result.peak_memory_mb = memory_stats["peak"] / (1024 * 1024)
                result.allocated_memory_mb = memory_stats["allocated"] / (1024 * 1024)

            # Throughput
            total_tokens = batch_size * seq_len
            result.tokens_per_second = (total_tokens / result.forward_time_ms) * 1000
            result.samples_per_second = (batch_size / result.forward_time_ms) * 1000

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            if self.config.save_results:
                traceback.print_exc()

        return result

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks based on configuration.

        Returns:
            List of benchmark results
        """
        self.results = []
        total_configs = (
            len(self.config.implementations)
            * len(self.config.batch_sizes)
            * len(self.config.sequence_lengths)
            * len(self.config.num_heads)
            * len(self.config.embed_dims)
        )

        print(f"Running {total_configs} benchmark configurations...")
        config_idx = 0

        for impl in self.config.implementations:
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    for num_heads in self.config.num_heads:
                        for embed_dim in self.config.embed_dims:
                            config_idx += 1

                            # Find appropriate segment lengths
                            segment_lengths = None
                            dilation_rates = None

                            for segs, dils in zip(
                                self.config.segment_lengths, self.config.dilation_rates
                            ):
                                if seq_len % max(segs) == 0:
                                    segment_lengths = segs
                                    dilation_rates = dils
                                    break

                            if segment_lengths is None:
                                print(
                                    f"Skipping seq_len={seq_len} (not divisible by segment lengths)"
                                )
                                continue

                            print(
                                f"[{config_idx}/{total_configs}] "
                                f"{impl} - B={batch_size}, L={seq_len}, "
                                f"H={num_heads}, D={embed_dim}"
                            )

                            # Run benchmark
                            result = self.benchmark_single_configuration(
                                implementation=impl,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_heads=num_heads,
                                embed_dim=embed_dim,
                                segment_lengths=segment_lengths,
                                dilation_rates=dilation_rates,
                            )

                            self.results.append(result)

                            # Print result
                            if result.error:
                                print(f"  ERROR: {result.error}")
                            else:
                                print(
                                    f"  Forward: {result.forward_time_ms:.2f}ms, "
                                    f"Memory: {result.peak_memory_mb:.1f}MB, "
                                    f"Throughput: {result.tokens_per_second:.0f} tok/s"
                                )

        # Save results
        if self.config.save_results:
            self.output_manager.save_results(self.results)

        # Display summary
        self.output_manager.display_summary(self.results)

        return self.results

    def run_sparse_benchmarks(self) -> List[BenchmarkResult]:
        """Run block sparse specific benchmarks."""
        self.results = []

        for sparsity in self.config.sparsity_ratios:
            for block_size in self.config.block_sizes:
                for pattern in self.config.pattern_types:
                    for batch_size in self.config.batch_sizes:
                        for seq_len in self.config.sequence_lengths:
                            print(
                                f"Block Sparse - sparsity={sparsity}, "
                                f"block={block_size}, pattern={pattern}"
                            )

                            # Find appropriate config
                            embed_dim = self.config.embed_dims[0]
                            num_heads = self.config.num_heads[0]

                            # Get segment lengths
                            segment_lengths = None
                            dilation_rates = None
                            for segs, dils in zip(
                                self.config.segment_lengths, self.config.dilation_rates
                            ):
                                if seq_len % max(segs) == 0:
                                    segment_lengths = segs
                                    dilation_rates = dils
                                    break

                            if segment_lengths is None:
                                continue

                            result = self.benchmark_single_configuration(
                                implementation="block_sparse",
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_heads=num_heads,
                                embed_dim=embed_dim,
                                segment_lengths=segment_lengths,
                                dilation_rates=dilation_rates,
                                sparsity_ratio=sparsity,
                                block_size=block_size,
                                pattern_type=pattern,
                            )

                            result.sparsity_ratio = sparsity
                            result.pattern_type = pattern

                            self.results.append(result)

        if self.config.save_results:
            self.output_manager.save_results(self.results)

        return self.results
