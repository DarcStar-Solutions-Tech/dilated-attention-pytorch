"""Unified benchmark for extreme sequence lengths across all implementations."""

import gc
from typing import Dict, Tuple

import torch
import torch.nn as nn

from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
)

try:
    from dilated_attention_pytorch import RingDilatedAttentionV2

    HAS_RING = True
except ImportError:
    HAS_RING = False

try:
    from dilated_attention_pytorch import BlockSparseRingDilatedAttention

    HAS_BLOCK_SPARSE = True
except ImportError:
    HAS_BLOCK_SPARSE = False

from benchmarks.framework import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from benchmarks.framework.config import (
    ExtremeSequenceConfig,
    get_segment_dilation_configs,
)
from benchmarks.framework.utils import create_attention_inputs


class ExtremeSequencesBenchmark(BaseBenchmark):
    """Benchmark for testing implementations with extreme sequence lengths."""

    def __init__(
        self, test_multihead: bool = False, max_memory_gb: float = None, **kwargs
    ):
        """Initialize benchmark.

        Args:
            test_multihead: Whether to test multihead implementations
            max_memory_gb: Maximum GPU memory to use (auto-detect if None)
            **kwargs: Additional arguments for BaseBenchmark
        """
        # Update config with extreme sequence lengths
        if "config" not in kwargs:
            kwargs["config"] = BenchmarkConfig()

        extreme_config = ExtremeSequenceConfig()
        kwargs["config"].seq_lengths = extreme_config.seq_lengths
        kwargs["config"].batch_sizes = [extreme_config.batch_size]
        kwargs["config"].warmup_steps = 1  # Minimal warmup for extreme lengths
        kwargs["config"].benchmark_steps = 3  # Fewer steps for extreme lengths

        super().__init__(**kwargs)
        self.test_multihead = test_multihead
        self.max_memory_gb = max_memory_gb

        if self.max_memory_gb is None and self.device.type == "cuda":
            # Auto-detect available memory
            self.max_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            print(f"Auto-detected GPU memory: {self.max_memory_gb:.1f} GB")

    def setup_models(self) -> Dict[str, nn.Module]:
        """Setup only memory-efficient implementations."""
        models = {}

        # Use minimal configuration for extreme sequences
        embed_dim = self.config.num_heads_list[0] * self.config.head_dim

        # Start with a reasonable segment configuration
        # Will be updated dynamically based on sequence length
        segment_lengths = [2048, 4096]
        dilation_rates = [1, 2]

        # Improved implementation (most memory efficient single-GPU)
        models["improved"] = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            use_xpos=False,
            use_rel_pos_bias=False,
        ).to(self.device)

        if self.test_multihead:
            models["improved_multihead"] = ImprovedMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=self.config.num_heads_list[0],
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                batch_first=True,
            ).to(self.device)

        # Ring attention (for distributed memory)
        if HAS_RING:
            try:
                models["ring_v2"] = RingDilatedAttentionV2(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    use_xpos=False,
                    use_rel_pos_bias=False,
                    ring_size=4,  # Simulate 4-GPU ring
                ).to(self.device)
            except Exception as e:
                print(f"Failed to create ring attention: {e}")

        # Block sparse (most memory efficient)
        if HAS_BLOCK_SPARSE:
            try:
                models["block_sparse_90"] = BlockSparseRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    use_xpos=False,
                    use_rel_pos_bias=False,
                    ring_size=1,
                    block_size=256,
                    sparsity_ratio=0.9,  # 90% sparse
                ).to(self.device)

                models["block_sparse_95"] = BlockSparseRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    use_xpos=False,
                    use_rel_pos_bias=False,
                    ring_size=1,
                    block_size=256,
                    sparsity_ratio=0.95,  # 95% sparse
                ).to(self.device)
            except Exception as e:
                print(f"Failed to create block sparse attention: {e}")

        # Set all models to eval mode
        for model in models.values():
            model.eval()

        return models

    def get_model_inputs(
        self, batch_size: int, seq_length: int, num_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, ...]:
        """Create minimal inputs to save memory."""
        # For extreme sequences, we only test core format
        q, k, v = create_attention_inputs(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            self.device,
            self.dtype,
            is_multihead=False,
        )
        return (q, k, v)

    def estimate_memory_usage(
        self, seq_length: int, num_heads: int, head_dim: int, batch_size: int = 1
    ) -> float:
        """Estimate memory usage in GB for a given configuration."""
        # Input tensors: 3 * (batch * seq * heads * dim)
        input_memory = 3 * batch_size * seq_length * num_heads * head_dim * 2  # fp16

        # Attention scores: batch * heads * seq * seq (worst case)
        attention_memory = batch_size * num_heads * seq_length * seq_length * 2

        # Output: same as one input
        output_memory = batch_size * seq_length * num_heads * head_dim * 2

        # Add 50% overhead for intermediate activations
        total_bytes = (input_memory + attention_memory + output_memory) * 1.5

        return total_bytes / (1024**3)  # Convert to GB

    def find_max_sequence_length(
        self, model: nn.Module, implementation: str, num_heads: int, head_dim: int
    ) -> Tuple[int, float, float]:
        """Find maximum sequence length that fits in memory.

        Returns:
            Tuple of (max_seq_length, time_ms, memory_mb)
        """
        batch_size = 1
        test_lengths = self.config.seq_lengths

        max_length = 0
        max_time = 0.0
        max_memory = 0.0

        for seq_length in test_lengths:
            # Skip if estimated memory exceeds limit
            est_memory_gb = self.estimate_memory_usage(seq_length, num_heads, head_dim)
            if self.max_memory_gb and est_memory_gb > self.max_memory_gb * 0.8:
                print(f"    Skipping {seq_length} (est. {est_memory_gb:.1f} GB)")
                continue

            # Update model configuration
            configs = get_segment_dilation_configs(seq_length)
            if configs and hasattr(model, "segment_lengths"):
                segment_lengths, dilation_rates = configs[0]
                model.segment_lengths = segment_lengths
                model.dilation_rates = dilation_rates

            # Try to run
            result = self.benchmark_configuration(
                implementation, model, batch_size, seq_length, num_heads, head_dim
            )

            if result.success:
                max_length = seq_length
                max_time = result.time_ms
                max_memory = result.memory_mb
                print(
                    f"    ✓ {seq_length}: {result.time_ms:.1f}ms, {result.memory_mb:.0f}MB"
                )
            else:
                print(f"    ✗ {seq_length}: {result.error}")
                break  # Stop on first failure

            # Clear memory after each test
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return max_length, max_time, max_memory

    def run(self):
        """Run extreme sequence length tests."""
        print(f"Setting up models on {self.device} with dtype {self.dtype}")
        print(f"Memory limit: {self.max_memory_gb:.1f} GB")
        self.models = self.setup_models()

        print("\nFinding maximum sequence lengths for each implementation...")

        # Test with minimal configuration
        num_heads = self.config.num_heads_list[0]
        head_dim = self.config.head_dim

        # Summary results
        max_lengths = {}

        for impl_name, model in self.models.items():
            print(f"\nTesting {impl_name}...")
            max_length, time_ms, memory_mb = self.find_max_sequence_length(
                model, impl_name, num_heads, head_dim
            )

            max_lengths[impl_name] = {
                "max_length": max_length,
                "time_ms": time_ms,
                "memory_mb": memory_mb,
            }

            # Add to results
            self.results.append(
                BenchmarkResult(
                    implementation=impl_name,
                    batch_size=1,
                    seq_length=max_length,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    time_ms=time_ms,
                    memory_mb=memory_mb,
                    throughput=max_length / time_ms * 1000 if time_ms > 0 else 0,
                    success=max_length > 0,
                    extra_metrics={"max_sequence_length": max_length},
                )
            )

        # Print summary
        print("\n" + "=" * 80)
        print("MAXIMUM SEQUENCE LENGTH SUMMARY")
        print("=" * 80)

        for impl, data in sorted(
            max_lengths.items(), key=lambda x: x[1]["max_length"], reverse=True
        ):
            if data["max_length"] > 0:
                print(f"\n{impl}:")
                print(f"  Max length: {data['max_length']:,} tokens")
                print(f"  Time: {data['time_ms']:.1f} ms")
                print(
                    f"  Memory: {data['memory_mb']:.0f} MB ({data['memory_mb'] / 1024:.1f} GB)"
                )
                print(
                    f"  Throughput: {data['max_length'] / data['time_ms'] * 1000:.0f} tokens/s"
                )

    def analyze(self):
        """Analyze extreme sequence results."""
        if not self.results:
            print("No results to analyze")
            return

        # Sort by max sequence length
        sorted_results = sorted(
            [r for r in self.results if r.success],
            key=lambda x: x.seq_length,
            reverse=True,
        )

        if sorted_results:
            print("\n" + "=" * 80)
            print("EXTREME SEQUENCE LENGTH ANALYSIS")
            print("=" * 80)

            # Best implementation
            best = sorted_results[0]
            print(f"\nBest implementation for extreme sequences: {best.implementation}")
            print(f"  - Achieved {best.seq_length:,} tokens")
            print(
                f"  - Memory efficiency: {best.seq_length / (best.memory_mb / 1024):.0f} tokens/GB"
            )

            # Relative performance
            if len(sorted_results) > 1:
                print("\nRelative maximum sequence lengths:")
                baseline_length = sorted_results[-1].seq_length  # Shortest as baseline

                for result in sorted_results:
                    ratio = result.seq_length / baseline_length
                    print(
                        f"  {result.implementation}: {ratio:.1f}x ({result.seq_length:,} tokens)"
                    )


def main():
    """Run extreme sequences benchmark."""
    benchmark = ExtremeSequencesBenchmark.from_args()

    print("=" * 80)
    print("EXTREME SEQUENCE LENGTH BENCHMARK")
    print("=" * 80)

    benchmark.run()
    benchmark.analyze()
    benchmark.save_results("extreme_sequences")


if __name__ == "__main__":
    main()
