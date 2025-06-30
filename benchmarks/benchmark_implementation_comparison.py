"""Unified benchmark comparing all dilated attention implementations."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
    ImprovedMultiheadDilatedAttention,
)

try:
    from dilated_attention_pytorch import (
        BlockSparseRingDilatedAttention,
        BlockSparseRingMultiheadDilatedAttention,
    )

    HAS_BLOCK_SPARSE = True
except ImportError:
    HAS_BLOCK_SPARSE = False

try:
    from dilated_attention_pytorch import RingDilatedAttentionV2Collective

    HAS_RING = True
except ImportError:
    HAS_RING = False

from framework import BaseBenchmark
from framework.config import get_segment_dilation_configs
from framework.utils import create_attention_inputs, compare_implementations


class ImplementationComparisonBenchmark(BaseBenchmark):
    """Comprehensive comparison of all dilated attention implementations."""

    def __init__(self, include_multihead: bool = True, **kwargs):
        """Initialize benchmark.

        Args:
            include_multihead: Whether to include multihead implementations
            **kwargs: Additional arguments for BaseBenchmark
        """
        super().__init__(**kwargs)
        self.include_multihead = include_multihead

    def setup_models(self) -> Dict[str, nn.Module]:
        """Setup all available implementations."""
        models = {}

        # Get appropriate segment/dilation configs for the first test sequence length
        seq_length = self.config.seq_lengths[0]
        configs = get_segment_dilation_configs(seq_length)

        if configs:
            segment_lengths, dilation_rates = configs[0]  # Use first config
        else:
            # Fallback for short sequences
            segment_lengths = [seq_length]
            dilation_rates = [1]

        embed_dim = self.config.num_heads_list[0] * self.config.head_dim

        # Core implementations
        models["standard"] = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=0.0,
        ).to(self.device)

        models["improved"] = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
        ).to(self.device)

        # Multihead implementations
        if self.include_multihead:
            models["multihead_standard"] = MultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=self.config.num_heads_list[0],
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            ).to(self.device)

            models["multihead_improved"] = ImprovedMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=self.config.num_heads_list[0],
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            ).to(self.device)

        # Ring attention (if available)
        if HAS_RING:
            try:
                models["ring_v2_collective"] = RingDilatedAttentionV2Collective(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    ring_size=1,  # Single GPU simulation
                ).to(self.device)
            except Exception as e:
                print(f"Failed to create ring attention: {e}")

        # Block sparse (if available)
        if HAS_BLOCK_SPARSE:
            try:
                models["block_sparse"] = BlockSparseRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    ring_size=1,
                    sparsity_ratio=0.9,
                ).to(self.device)

                if self.include_multihead:
                    models["block_sparse_multihead"] = (
                        BlockSparseRingMultiheadDilatedAttention(
                            embed_dim=embed_dim,
                            num_heads=self.config.num_heads_list[0],
                            segment_lengths=segment_lengths,
                            dilation_rates=dilation_rates,
                            dropout=0.0,
                            ring_size=1,
                            sparsity_ratio=0.9,
                        ).to(self.device)
                    )
            except Exception as e:
                print(f"Failed to create block sparse attention: {e}")

        # Set all models to eval mode
        for model in models.values():
            model.eval()

        return models

    def get_model_inputs(
        self, batch_size: int, seq_length: int, num_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, ...]:
        """Get appropriate inputs based on model type."""
        # For this benchmark, we'll create both formats and use as needed
        q_core, k_core, v_core = create_attention_inputs(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            self.device,
            self.dtype,
            is_multihead=False,
        )

        q_multi, k_multi, v_multi = create_attention_inputs(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            self.device,
            self.dtype,
            is_multihead=True,
        )

        # Return both sets - models will use appropriate ones
        return (q_core, k_core, v_core, q_multi, k_multi, v_multi)

    def benchmark_configuration(
        self,
        implementation_name: str,
        model: nn.Module,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
    ):
        """Override to handle different input formats."""
        try:
            # Get all inputs
            q_core, k_core, v_core, q_multi, k_multi, v_multi = self.get_model_inputs(
                batch_size, seq_length, num_heads, head_dim
            )

            # Update model's segment lengths if sequence length changed
            if hasattr(model, "segment_lengths"):
                configs = get_segment_dilation_configs(seq_length)
                if configs:
                    segment_lengths, dilation_rates = configs[0]
                    if max(segment_lengths) <= seq_length:
                        model.segment_lengths = segment_lengths
                        model.dilation_rates = dilation_rates

            # Select appropriate inputs
            if "multihead" in implementation_name:
                inputs = (q_multi, k_multi, v_multi)
            else:
                inputs = (q_core, k_core, v_core)

            # Measure performance
            time_ms, memory_mb = self.measure_performance(
                model, inputs, num_runs=self.config.benchmark_steps
            )

            # Calculate throughput
            total_tokens = batch_size * seq_length
            throughput = (total_tokens / time_ms) * 1000

            from framework.base import BenchmarkResult

            return BenchmarkResult(
                implementation=implementation_name,
                batch_size=batch_size,
                seq_length=seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=time_ms,
                memory_mb=memory_mb,
                throughput=throughput,
                success=True,
            )

        except Exception as e:
            from framework.base import BenchmarkResult

            return BenchmarkResult(
                implementation=implementation_name,
                batch_size=batch_size,
                seq_length=seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=0.0,
                memory_mb=0.0,
                throughput=0.0,
                success=False,
                error=str(e),
            )

    def analyze(self):
        """Extended analysis with implementation comparison."""
        super().analyze()

        if len(self.results) > 10:  # Only if we have enough data
            print("\n" + "=" * 80)
            print("RELATIVE PERFORMANCE")
            print("=" * 80)

            # Compare against standard implementation
            comparison_df = compare_implementations(
                [r.to_dict() for r in self.results], baseline="standard"
            )

            if not comparison_df.empty:
                print("\nPerformance relative to standard implementation:")
                print(comparison_df.to_string())

                # Summary statistics
                print("\nAverage performance across all configurations:")
                for impl in comparison_df["implementation"].unique():
                    impl_data = comparison_df[comparison_df["implementation"] == impl]
                    avg_speedup = impl_data["speedup"].mean()
                    avg_memory = impl_data["memory_ratio"].mean()

                    print(f"\n{impl}:")
                    print(f"  Average speedup: {avg_speedup:.2f}x")
                    print(f"  Average memory ratio: {avg_memory:.2f}x")

    @classmethod
    def add_arguments(cls, parser):
        """Add benchmark-specific arguments."""
        parser.add_argument(
            "--no-multihead",
            action="store_true",
            help="Exclude multihead implementations",
        )


def main():
    """Run implementation comparison benchmark."""
    benchmark = ImplementationComparisonBenchmark.from_args()

    # Override include_multihead if specified
    import sys

    if "--no-multihead" in sys.argv:
        benchmark.include_multihead = False

    print("=" * 80)
    print("DILATED ATTENTION IMPLEMENTATION COMPARISON")
    print("=" * 80)

    benchmark.run()
    benchmark.analyze()
    benchmark.save_results("implementation_comparison")


if __name__ == "__main__":
    main()
