"""Example of creating a custom benchmark using the framework."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
)

from benchmarks.framework import BaseBenchmark, BenchmarkConfig
from benchmarks.framework.utils import create_attention_inputs


class CustomMemoryBenchmark(BaseBenchmark):
    """Example custom benchmark focusing on memory usage patterns."""

    def setup_models(self) -> Dict[str, nn.Module]:
        """Setup models to test."""
        models = {}

        # Common configuration
        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]
        embed_dim = 512
        num_heads = 8

        # Standard improved attention
        models["improved_base"] = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        ).to(self.device)

        # With memory optimizations
        models["improved_optimized"] = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_memory_efficient_attention=True,
        ).to(self.device)

        # Multihead version
        models["multihead_optimized"] = ImprovedMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_memory_efficient_attention=True,
            batch_first=True,
        ).to(self.device)

        return models

    def get_model_inputs(
        self, batch_size: int, seq_length: int, num_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, ...]:
        """Create inputs based on model requirements."""
        # Create both core and multihead formats
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
        """Override to select appropriate inputs."""
        # Get all inputs
        q_core, k_core, v_core, q_multi, k_multi, v_multi = self.get_model_inputs(
            batch_size, seq_length, num_heads, head_dim
        )

        # Select inputs based on model type
        if "multihead" in implementation_name:
            _ = (q_multi, k_multi, v_multi)
        else:
            _ = (q_core, k_core, v_core)

        # Call parent implementation
        return super().benchmark_configuration(
            implementation_name, model, batch_size, seq_length, num_heads, head_dim
        )

    def analyze(self):
        """Custom analysis focusing on memory efficiency."""
        super().analyze()

        print("\n" + "=" * 80)
        print("MEMORY EFFICIENCY ANALYSIS")
        print("=" * 80)

        # Group results by sequence length
        seq_results = {}
        for result in self.results:
            if result.success:
                seq_len = result.seq_length
                if seq_len not in seq_results:
                    seq_results[seq_len] = []
                seq_results[seq_len].append(result)

        # Analyze memory efficiency
        for seq_len in sorted(seq_results.keys()):
            results = seq_results[seq_len]
            print(f"\nSequence Length: {seq_len}")

            # Find most memory efficient
            best = min(results, key=lambda r: r.memory_mb)
            worst = max(results, key=lambda r: r.memory_mb)

            print(f"  Most efficient: {best.implementation} ({best.memory_mb:.0f} MB)")
            print(
                f"  Least efficient: {worst.implementation} ({worst.memory_mb:.0f} MB)"
            )
            print(
                f"  Memory savings: {(1 - best.memory_mb / worst.memory_mb) * 100:.1f}%"
            )


def main():
    """Run custom benchmark with specific configuration."""
    # Create custom configuration
    config = BenchmarkConfig(
        batch_sizes=[1, 2],
        seq_lengths=[2048, 4096, 8192],
        num_heads_list=[8],
        head_dim=64,
        warmup_steps=2,
        benchmark_steps=5,
    )

    # Create and run benchmark
    benchmark = CustomMemoryBenchmark(config=config)

    print("=" * 80)
    print("CUSTOM MEMORY EFFICIENCY BENCHMARK")
    print("=" * 80)

    benchmark.run()
    benchmark.analyze()
    benchmark.save_results("custom_memory")


if __name__ == "__main__":
    main()
