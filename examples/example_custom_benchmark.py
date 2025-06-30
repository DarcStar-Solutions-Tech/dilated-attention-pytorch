"""Example of creating a custom benchmark using the framework."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
)

from framework import BaseBenchmark, BenchmarkConfig
from framework.utils import create_attention_inputs


class CustomMemoryBenchmark(BaseBenchmark):
    """Example custom benchmark focusing on memory usage patterns."""

    def setup_models(self) -> Dict[str, nn.Module]:
        """Setup models to test."""
        models = {}

        # Common configuration
        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]
        _ = 512
        _ = 8

        # Standard improved attention
        models["improved_base"] = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        ).to(self.device)

        # With different dropout
        models["improved_dropout"] = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.1,
        ).to(self.device)

        # You could add multihead versions here, but they need
        # different input shapes (batch, seq, embed_dim)

        return models

    def get_model_inputs(
        self, batch_size: int, seq_length: int, num_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, ...]:
        """Create inputs based on model requirements."""
        # For this simple example, only test core attention format
        q, k, v = create_attention_inputs(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            self.device,
            self.dtype,
            is_multihead=False,  # Core format for all models
        )
        return (q, k, v)

    # Remove the override - let the base class handle it
    # The base class already calls get_model_inputs properly

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
