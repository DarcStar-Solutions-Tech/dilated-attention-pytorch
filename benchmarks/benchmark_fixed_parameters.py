#!/usr/bin/env python3
"""
Fixed benchmark script with correct parameters for all implementations.

This addresses the parameter mismatches found in the initial benchmarks.
"""

import torch
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime, timezone
import gc

# Suppress warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    implementation: str
    category: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    peak_memory_mb: float
    throughput_tokens_per_sec: float
    success: bool
    error: Optional[str] = None


class FixedParametersBenchmark:
    """Benchmark with corrected parameters for all implementations."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []

    def get_implementations(self) -> Dict[str, Any]:
        """Get implementations with CORRECTED parameters."""
        implementations = {}

        # 1. Core implementations (these work fine)
        implementations["core"] = []

        try:
            from dilated_attention_pytorch import DilatedAttention

            implementations["core"].append(
                {
                    "name": "DilatedAttention",
                    "class": DilatedAttention,
                    "init": lambda: DilatedAttention(
                        segment_lengths=[512, 1024, 2048], dilation_rates=[1, 2, 4]
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load DilatedAttention: {e}")

        try:
            from dilated_attention_pytorch import ImprovedDilatedAttention

            implementations["core"].append(
                {
                    "name": "ImprovedDilatedAttention",
                    "class": ImprovedDilatedAttention,
                    "init": lambda: ImprovedDilatedAttention(
                        segment_lengths=[512, 1024, 2048], dilation_rates=[1, 2, 4]
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load ImprovedDilatedAttention: {e}")

        # 2. Ring attention implementations (FIXED)
        implementations["ring"] = []

        # RingDilatedAttentionProduction - FIX: Use correct config without 'dim'
        try:
            from dilated_attention_pytorch import (
                RingDilatedAttentionProduction,
                RingAttentionConfig,
            )

            implementations["ring"].append(
                {
                    "name": "RingDilatedAttentionProduction",
                    "class": RingDilatedAttentionProduction,
                    "init": lambda: RingDilatedAttentionProduction(
                        config=RingAttentionConfig(
                            # No 'dim' parameter here!
                            segment_lengths=[512, 1024, 2048],
                            dilation_rates=[1, 2, 4],
                            ring_size=1,
                            block_size=512,
                        )
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load RingDilatedAttentionProduction: {e}")

        # RingDilatedAttentionProductionFixed - with retain_graph handling
        try:
            from dilated_attention_pytorch.ring_dilated_attention_production_fixed import (
                RingDilatedAttentionProductionFixed,
            )

            implementations["ring"].append(
                {
                    "name": "RingDilatedAttentionProductionFixed",
                    "class": RingDilatedAttentionProductionFixed,
                    "init": lambda: RingDilatedAttentionProductionFixed(
                        dim=64,
                        heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                    ),
                    "input_format": "4d",
                    "needs_retain_graph": True,  # Special flag for backward pass
                }
            )
        except Exception as e:
            print(f"Failed to load RingDilatedAttentionProductionFixed: {e}")

        # 3. Kernel implementations (FIXED)
        implementations["kernels"] = []

        # HilbertAttentionTritonFixed - FIX: Use correct parameters
        try:
            from dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton_fixed import (
                HilbertAttentionTritonFixed,
            )

            implementations["kernels"].append(
                {
                    "name": "HilbertAttentionTritonFixed",
                    "class": HilbertAttentionTritonFixed,
                    "init": lambda: HilbertAttentionTritonFixed(
                        hidden_dim=768,  # Not segment_lengths!
                        num_heads=12,
                        segment_size=2048,  # Single segment size, not list
                        dilation_rate=1,  # Single dilation rate, not list
                        dropout=0.0,
                    ),
                    "input_format": "3d",  # Uses hidden_dim format
                }
            )
        except Exception as e:
            print(f"Failed to load HilbertAttentionTritonFixed: {e}")

        # Keep successful implementations
        implementations["multihead"] = []

        try:
            from dilated_attention_pytorch import MultiheadDilatedAttention

            implementations["multihead"].append(
                {
                    "name": "MultiheadDilatedAttention",
                    "class": MultiheadDilatedAttention,
                    "init": lambda: MultiheadDilatedAttention(
                        embed_dim=768,
                        num_heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        dropout=0.0,
                    ),
                    "input_format": "3d",
                }
            )
        except Exception as e:
            print(f"Failed to load MultiheadDilatedAttention: {e}")

        try:
            from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention

            implementations["multihead"].append(
                {
                    "name": "ImprovedMultiheadDilatedAttention",
                    "class": ImprovedMultiheadDilatedAttention,
                    "init": lambda: ImprovedMultiheadDilatedAttention(
                        embed_dim=768,
                        num_heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        dropout=0.0,
                    ),
                    "input_format": "3d",
                }
            )
        except Exception as e:
            print(f"Failed to load ImprovedMultiheadDilatedAttention: {e}")

        # Block-sparse implementations (these work fine)
        implementations["block_sparse"] = []

        try:
            from dilated_attention_pytorch import BlockSparseRingDilatedAttention

            implementations["block_sparse"].append(
                {
                    "name": "BlockSparseRingDilatedAttention",
                    "class": BlockSparseRingDilatedAttention,
                    "init": lambda: BlockSparseRingDilatedAttention(
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        sparsity_ratio=0.1,
                        block_size=64,
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load BlockSparseRingDilatedAttention: {e}")

        return implementations

    def create_inputs(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        input_format: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create properly formatted inputs."""
        if input_format == "4d":
            # [batch, seq_len, num_heads, head_dim]
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            k = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            v = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
        elif input_format == "3d":
            # [batch, seq_len, embed_dim]
            embed_dim = num_heads * head_dim
            q = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float32
            )
            k = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float32
            )
            v = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float32
            )
        else:
            raise ValueError(f"Unknown input format: {input_format}")

        return q, k, v

    def benchmark_implementation(
        self,
        impl_info: Dict[str, Any],
        category: str,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a single implementation."""
        name = impl_info["name"]

        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            # Create model
            model = impl_info["init"]().to(self.device)
            model.eval()

            # Create inputs
            q, k, v = self.create_inputs(
                batch_size, seq_len, num_heads, head_dim, impl_info["input_format"]
            )

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = model(q, k, v)

            # Measure forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            forward_times = []
            for _ in range(num_iterations):
                start = time.time()
                with torch.no_grad():
                    output = model(q, k, v)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_times.append(time.time() - start)

            forward_time = sum(forward_times) / len(forward_times)

            # Measure backward pass with special handling for retain_graph
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            backward_times = []
            needs_retain_graph = impl_info.get("needs_retain_graph", False)

            for i in range(num_iterations):
                start = time.time()
                output = model(q, k, v)
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.sum()

                # Use retain_graph for implementations that need it
                if needs_retain_graph and i < num_iterations - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_times.append(time.time() - start)

                # Clear gradients for next iteration
                if q.grad is not None:
                    q.grad.zero_()
                    k.grad.zero_()
                    v.grad.zero_()

            backward_time = sum(backward_times) / len(backward_times)

            # Get peak memory
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_memory_mb = 0.0

            # Calculate throughput
            total_tokens = batch_size * seq_len
            throughput = total_tokens / forward_time

            return BenchmarkResult(
                implementation=name,
                category=category,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time_ms=forward_time * 1000,
                backward_time_ms=backward_time * 1000,
                total_time_ms=(forward_time + backward_time) * 1000,
                peak_memory_mb=peak_memory_mb,
                throughput_tokens_per_sec=throughput,
                success=True,
            )

        except Exception as e:
            traceback.print_exc()
            return BenchmarkResult(
                implementation=name,
                category=category,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time_ms=0.0,
                backward_time_ms=0.0,
                total_time_ms=0.0,
                peak_memory_mb=0.0,
                throughput_tokens_per_sec=0.0,
                success=False,
                error=str(e),
            )

    def run_benchmarks(
        self,
        batch_sizes: List[int] = [2],
        seq_lens: List[int] = [2048],
        num_heads: int = 12,
        head_dim: int = 64,
    ):
        """Run benchmarks for all implementations."""
        implementations = self.get_implementations()

        total_impls = sum(len(impls) for impls in implementations.values())
        print(f"Testing {total_impls} implementations with FIXED parameters")
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\nBenchmarking: batch_size={batch_size}, seq_len={seq_len}")
                print("-" * 60)

                for category, impls in implementations.items():
                    if impls:
                        print(f"\n{category.upper()} implementations:")
                        for impl in impls:
                            print(f"  Testing {impl['name']}...", end=" ", flush=True)

                            result = self.benchmark_implementation(
                                impl_info=impl,
                                category=category,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_heads=num_heads,
                                head_dim=head_dim,
                            )

                            self.results.append(result)

                            if result.success:
                                print(
                                    f"✓ {result.forward_time_ms:.1f}ms fwd, "
                                    f"{result.backward_time_ms:.1f}ms bwd, "
                                    f"{result.peak_memory_mb:.0f}MB"
                                )
                            else:
                                print(f"✗ {result.error}")

    def save_results(self, filename: str):
        """Save results to JSON."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "note": "Fixed parameter benchmark - addresses all known parameter issues",
            "results": [
                {
                    "implementation": r.implementation,
                    "category": r.category,
                    "batch_size": r.batch_size,
                    "seq_len": r.seq_len,
                    "num_heads": r.num_heads,
                    "head_dim": r.head_dim,
                    "forward_time_ms": r.forward_time_ms,
                    "backward_time_ms": r.backward_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "peak_memory_mb": r.peak_memory_mb,
                    "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("FIXED PARAMETERS BENCHMARK SUMMARY")
        print("=" * 80)

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        print(
            f"\nOverall: {len(successful)}/{len(self.results)} succeeded "
            f"({len(successful) / len(self.results) * 100:.1f}%)"
        )

        if successful:
            print("\n✅ SUCCESSFUL IMPLEMENTATIONS:")
            for r in successful:
                print(
                    f"  {r.implementation}: {r.forward_time_ms:.1f}ms fwd, "
                    f"{r.backward_time_ms:.1f}ms bwd"
                )

        if failed:
            print("\n❌ STILL FAILING:")
            for r in failed:
                print(f"  {r.implementation}: {r.error}")


def main():
    """Main benchmark function."""
    benchmark = FixedParametersBenchmark()

    # Run benchmarks
    print("Running benchmarks with FIXED parameters...")
    benchmark.run_benchmarks(
        batch_sizes=[2], seq_lens=[2048], num_heads=12, head_dim=64
    )

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/fixed_parameters_benchmark_{timestamp}.json"
    benchmark.save_results(filename)
    print(f"\nResults saved to {filename}")

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
