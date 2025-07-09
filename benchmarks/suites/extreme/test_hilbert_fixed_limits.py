#!/usr/bin/env python3
"""
Push the fixed Hilbert implementation to hardware limits using safe benchmarking.

This script tests:
1. Maximum sequence lengths with memory safety
2. Performance comparison (Hilbert vs non-Hilbert)
3. Multi-segment configurations
4. Different batch sizes and head configurations
"""

import torch
import time
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core_fixed import (
    RingDilatedAttentionHilbertCoreFixed,
)
from benchmarks.core.utils.safety import (
    SafetyConfig,
    MemorySafetyChecker,
    ProgressiveTester,
)


class HilbertBenchmarkSuite:
    """Comprehensive benchmark suite for Hilbert attention."""

    def __init__(self, safety_config: Optional[SafetyConfig] = None):
        self.safety_config = safety_config or SafetyConfig(
            max_memory_fraction=0.85,  # Use up to 85% GPU memory
            min_free_memory_gb=1.5,  # Keep 1.5GB free
            progressive_steps=8,  # More steps for thorough testing
        )
        self.safety_checker = MemorySafetyChecker(self.safety_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

    def benchmark_config(
        self,
        model,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        use_hilbert: bool,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
    ) -> Dict:
        """Benchmark a specific configuration."""
        # Check memory before proceeding
        shape = (batch_size, seq_len, num_heads, head_dim)
        required_memory = (
            self.safety_checker.estimate_tensor_memory(
                shape,
                torch.float32,
                num_tensors=4,  # Q, K, V, Output
            )
            * 1.5
        )  # Add overhead

        can_allocate, message = self.safety_checker.check_memory_available(
            required_memory
        )
        if not can_allocate:
            return {
                "status": "skipped",
                "reason": message,
                "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "use_hilbert": use_hilbert,
                },
            }

        try:
            # Move model to device
            model = model.to(self.device)
            dtype = torch.float32  # Use fp32 as requested

            # Create inputs
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup
            for _ in range(warmup_iterations):
                _ = model(q, k, v)

            # Synchronize
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # Time forward passes
            forward_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                output = model(q, k, v)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                forward_times.append((end - start) * 1000)  # ms

            # Get memory stats
            if self.device.type == "cuda":
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                current_memory_gb = torch.cuda.memory_allocated() / 1e9
            else:
                peak_memory_gb = 0
                current_memory_gb = 0

            # Calculate statistics
            result = {
                "status": "success",
                "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "use_hilbert": use_hilbert,
                    "total_params": sum(p.numel() for p in model.parameters()),
                },
                "performance": {
                    "mean_time_ms": np.mean(forward_times),
                    "std_time_ms": np.std(forward_times),
                    "min_time_ms": np.min(forward_times),
                    "max_time_ms": np.max(forward_times),
                    "throughput_seq_per_sec": 1000 / np.mean(forward_times),
                },
                "memory": {
                    "peak_gb": peak_memory_gb,
                    "current_gb": current_memory_gb,
                    "efficiency": (batch_size * seq_len * num_heads * head_dim * 2)
                    / (peak_memory_gb * 1e9)
                    if peak_memory_gb > 0
                    else 0,
                },
            }

            # Cleanup
            del q, k, v, output
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            # Cleanup on error
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return {
                "status": "failed",
                "error": str(e),
                "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "use_hilbert": use_hilbert,
                },
            }

    def find_max_sequence_length(
        self,
        model_class,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        segment_lengths: List[int] = [2048, 4096, 8192],
        dilation_rates: List[int] = [1, 2, 4],
        use_hilbert: bool = True,
    ) -> int:
        """Find maximum sequence length that fits in memory."""
        print(
            f"\nFinding max sequence length for batch_size={batch_size}, heads={num_heads}"
        )

        # Binary search for max sequence length
        min_seq = 1024
        max_seq = 1024 * 1024  # 1M tokens

        # First, quickly find upper bound
        test_seq = min_seq
        while test_seq <= max_seq:
            shape = (batch_size, test_seq, num_heads, head_dim)
            memory_gb = (
                self.safety_checker.estimate_tensor_memory(shape, torch.float32, 4)
                * 1.5
            )
            can_allocate, _ = self.safety_checker.check_memory_available(memory_gb)

            if not can_allocate:
                max_seq = test_seq // 2
                break

            test_seq *= 2

        # Binary search for exact limit
        best_seq = min_seq
        while min_seq <= max_seq:
            mid_seq = (min_seq + max_seq) // 2
            # Round to nearest segment boundary
            mid_seq = (mid_seq // segment_lengths[0]) * segment_lengths[0]

            print(f"  Testing seq_len={mid_seq:,}...", end="", flush=True)

            # Create model
            model = model_class(
                dim=num_heads * head_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_hilbert=use_hilbert,
            )

            result = self.benchmark_config(
                model,
                batch_size,
                mid_seq,
                num_heads,
                head_dim,
                use_hilbert,
                num_iterations=3,  # Fewer iterations for search
            )

            if result["status"] == "success":
                print(f" ✓ ({result['performance']['mean_time_ms']:.1f}ms)")
                best_seq = mid_seq
                min_seq = mid_seq + segment_lengths[0]
            else:
                print(f" ✗ ({result.get('reason', result.get('error', 'failed'))})")
                max_seq = mid_seq - segment_lengths[0]

            del model

        return best_seq

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks pushing hardware limits."""
        print("\n" + "=" * 80)
        print(f"Hilbert Attention Hardware Limit Benchmark - {datetime.now()}")
        print("=" * 80)

        # Show system info
        if self.device.type == "cuda":
            used, free, total = self.safety_checker.get_gpu_memory_info()
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")
            print(
                f"Safety limits: {self.safety_config.max_memory_fraction:.0%} max usage, {self.safety_config.min_free_memory_gb}GB min free"
            )

        # Test configurations
        test_configs = [
            # Find maximum sequence lengths for different batch sizes
            {"batch_size": 1, "num_heads": 8, "head_dim": 64},
            {"batch_size": 2, "num_heads": 8, "head_dim": 64},
            {"batch_size": 4, "num_heads": 8, "head_dim": 64},
            {"batch_size": 1, "num_heads": 16, "head_dim": 64},
            {"batch_size": 1, "num_heads": 32, "head_dim": 32},
        ]

        max_lengths = {}

        # Phase 1: Find maximum sequence lengths
        print("\n--- Phase 1: Finding Maximum Sequence Lengths ---")
        for config in test_configs:
            key = f"b{config['batch_size']}_h{config['num_heads']}"
            max_seq = self.find_max_sequence_length(
                RingDilatedAttentionHilbertCoreFixed, **config
            )
            max_lengths[key] = max_seq
            print(f"  Max for {key}: {max_seq:,} tokens")
            self.safety_checker.force_cleanup()

        # Phase 2: Compare Hilbert vs Non-Hilbert at max lengths
        print("\n--- Phase 2: Hilbert vs Non-Hilbert Comparison ---")
        comparison_results = []

        for config in test_configs[:3]:  # Test first 3 configs
            key = f"b{config['batch_size']}_h{config['num_heads']}"
            max_seq = max_lengths.get(key, 8192)

            print(f"\nComparing at seq_len={max_seq:,}, batch={config['batch_size']}")

            # Test with Hilbert
            model_hilbert = RingDilatedAttentionHilbertCoreFixed(
                dim=config["num_heads"] * config["head_dim"],
                heads=config["num_heads"],
                segment_lengths=[2048, 4096, 8192],
                dilation_rates=[1, 2, 4],
                use_hilbert=True,
            )

            result_hilbert = self.benchmark_config(
                model_hilbert,
                config["batch_size"],
                max_seq,
                config["num_heads"],
                config["head_dim"],
                use_hilbert=True,
            )

            # Test without Hilbert
            model_no_hilbert = RingDilatedAttentionHilbertCoreFixed(
                dim=config["num_heads"] * config["head_dim"],
                heads=config["num_heads"],
                segment_lengths=[2048, 4096, 8192],
                dilation_rates=[1, 2, 4],
                use_hilbert=False,
            )

            result_no_hilbert = self.benchmark_config(
                model_no_hilbert,
                config["batch_size"],
                max_seq,
                config["num_heads"],
                config["head_dim"],
                use_hilbert=False,
            )

            if (
                result_hilbert["status"] == "success"
                and result_no_hilbert["status"] == "success"
            ):
                speedup = (
                    result_no_hilbert["performance"]["mean_time_ms"]
                    / result_hilbert["performance"]["mean_time_ms"]
                )
                print(
                    f"  Hilbert: {result_hilbert['performance']['mean_time_ms']:.1f}ms"
                )
                print(
                    f"  No Hilbert: {result_no_hilbert['performance']['mean_time_ms']:.1f}ms"
                )
                print(f"  Speedup: {speedup:.2f}x")

                comparison_results.append(
                    {
                        "config": config,
                        "seq_len": max_seq,
                        "hilbert_ms": result_hilbert["performance"]["mean_time_ms"],
                        "no_hilbert_ms": result_no_hilbert["performance"][
                            "mean_time_ms"
                        ],
                        "speedup": speedup,
                    }
                )

            del model_hilbert, model_no_hilbert
            self.safety_checker.force_cleanup()

        # Phase 3: Extreme sequence length test
        print("\n--- Phase 3: Extreme Sequence Length Test ---")

        # Progressive test to absolute limit
        tester = ProgressiveTester(self.safety_checker)

        def test_extreme_length(seq_len):
            model = RingDilatedAttentionHilbertCoreFixed(
                dim=512,
                heads=8,
                segment_lengths=[2048, 4096, 8192, 16384],
                dilation_rates=[1, 2, 4, 8],
                use_hilbert=True,
            )

            return self.benchmark_config(
                model, 1, seq_len, 8, 64, True, num_iterations=1
            )

        print("Pushing to absolute memory limits...")
        extreme_result = tester.test_with_safety(
            test_extreme_length,
            {},
            size_param_name="seq_len",
            target_size=256 * 1024,  # Target 256K tokens
        )

        # Phase 4: Generate report
        self.generate_report(max_lengths, comparison_results, extreme_result)

    def generate_report(self, max_lengths, comparison_results, extreme_result):
        """Generate comprehensive benchmark report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("BENCHMARK REPORT - Fixed Hilbert Implementation")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now()}")

        # System info
        if self.device.type == "cuda":
            report.append(f"\nGPU: {torch.cuda.get_device_name(0)}")
            used, free, total = self.safety_checker.get_gpu_memory_info()
            report.append(f"GPU Memory: {total:.1f}GB total")

        # Maximum sequence lengths
        report.append("\n## Maximum Sequence Lengths")
        report.append("-" * 40)
        for key, max_seq in max_lengths.items():
            report.append(f"  {key}: {max_seq:,} tokens")

        # Hilbert comparison
        if comparison_results:
            report.append("\n## Hilbert vs Non-Hilbert Performance")
            report.append("-" * 40)
            total_speedup = 0
            for result in comparison_results:
                report.append(
                    f"  Config: batch={result['config']['batch_size']}, "
                    f"heads={result['config']['num_heads']}, "
                    f"seq_len={result['seq_len']:,}"
                )
                report.append(f"    Hilbert: {result['hilbert_ms']:.1f}ms")
                report.append(f"    No Hilbert: {result['no_hilbert_ms']:.1f}ms")
                report.append(f"    Speedup: {result['speedup']:.2f}x")
                total_speedup += result["speedup"]

            avg_speedup = total_speedup / len(comparison_results)
            report.append(f"\n  Average Speedup: {avg_speedup:.2f}x")

        # Extreme test results
        if extreme_result and extreme_result.get("status") == "success":
            report.append("\n## Extreme Sequence Length Achievement")
            report.append("-" * 40)
            config = extreme_result["config"]
            report.append(f"  Maximum tested: {config['seq_len']:,} tokens")
            report.append(
                f"  Time: {extreme_result['performance']['mean_time_ms']:.1f}ms"
            )
            report.append(f"  Memory: {extreme_result['memory']['peak_gb']:.2f}GB")
            report.append(
                f"  Throughput: {extreme_result['performance']['throughput_seq_per_sec']:.1f} seq/sec"
            )

        # Save report
        report_text = "\n".join(report)
        print(report_text)

        # Save to file
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        report_path = Path(__file__).parent / f"hilbert-fixed-limits-{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"\nReport saved to: {report_path}")


def main():
    """Run the benchmark suite."""
    # Create custom safety config for pushing limits
    safety_config = SafetyConfig(
        max_memory_fraction=0.90,  # Use up to 90% for extreme testing
        min_free_memory_gb=1.0,  # Only keep 1GB free
        progressive_steps=10,  # More granular steps
        cleanup_threshold=0.8,  # Cleanup at 80% usage
    )

    # Create and run benchmark suite
    suite = HilbertBenchmarkSuite(safety_config)
    suite.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
