"""
Quick validation benchmark for rapid performance testing.

This script provides:
1. Fast performance validation (< 1 minute)
2. Basic functionality testing
3. Memory usage estimation
4. Quick comparison between implementations

Useful for:
- CI/CD pipelines
- Pre-commit hooks
- Quick regression testing
- Development iteration
"""

import torch
import time
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from dilated_attention_pytorch import (
    DilatedAttention,
    MultiheadDilatedAttention,
    ImprovedDilatedAttention,
    ImprovedDilatedAttentionV2,
    ImprovedMultiheadDilatedAttention,
)

# Ring Attention implementations
try:
    from dilated_attention_pytorch.ring_dilated_attention_v2 import (
        RingDilatedAttentionV2,
    )

    RING_V2_AVAILABLE = True
except ImportError:
    RING_V2_AVAILABLE = False

try:
    from dilated_attention_pytorch.ring_dilated_attention_v3 import (
        RingDilatedAttentionV3,
    )

    RING_V3_AVAILABLE = True
except ImportError:
    RING_V3_AVAILABLE = False

try:
    from dilated_attention_pytorch import RingDilatedAttentionProduction

    RING_PRODUCTION_AVAILABLE = True
except ImportError:
    RING_PRODUCTION_AVAILABLE = False

# Block-Sparse implementations
try:
    from dilated_attention_pytorch import (
        BlockSparseRingDilatedAttention,
        BlockSparseRingMultiheadDilatedAttention,
    )

    BLOCK_SPARSE_AVAILABLE = True
except ImportError:
    BLOCK_SPARSE_AVAILABLE = False


@dataclass
class QuickValidationResult:
    """Quick validation result."""

    implementation: str
    test_name: str
    sequence_length: int
    batch_size: int
    time_ms: float
    memory_mb: float
    passed: bool
    error: Optional[str] = None


class QuickValidator:
    """Quick validation benchmarker."""

    # Quick test configurations
    QUICK_TESTS = [
        {"name": "tiny", "seq_len": 1024, "batch_size": 4},
        {"name": "small", "seq_len": 4096, "batch_size": 2},
        {"name": "medium", "seq_len": 16384, "batch_size": 1},
    ]

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = []

    def validate_implementation(
        self,
        implementation: str,
        seq_len: int,
        batch_size: int,
        num_heads: int = 8,
        head_dim: int = 64,
        test_name: str = "test",
    ) -> QuickValidationResult:
        """Validate a single implementation."""
        try:
            # Common kwargs
            kwargs = {
                "segment_lengths": [512, 1024],
                "dilation_rates": [1, 2],
            }

            # Create model based on implementation
            if implementation == "dilated":
                model = DilatedAttention(**kwargs).to(self.device)
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "multihead":
                model = MultiheadDilatedAttention(
                    embed_dim=num_heads * head_dim, num_heads=num_heads, **kwargs
                ).to(self.device)
                shape = (batch_size, seq_len, num_heads * head_dim)
            elif implementation == "improved":
                model = ImprovedDilatedAttention(**kwargs).to(self.device)
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "improved_v2":
                model = ImprovedDilatedAttentionV2(**kwargs).to(self.device)
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "improved_multihead":
                model = ImprovedMultiheadDilatedAttention(
                    embed_dim=num_heads * head_dim, num_heads=num_heads, **kwargs
                ).to(self.device)
                shape = (batch_size, seq_len, num_heads * head_dim)
            elif implementation == "ring_v2" and RING_V2_AVAILABLE:
                model = RingDilatedAttentionV2(use_pattern_cache=True, **kwargs).to(
                    self.device
                )
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "ring_v3" and RING_V3_AVAILABLE:
                model = RingDilatedAttentionV3(use_pattern_cache=True, **kwargs).to(
                    self.device
                )
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "ring_production" and RING_PRODUCTION_AVAILABLE:
                model = RingDilatedAttentionProduction(**kwargs).to(self.device)
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "block_sparse" and BLOCK_SPARSE_AVAILABLE:
                model = BlockSparseRingDilatedAttention(
                    sparsity_ratio=0.9, pattern_type="dilated_sparse", **kwargs
                ).to(self.device)
                shape = (batch_size, seq_len, num_heads, head_dim)
            elif implementation == "block_sparse_multihead" and BLOCK_SPARSE_AVAILABLE:
                model = BlockSparseRingMultiheadDilatedAttention(
                    embed_dim=num_heads * head_dim,
                    num_heads=num_heads,
                    sparsity_ratio=0.9,
                    pattern_type="dilated_sparse",
                    **kwargs,
                ).to(self.device)
                shape = (batch_size, seq_len, num_heads * head_dim)
            else:
                raise ValueError(
                    f"Unknown or unavailable implementation: {implementation}"
                )

            # Create input tensors
            # Use float32 for multihead implementations to avoid dtype issues
            if "multihead" in implementation:
                dtype = torch.float32
            else:
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            q = torch.randn(shape, device=self.device, dtype=dtype)
            k = torch.randn(shape, device=self.device, dtype=dtype)
            v = torch.randn(shape, device=self.device, dtype=dtype)

            # Warmup
            _ = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated() / 1024 / 1024

            # Time the forward pass
            start_time = time.time()
            output = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            # Calculate metrics
            time_ms = (end_time - start_time) * 1000

            if self.device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_mb = peak_memory - start_memory
            else:
                memory_mb = 0.0

            # Basic validation
            assert output.shape == q.shape, (
                f"Output shape mismatch: {output.shape} != {q.shape}"
            )
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"

            return QuickValidationResult(
                implementation=implementation,
                test_name=test_name,
                sequence_length=seq_len,
                batch_size=batch_size,
                time_ms=time_ms,
                memory_mb=memory_mb,
                passed=True,
            )

        except Exception as e:
            return QuickValidationResult(
                implementation=implementation,
                test_name=test_name,
                sequence_length=seq_len,
                batch_size=batch_size,
                time_ms=0.0,
                memory_mb=0.0,
                passed=False,
                error=str(e),
            )
        finally:
            # Cleanup
            if "model" in locals():
                del model
            if "q" in locals():
                del q, k, v
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def run_quick_validation(
        self, implementations: List[str]
    ) -> Dict[str, List[QuickValidationResult]]:
        """Run quick validation for all implementations."""
        results = {impl: [] for impl in implementations}

        print("Running quick validation tests...")
        print(f"Device: {self.device}")
        print(f"Implementations: {implementations}")
        print("=" * 60)

        for test_config in self.QUICK_TESTS:
            print(
                f"\nTest: {test_config['name']} (seq_len={test_config['seq_len']}, batch={test_config['batch_size']})"
            )

            for impl in implementations:
                print(f"  {impl}...", end=" ", flush=True)

                result = self.validate_implementation(
                    implementation=impl,
                    seq_len=test_config["seq_len"],
                    batch_size=test_config["batch_size"],
                    test_name=test_config["name"],
                )

                results[impl].append(result)

                if result.passed:
                    print(f"✓ {result.time_ms:.1f}ms, {result.memory_mb:.1f}MB")
                else:
                    print(f"✗ {result.error}")

        return results

    def generate_summary(self, results: Dict[str, List[QuickValidationResult]]) -> str:
        """Generate validation summary."""
        lines = []
        lines.append("Quick Validation Summary")
        lines.append("=" * 60)

        # Overall pass/fail
        lines.append("\nOverall Results:")
        for impl, impl_results in results.items():
            passed = sum(1 for r in impl_results if r.passed)
            total = len(impl_results)
            status = "✓ PASSED" if passed == total else f"✗ FAILED ({passed}/{total})"
            lines.append(f"  {impl}: {status}")

        # Performance comparison table
        lines.append("\nPerformance Summary:")
        lines.append("| Implementation | Test | Time (ms) | Memory (MB) | Status |")
        lines.append("|----------------|------|-----------|-------------|---------|")

        for impl, impl_results in results.items():
            for result in impl_results:
                status = "✓" if result.passed else "✗"
                lines.append(
                    f"| {result.implementation} | {result.test_name} | "
                    f"{result.time_ms:.1f} | {result.memory_mb:.1f} | {status} |"
                )

        # Average performance
        lines.append("\nAverage Performance (successful tests only):")
        for impl, impl_results in results.items():
            successful = [r for r in impl_results if r.passed]
            if successful:
                avg_time = sum(r.time_ms for r in successful) / len(successful)
                avg_memory = sum(r.memory_mb for r in successful) / len(successful)
                lines.append(f"  {impl}: {avg_time:.1f}ms, {avg_memory:.1f}MB")

        # Relative performance
        baseline_impl = "dilated"
        if baseline_impl in results:
            baseline_results = results[baseline_impl]
            baseline_times = {
                r.test_name: r.time_ms for r in baseline_results if r.passed
            }

            if baseline_times:
                lines.append(f"\nRelative Performance (vs {baseline_impl}):")
                for impl, impl_results in results.items():
                    if impl == baseline_impl:
                        continue

                    speedups = []
                    for result in impl_results:
                        if result.passed and result.test_name in baseline_times:
                            speedup = baseline_times[result.test_name] / result.time_ms
                            speedups.append(speedup)

                    if speedups:
                        avg_speedup = sum(speedups) / len(speedups)
                        lines.append(f"  {impl}: {avg_speedup:.2f}x")

        return "\n".join(lines)

    def save_results(
        self, results: Dict[str, List[QuickValidationResult]], output_file: str
    ):
        """Save results to file."""
        # Convert to JSON-serializable format
        json_data = {}
        for impl, impl_results in results.items():
            json_data[impl] = [
                {
                    "test_name": r.test_name,
                    "sequence_length": r.sequence_length,
                    "batch_size": r.batch_size,
                    "time_ms": r.time_ms,
                    "memory_mb": r.memory_mb,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in impl_results
            ]

        # Save JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "device": str(self.device),
                    "results": json_data,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick validation benchmark")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["dilated", "improved", "multihead"],
        help="Implementations to validate",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/quick_validation.json",
        help="Output file for results",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Initialize validator
    validator = QuickValidator(device=args.device)

    # Run validation
    start_time = time.time()
    results = validator.run_quick_validation(args.implementations)
    end_time = time.time()

    # Generate and print summary
    summary = validator.generate_summary(results)
    print("\n" + summary)

    # Save results
    validator.save_results(results, args.output)

    # Print timing
    total_time = end_time - start_time
    print(f"\nTotal validation time: {total_time:.1f} seconds")

    # Exit with appropriate code
    all_passed = all(
        all(r.passed for r in impl_results) for impl_results in results.values()
    )

    if all_passed:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Some tests failed!")
        exit(1)


if __name__ == "__main__":
    main()
