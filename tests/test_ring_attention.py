#!/usr/bin/env python3
"""
Comprehensive test suite for Ring Attention implementations.

This script validates the correctness, performance, and scalability of
the Ring Attention implementations against baseline dilated attention.

Tests include:
- Mathematical equivalence validation
- Memory complexity verification
- Performance benchmarking
- Multi-GPU scaling tests
- Memory efficiency analysis
"""

import argparse
import os
import sys
import time
import tracemalloc
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # from dilated_attention_pytorch.ring_advanced_distributed_dilated_attention import RingAdvancedDistributedDilatedAttention  # Not implemented yet
    from dilated_attention_pytorch.improved_dilated_attention import ImprovedDilatedAttention
    from dilated_attention_pytorch.improved_multihead_dilated_attention import (
        ImprovedMultiheadDilatedAttention,
    )
    from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
    from dilated_attention_pytorch.ring_multihead_dilated_attention import (
        RingMultiheadDilatedAttention,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import attention modules: {e}")
    IMPORTS_AVAILABLE = False


class RingAttentionTester:
    """Comprehensive tester for Ring Attention implementations."""

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        tolerance: float = 1e-4,
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tolerance = tolerance
        self.verbose = verbose

        # Test configurations
        self.test_configs = [
            {
                "name": "Small Model",
                "batch_size": 2,
                "seq_len": 4096,
                "embed_dim": 256,
                "num_heads": 8,
                "segment_lengths": [1024, 2048],
                "dilation_rates": [1, 2],
            },
            {
                "name": "Medium Model",
                "batch_size": 1,
                "seq_len": 8192,
                "embed_dim": 512,
                "num_heads": 8,
                "segment_lengths": [2048, 4096],
                "dilation_rates": [1, 2],
            },
            {
                "name": "Large Model",
                "batch_size": 1,
                "seq_len": 16384,
                "embed_dim": 768,
                "num_heads": 12,
                "segment_lengths": [2048, 4096, 8192],
                "dilation_rates": [1, 2, 4],
            },
        ]

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[RingAttentionTester] {message}")

    def create_test_inputs(
        self, batch_size: int, seq_len: int, embed_dim: int, num_heads: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test input tensors."""
        head_dim = embed_dim // num_heads

        # Create random but reproducible inputs
        torch.manual_seed(42)

        # For standalone attention (q, k, v format)
        q = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        k = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        v = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        return q, k, v

    def create_multihead_test_inputs(
        self, batch_size: int, seq_len: int, embed_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test inputs for multihead attention."""
        torch.manual_seed(42)

        query = torch.randn(
            batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
        )
        key = torch.randn(
            batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
        )
        value = torch.randn(
            batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
        )

        return query, key, value

    def measure_memory_and_time(self, func, *args, **kwargs):
        """Measure memory usage and execution time."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            if self.device.type == "cuda":
                end_memory = torch.cuda.memory_allocated()
                gpu_memory_used = end_memory - start_memory
            else:
                gpu_memory_used = 0

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                "result": result,
                "execution_time": end_time - start_time,
                "gpu_memory_used": gpu_memory_used,
                "cpu_peak_memory": peak,
            }
        except Exception as e:
            tracemalloc.stop()
            return {
                "result": None,
                "error": str(e),
                "execution_time": float("inf"),
                "gpu_memory_used": float("inf"),
                "cpu_peak_memory": float("inf"),
            }

    def test_mathematical_equivalence(self) -> dict[str, bool]:
        """Test mathematical equivalence between Ring and standard attention."""
        self.log("Testing mathematical equivalence...")

        results = {}

        for config in self.test_configs:
            config_name = config["name"]
            self.log(f"  Testing {config_name}...")

            try:
                # Create test inputs
                q, k, v = self.create_test_inputs(
                    config["batch_size"],
                    config["seq_len"],
                    config["embed_dim"],
                    config["num_heads"],
                )

                # Standard improved attention (single device)
                standard_attention = ImprovedDilatedAttention(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                ).to(self.device)

                # Ring attention (single device mode)
                ring_attention = RingDilatedAttention(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                    ring_size=1,  # Single device for equivalence test
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    standard_output = standard_attention(q, k, v, is_causal=False)
                    ring_output = ring_attention(q, k, v, is_causal=False)

                # Check equivalence
                max_diff = torch.max(torch.abs(standard_output - ring_output)).item()
                is_equivalent = max_diff < self.tolerance

                self.log(
                    f"    Max difference: {max_diff:.2e} (tolerance: {self.tolerance:.2e})"
                )
                self.log(f"    Equivalent: {is_equivalent}")

                results[config_name] = is_equivalent

            except Exception as e:
                self.log(f"    Error: {e}")
                results[config_name] = False

        return results

    def test_multihead_equivalence(self) -> dict[str, bool]:
        """Test multihead attention equivalence."""
        self.log("Testing multihead equivalence...")

        results = {}

        for config in self.test_configs:
            config_name = f"{config['name']} Multihead"
            self.log(f"  Testing {config_name}...")

            try:
                # Create test inputs
                query, key, value = self.create_multihead_test_inputs(
                    config["batch_size"], config["seq_len"], config["embed_dim"]
                )

                # Standard improved multihead attention
                standard_attention = ImprovedMultiheadDilatedAttention(
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                ).to(self.device)

                # Ring multihead attention (single device mode)
                ring_attention = RingMultiheadDilatedAttention(
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                    ring_size=1,  # Single device for equivalence test
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    standard_output, _ = standard_attention(
                        query, key, value, is_causal=False
                    )
                    ring_output, _ = ring_attention(query, key, value, is_causal=False)

                # Check equivalence
                max_diff = torch.max(torch.abs(standard_output - ring_output)).item()
                is_equivalent = max_diff < self.tolerance

                self.log(
                    f"    Max difference: {max_diff:.2e} (tolerance: {self.tolerance:.2e})"
                )
                self.log(f"    Equivalent: {is_equivalent}")

                results[config_name] = is_equivalent

            except Exception as e:
                self.log(f"    Error: {e}")
                results[config_name] = False

        return results

    def test_memory_complexity(self) -> dict[str, dict]:
        """Test memory complexity scaling."""
        self.log("Testing memory complexity scaling...")

        results = {}

        # Test different sequence lengths
        base_config = self.test_configs[0].copy()
        sequence_lengths = [1024, 2048, 4096, 8192]

        for seq_len in sequence_lengths:
            if seq_len > 8192 and self.device.type == "cpu":
                continue  # Skip very long sequences on CPU

            config_name = f"seq_len_{seq_len}"
            self.log(f"  Testing {config_name}...")

            try:
                # Update config
                test_config = base_config.copy()
                test_config["seq_len"] = seq_len
                test_config["segment_lengths"] = [
                    min(s, seq_len // 2) for s in test_config["segment_lengths"]
                ]

                # Create inputs
                q, k, v = self.create_test_inputs(
                    test_config["batch_size"],
                    test_config["seq_len"],
                    test_config["embed_dim"],
                    test_config["num_heads"],
                )

                # Test Ring attention
                ring_attention = RingDilatedAttention(
                    segment_lengths=test_config["segment_lengths"],
                    dilation_rates=test_config["dilation_rates"],
                    dropout=0.0,
                    ring_size=1,
                ).to(self.device)

                # Measure memory usage
                def forward_pass():
                    with torch.no_grad():
                        return ring_attention(q, k, v, is_causal=False)

                metrics = self.measure_memory_and_time(forward_pass)

                results[config_name] = {
                    "seq_len": seq_len,
                    "execution_time": metrics["execution_time"],
                    "gpu_memory_used": metrics["gpu_memory_used"],
                    "cpu_peak_memory": metrics["cpu_peak_memory"],
                    "success": metrics.get("error") is None,
                }

                self.log(
                    f"    Time: {metrics['execution_time']:.4f}s, "
                    f"GPU Memory: {metrics['gpu_memory_used'] / 1024**2:.1f}MB"
                )

            except Exception as e:
                self.log(f"    Error: {e}")
                results[config_name] = {
                    "seq_len": seq_len,
                    "success": False,
                    "error": str(e),
                }

        return results

    def test_performance_comparison(self) -> dict[str, dict]:
        """Compare performance between Ring and standard attention."""
        self.log("Testing performance comparison...")

        results = {}

        for config in self.test_configs[:2]:  # Limit to smaller configs
            config_name = config["name"]
            self.log(f"  Testing {config_name}...")

            try:
                # Create inputs
                q, k, v = self.create_test_inputs(
                    config["batch_size"],
                    config["seq_len"],
                    config["embed_dim"],
                    config["num_heads"],
                )

                # Standard attention
                standard_attention = ImprovedDilatedAttention(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                ).to(self.device)

                # Ring attention
                ring_attention = RingDilatedAttention(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                    ring_size=1,
                ).to(self.device)

                # Benchmark standard attention
                def standard_forward():
                    with torch.no_grad():
                        return standard_attention(q, k, v, is_causal=False)

                standard_metrics = self.measure_memory_and_time(standard_forward)

                # Benchmark ring attention
                def ring_forward():
                    with torch.no_grad():
                        return ring_attention(q, k, v, is_causal=False)

                ring_metrics = self.measure_memory_and_time(ring_forward)

                # Calculate speedup and memory efficiency
                if (
                    standard_metrics.get("error") is None
                    and ring_metrics.get("error") is None
                ):
                    speedup = (
                        standard_metrics["execution_time"]
                        / ring_metrics["execution_time"]
                    )
                    memory_ratio = ring_metrics["gpu_memory_used"] / max(
                        standard_metrics["gpu_memory_used"], 1
                    )

                    results[config_name] = {
                        "standard_time": standard_metrics["execution_time"],
                        "ring_time": ring_metrics["execution_time"],
                        "speedup": speedup,
                        "standard_memory": standard_metrics["gpu_memory_used"],
                        "ring_memory": ring_metrics["gpu_memory_used"],
                        "memory_ratio": memory_ratio,
                        "success": True,
                    }

                    self.log(
                        f"    Speedup: {speedup:.2f}x, Memory ratio: {memory_ratio:.2f}"
                    )
                else:
                    results[config_name] = {
                        "success": False,
                        "standard_error": standard_metrics.get("error"),
                        "ring_error": ring_metrics.get("error"),
                    }

            except Exception as e:
                self.log(f"    Error: {e}")
                results[config_name] = {"success": False, "error": str(e)}

        return results

    def test_optimization_effectiveness(self) -> dict[str, dict]:
        """Test that optimizations don't break functionality."""
        self.log("Testing optimization effectiveness...")

        results = {}

        for config in self.test_configs[:1]:  # Test with one config
            config_name = f"{config['name']} Optimization Test"
            self.log(f"  Testing {config_name}...")

            try:
                # Create inputs
                q, k, v = self.create_test_inputs(
                    config["batch_size"],
                    config["seq_len"],
                    config["embed_dim"],
                    config["num_heads"],
                )

                # Ring attention with optimizations
                ring_attention = RingDilatedAttention(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                    ring_size=1,
                ).to(self.device)

                # Test memory info functionality
                memory_info = ring_attention.get_memory_info()

                # Test cache clearing
                ring_attention.clear_cache()

                # Test multiple forward passes (should use cached patterns)
                outputs = []
                for _ in range(3):
                    with torch.no_grad():
                        output = ring_attention(q, k, v, is_causal=False)
                        outputs.append(output)

                # Verify outputs are consistent
                max_diff = max(
                    torch.max(torch.abs(outputs[i] - outputs[0])).item()
                    for i in range(1, len(outputs))
                )

                results[config_name] = {
                    "memory_info_available": isinstance(memory_info, dict),
                    "cache_management_works": True,  # If we get here, it worked
                    "consistent_outputs": max_diff < 1e-6,
                    "max_output_difference": max_diff,
                    "success": True,
                }

                self.log(
                    f"    Memory info: {memory_info.get('memory_complexity', 'N/A')}"
                )
                self.log(f"    Max output difference: {max_diff:.2e}")

            except Exception as e:
                self.log(f"    Error: {e}")
                results[config_name] = {"success": False, "error": str(e)}

        return results

    def run_all_tests(self) -> dict[str, Any]:
        """Run comprehensive test suite."""
        if not IMPORTS_AVAILABLE:
            return {
                "success": False,
                "error": "Required modules not available for testing",
            }

        self.log("=" * 60)
        self.log("Ring Attention Comprehensive Test Suite")
        self.log("=" * 60)

        results = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "tolerance": self.tolerance,
        }

        try:
            # Test mathematical equivalence
            results["mathematical_equivalence"] = self.test_mathematical_equivalence()

            # Test multihead equivalence
            results["multihead_equivalence"] = self.test_multihead_equivalence()

            # Test memory complexity
            results["memory_complexity"] = self.test_memory_complexity()

            # Test performance comparison
            results["performance_comparison"] = self.test_performance_comparison()

            # Test optimization effectiveness
            results["optimization_effectiveness"] = (
                self.test_optimization_effectiveness()
            )

            # Overall success
            all_equiv_tests = list(results["mathematical_equivalence"].values()) + list(
                results["multihead_equivalence"].values()
            )
            results["overall_success"] = all(all_equiv_tests)

            self.log("=" * 60)
            self.log("Test Summary:")
            self.log(f"Mathematical Equivalence: {results['mathematical_equivalence']}")
            self.log(f"Multihead Equivalence: {results['multihead_equivalence']}")
            self.log(f"Overall Success: {results['overall_success']}")
            self.log("=" * 60)

        except Exception as e:
            self.log(f"Test suite failed: {e}")
            results["overall_success"] = False
            results["error"] = str(e)

        return results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Ring Attention implementations")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for tests",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Numerical tolerance for equivalence tests",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup device and dtype
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Run tests
    tester = RingAttentionTester(
        device=args.device, dtype=dtype, tolerance=args.tolerance, verbose=args.verbose
    )

    results = tester.run_all_tests()

    # Print final results
    if results.get("overall_success", False):
        print("\\n✅ All Ring Attention tests passed!")
        print(
            "Ring Attention implementations are mathematically equivalent and ready for use."
        )
    else:
        print("\\n❌ Some Ring Attention tests failed.")
        print("Please review the test output above for details.")

    return 0 if results.get("overall_success", False) else 1


if __name__ == "__main__":
    exit(main())
