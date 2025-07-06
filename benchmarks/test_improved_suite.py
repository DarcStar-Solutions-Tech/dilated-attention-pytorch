"""Comprehensive test suite for ImprovedDilatedAttention."""

import argparse
from typing import Dict, Any

import torch

from dilated_attention_pytorch import ImprovedDilatedAttention
from core.base_benchmark import BaseBenchmark
from core.utils import (
    generate_qkv_data,
    get_standard_configs,
    MemoryMonitor,
    Timer,
)


class ImprovedAttentionBenchmark(BaseBenchmark):
    """Benchmark suite for ImprovedDilatedAttention."""

    def __init__(self, config_preset: str = "small", **kwargs):
        """Initialize benchmark.

        Args:
            config_preset: Configuration preset name
            **kwargs: Additional arguments for BaseBenchmark
        """
        super().__init__(**kwargs)
        self.configs = get_standard_configs()[config_preset]

    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic functionality with various configurations."""
        results = []

        for seg_lens in zip(
            self.configs["segment_lengths"], self.configs["dilation_rates"]
        ):
            segment_lengths = seg_lens[: len(seg_lens) // 2]
            dilation_rates = seg_lens[len(seg_lens) // 2 :]

            # Create model
            model = (
                ImprovedDilatedAttention(
                    segment_lengths=list(segment_lengths),
                    dilation_rates=list(dilation_rates),
                    dropout=0.0,
                )
                .to(self.device)
                .to(self.dtype)
            )

            # Test with smallest sequence length
            seq_len = self.configs["seq_lengths"][0]
            batch_size = self.configs["batch_sizes"][0]
            num_heads = self.configs["num_heads"][0]
            head_dim = self.configs["head_dims"][0]

            q, k, v = generate_qkv_data(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                dtype=self.dtype,
                device=self.device,
            )

            try:
                with Timer(f"Forward pass - seq_len={seq_len}", verbose=False) as timer:
                    output = model(q, k, v)

                results.append(
                    {
                        "segment_lengths": segment_lengths,
                        "dilation_rates": dilation_rates,
                        "seq_len": seq_len,
                        "success": True,
                        "time": timer.elapsed,
                        "output_shape": list(output.shape),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "segment_lengths": segment_lengths,
                        "dilation_rates": dilation_rates,
                        "seq_len": seq_len,
                        "success": False,
                        "error": str(e),
                    }
                )

        return {"basic_functionality": results}

    def test_memory_scaling(self) -> Dict[str, Any]:
        """Test memory scaling with different sequence lengths."""
        results = []

        # Use first configuration
        segment_lengths = self.configs["segment_lengths"][:2]
        dilation_rates = self.configs["dilation_rates"][:2]

        model = (
            ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            )
            .to(self.device)
            .to(self.dtype)
        )

        batch_size = 1
        num_heads = self.configs["num_heads"][0]
        head_dim = self.configs["head_dims"][0]

        for seq_len in self.configs["seq_lengths"]:
            try:
                q, k, v = generate_qkv_data(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )

                with MemoryMonitor(self.device, f"seq_len={seq_len}") as monitor:
                    _ = model(q, k, v)

                results.append(
                    {
                        "seq_len": seq_len,
                        "memory_mb": monitor.memory_used,
                        "success": True,
                    }
                )

            except torch.cuda.OutOfMemoryError:
                results.append(
                    {
                        "seq_len": seq_len,
                        "memory_mb": -1,
                        "success": False,
                        "error": "OOM",
                    }
                )
                self.cleanup_memory()
            except Exception as e:
                results.append(
                    {
                        "seq_len": seq_len,
                        "memory_mb": -1,
                        "success": False,
                        "error": str(e),
                    }
                )

        return {"memory_scaling": results}

    def test_performance(self) -> Dict[str, Any]:
        """Test performance with different configurations."""
        results = []

        for seq_len in self.configs["seq_lengths"][:3]:  # Test first 3 sequence lengths
            for batch_size in self.configs["batch_sizes"][
                :2
            ]:  # Test first 2 batch sizes
                segment_lengths = self.configs["segment_lengths"][:2]
                dilation_rates = self.configs["dilation_rates"][:2]

                model = (
                    ImprovedDilatedAttention(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=0.0,
                    )
                    .to(self.device)
                    .to(self.dtype)
                )

                num_heads = self.configs["num_heads"][0]
                head_dim = self.configs["head_dims"][0]

                try:
                    q, k, v = generate_qkv_data(
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )

                    # Time the operation
                    timing_results = self.time_operation(
                        model, q, k, v, warmup=3, iterations=10
                    )

                    results.append(
                        {
                            "seq_len": seq_len,
                            "batch_size": batch_size,
                            "mean_time": timing_results["mean"],
                            "std_time": timing_results["std"],
                            "throughput": (batch_size * seq_len)
                            / timing_results["mean"],
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "seq_len": seq_len,
                            "batch_size": batch_size,
                            "error": str(e),
                        }
                    )

        return {"performance": results}

    def run(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        results = {}

        print("Testing basic functionality...")
        results.update(self.test_basic_functionality())

        print("Testing memory scaling...")
        results.update(self.test_memory_scaling())

        print("Testing performance...")
        results.update(self.test_performance())

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark ImprovedDilatedAttention")
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large", "extreme"],
        help="Configuration preset",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Create and run benchmark
    benchmark = ImprovedAttentionBenchmark(
        config_preset=args.config,
        device=torch.device(args.device),
        dtype=dtype,
    )

    results = benchmark.run()

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Basic functionality
    if "basic_functionality" in results:
        success_count = sum(
            1 for r in results["basic_functionality"] if r.get("success", False)
        )
        total_count = len(results["basic_functionality"])
        print(f"\nBasic Functionality: {success_count}/{total_count} passed")

    # Memory scaling
    if "memory_scaling" in results:
        print("\nMemory Scaling:")
        for r in results["memory_scaling"]:
            if r["success"]:
                print(f"  seq_len={r['seq_len']:,}: {r['memory_mb']:.2f} MB")
            else:
                print(f"  seq_len={r['seq_len']:,}: {r.get('error', 'Failed')}")

    # Performance
    if "performance" in results:
        print("\nPerformance (tokens/sec):")
        for r in results["performance"]:
            if "throughput" in r:
                print(
                    f"  seq_len={r['seq_len']:,}, batch={r['batch_size']}: "
                    f"{r['throughput']:,.0f} tokens/sec"
                )


if __name__ == "__main__":
    main()
