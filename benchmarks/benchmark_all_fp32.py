#!/usr/bin/env python3
"""
Comprehensive FP32 benchmarks for all DilatedAttention implementations.
Specifically for Pascal GPUs where FP16 is severely limited.
"""

import torch
import time
import json
import gc
from datetime import datetime
from typing import Dict, List
import numpy as np

# Import all implementations
from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
    ImprovedMultiheadDilatedAttention,
    create_multihead_dilated_attention,
)


class FP32BenchmarkSuite:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.float32  # Force FP32
        self.results = {}

    def benchmark_implementation(
        self,
        name: str,
        model,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> Dict:
        """Benchmark a single implementation."""

        # Create inputs
        embed_dim = num_heads * head_dim

        # Different input shapes for different implementations
        if "Multihead" in name or "multihead" in str(type(model)):
            # Multihead expects (batch, seq, embed_dim)
            x = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
            )
            q, k, v = x, x, x
        else:
            # Raw attention expects (batch, seq, heads, head_dim)
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

        # Clear cache and measure initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_start = torch.cuda.memory_allocated() / 1024**2

        # Warmup
        try:
            for _ in range(num_warmup):
                with torch.no_grad():
                    output = model(q, k, v)
                    # Handle tuple returns
                    if isinstance(output, tuple):
                        output = output[0]
        except Exception as e:
            return {"error": str(e)}

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v)
                if isinstance(output, tuple):
                    output = output[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Get memory stats
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            final_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_memory = final_memory = 0

        times_ms = np.array(times) * 1000

        return {
            "implementation": name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": "float32",
            "timing": {
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "min_ms": float(np.min(times_ms)),
                "max_ms": float(np.max(times_ms)),
                "median_ms": float(np.median(times_ms)),
            },
            "memory": {
                "start_mb": float(mem_start),
                "peak_mb": float(peak_memory),
                "final_mb": float(final_memory),
                "delta_mb": float(peak_memory - mem_start),
            },
            "throughput": {
                "tokens_per_second": int(
                    (batch_size * seq_len) * 1000 / np.mean(times_ms)
                ),
                "sequences_per_second": float(1000 / np.mean(times_ms)),
            },
        }

    def create_models(self, seq_len: int, num_heads: int, head_dim: int) -> Dict:
        """Create all model variants."""
        embed_dim = num_heads * head_dim

        # Adaptive segment lengths based on sequence length
        if seq_len <= 1024:
            segment_lengths = [seq_len // 4, seq_len // 2]
            dilation_rates = [1, 2]
        elif seq_len <= 4096:
            segment_lengths = [seq_len // 4, seq_len // 2]
            dilation_rates = [1, 2]
        else:
            segment_lengths = [1024, 2048, 4096]
            dilation_rates = [1, 2, 4]

        models = {}

        # 1. Original implementations
        try:
            models["DilatedAttention"] = DilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                attention_dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create DilatedAttention: {e}")

        try:
            models["ImprovedDilatedAttention"] = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                attention_dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create ImprovedDilatedAttention: {e}")

        # 2. Multihead variants
        try:
            models["MultiheadDilatedAttention"] = MultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create MultiheadDilatedAttention: {e}")

        try:
            models["ImprovedMultiheadDilatedAttention"] = (
                ImprovedMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                ).to(self.device)
            )
        except Exception as e:
            print(f"Failed to create ImprovedMultiheadDilatedAttention: {e}")

        # 3. Factory-created variants
        try:
            models["Factory-Auto"] = create_multihead_dilated_attention(
                "auto",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create Factory-Auto: {e}")

        # 4. Ring attention (if sequence is long enough)
        if seq_len >= 2048:
            try:
                from dilated_attention_pytorch import RingDilatedAttentionHybrid

                models["RingDilatedAttentionHybrid"] = RingDilatedAttentionHybrid(
                    dim=head_dim,
                    heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=2,
                    dropout=0.0,
                ).to(self.device)
            except Exception as e:
                print(f"Failed to create RingDilatedAttentionHybrid: {e}")

        # 5. Block-sparse (if available)
        try:
            from dilated_attention_pytorch import BlockSparseRingDilatedAttention
            from dilated_attention_pytorch.core import RingAttentionConfig

            config = RingAttentionConfig(
                ring_size=1,  # Single GPU
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )

            models["BlockSparseRingDilated"] = BlockSparseRingDilatedAttention(
                dim=head_dim,
                heads=num_heads,
                ring_attention_config=config,
                block_size=128,
                sparsity_ratio=0.9,  # 90% sparse
                dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create BlockSparseRingDilated: {e}")

        return models

    def run_benchmarks(self):
        """Run comprehensive benchmarks."""

        # Test configurations
        configs = [
            # (batch_size, seq_len, num_heads, head_dim)
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (2, 2048, 8, 64),
            (2, 4096, 8, 64),
            (1, 8192, 8, 64),
            # Different head configurations
            (2, 2048, 4, 64),
            (2, 2048, 16, 64),
            # Larger model
            (1, 2048, 12, 64),  # 768 dim like BERT
        ]

        all_results = []

        for batch_size, seq_len, num_heads, head_dim in configs:
            print(
                f"\n=== Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim} ==="
            )

            # Create models for this configuration
            models = self.create_models(seq_len, num_heads, head_dim)

            config_results = {
                "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "embed_dim": num_heads * head_dim,
                },
                "results": {},
            }

            # Benchmark each model
            for name, model in models.items():
                print(f"  Testing {name}... ", end="", flush=True)

                result = self.benchmark_implementation(
                    name, model, batch_size, seq_len, num_heads, head_dim
                )

                if "error" in result:
                    print(f"✗ Failed: {result['error']}")
                else:
                    print(
                        f"✓ {result['timing']['mean_ms']:.1f}ms, {result['memory']['peak_mb']:.0f}MB, "
                        f"{result['throughput']['tokens_per_second']:,} tokens/sec"
                    )

                config_results["results"][name] = result

                # Clear memory between tests
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            all_results.append(config_results)

        return all_results

    def create_summary_table(self, results: List[Dict]) -> str:
        """Create a summary table of results."""
        lines = []
        lines.append("\n=== FP32 PERFORMANCE SUMMARY ===\n")

        # Header
        lines.append(
            f"{'Implementation':<30} | {'Seq Len':>8} | {'Time (ms)':>10} | {'Memory (MB)':>12} | {'Tokens/sec':>15} | {'vs Original':>12}"
        )
        lines.append("-" * 110)

        for config_result in results:
            config = config_result["config"]
            seq_len = config["seq_len"]

            # Get baseline (DilatedAttention) performance
            baseline = config_result["results"].get("DilatedAttention", {})
            if baseline and "error" not in baseline:
                baseline_time = baseline["timing"]["mean_ms"]
                _ = baseline["throughput"]["tokens_per_second"]
            else:
                baseline_time = _ = None

            # Print results for each implementation
            for name, result in config_result["results"].items():
                if "error" not in result:
                    time_ms = result["timing"]["mean_ms"]
                    memory_mb = result["memory"]["peak_mb"]
                    tokens_sec = result["throughput"]["tokens_per_second"]

                    if baseline_time:
                        speedup = baseline_time / time_ms
                        speedup_str = f"{speedup:.2f}x"
                    else:
                        speedup_str = "N/A"

                    lines.append(
                        f"{name:<30} | {seq_len:>8} | {time_ms:>10.1f} | {memory_mb:>12.1f} | {tokens_sec:>15,} | {speedup_str:>12}"
                    )

            lines.append("")  # Blank line between configs

        return "\n".join(lines)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print("\n=== COMPREHENSIVE FP32 BENCHMARKS ===")
    print("Using FP32 for fair comparison on Pascal GPU\n")

    # Run benchmarks
    suite = FP32BenchmarkSuite(device)
    results = suite.run_benchmarks()

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "N/A",
            "dtype": "float32",
            "pytorch_version": torch.__version__,
        },
        "benchmarks": results,
    }

    filename = f"benchmarks/all_implementations_fp32_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")

    # Print summary
    summary = suite.create_summary_table(results)
    print(summary)

    # Find best performers
    print("\n=== TOP PERFORMERS BY SEQUENCE LENGTH ===")

    seq_lens = set(r["config"]["seq_len"] for r in results)
    for seq_len in sorted(seq_lens):
        print(f"\nSeq Length {seq_len}:")

        # Collect all results for this sequence length
        seq_results = []
        for config_result in results:
            if config_result["config"]["seq_len"] == seq_len:
                for name, result in config_result["results"].items():
                    if "error" not in result:
                        seq_results.append((name, result))

        # Sort by throughput
        seq_results.sort(
            key=lambda x: x[1]["throughput"]["tokens_per_second"], reverse=True
        )

        # Show top 3
        for i, (name, result) in enumerate(seq_results[:3]):
            print(
                f"  {i + 1}. {name}: {result['throughput']['tokens_per_second']:,} tokens/sec "
                f"({result['timing']['mean_ms']:.1f}ms)"
            )


if __name__ == "__main__":
    main()
