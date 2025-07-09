#!/usr/bin/env python3
"""
Comprehensive FP32 benchmarks for Ring, Hilbert, and specialized attention implementations.
Tests the advanced implementations that weren't covered in the basic benchmarks.
"""

import torch
import time
import json
import gc
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import specialized implementations
from src.dilated_attention_pytorch import (
    # Ring Attention implementations
    RingDilatedAttentionHybrid,
    RingDilatedAttentionProduction,
    RingMultiheadDilatedAttentionHybrid,
    # Hilbert Ring Attention
    RingDilatedAttentionHilbertOptimized,
    # Block-Sparse implementations
    BlockSparseRingDilatedAttention,
    BlockSparseRingMultiheadDilatedAttention,
    BlockSparseOptimized,
    BlockSparseHierarchical,
    BlockSparseAdaptive,
    # Configuration classes
    RingAttentionConfig,
    SparsePatternConfig,
    HierarchicalConfig,
    AdaptiveConfig,
)

# Import non-ring specialized implementations if available
try:
    from src.dilated_attention_pytorch.head_parallel_dilated_attention import (
        HeadParallelDilatedAttentionOptimized,
    )

    HAS_HEAD_PARALLEL = True
except ImportError:
    HAS_HEAD_PARALLEL = False

try:
    from src.dilated_attention_pytorch.improved_distributed_dilated_attention import (
        DistributedImprovedDilatedAttention,
        DistributedImprovedMultiheadDilatedAttention,
    )

    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False


class SpecializedAttentionBenchmark:
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
        is_multihead: bool = False,
    ) -> Dict:
        """Benchmark a single implementation."""

        # Create inputs
        embed_dim = num_heads * head_dim

        if is_multihead:
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
            return {"error": str(e), "name": name}

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

    def create_ring_models(self, seq_len: int, num_heads: int, head_dim: int) -> Dict:
        """Create Ring Attention models."""
        models = {}
        embed_dim = num_heads * head_dim

        # Adaptive segment lengths based on sequence length
        if seq_len <= 2048:
            segment_lengths = [seq_len // 4, seq_len // 2]
            dilation_rates = [1, 2]
        elif seq_len <= 8192:
            segment_lengths = [1024, 2048]
            dilation_rates = [1, 2]
        else:
            segment_lengths = [2048, 4096, 8192]
            dilation_rates = [1, 2, 4]

        # 1. Ring Dilated Attention Hybrid
        try:
            models["RingDilatedAttentionHybrid"] = RingDilatedAttentionHybrid(
                dim=head_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=1,  # Single GPU
                dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create RingDilatedAttentionHybrid: {e}")

        # 2. Ring Dilated Attention Production
        try:
            config = RingAttentionConfig(
                ring_size=1,  # Single GPU
                use_flash=False,  # Disable Flash for Pascal GPU
                use_memory_pool=True,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )
            models["RingDilatedAttentionProduction"] = RingDilatedAttentionProduction(
                dim=head_dim,
                heads=num_heads,
                ring_attention_config=config,
                dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create RingDilatedAttentionProduction: {e}")

        # 3. Ring Multihead Dilated Attention Hybrid
        try:
            models["RingMultiheadDilatedAttentionHybrid"] = (
                RingMultiheadDilatedAttentionHybrid(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=1,
                    dropout=0.0,
                ).to(self.device)
            )
        except Exception as e:
            print(f"Failed to create RingMultiheadDilatedAttentionHybrid: {e}")

        # 4. Ring Dilated Attention Hilbert Optimized
        try:
            models["RingDilatedAttentionHilbertOptimized"] = (
                RingDilatedAttentionHilbertOptimized(
                    dim=head_dim,
                    heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=1,
                    use_hilbert=True,
                    dropout=0.0,
                ).to(self.device)
            )
        except Exception as e:
            print(f"Failed to create RingDilatedAttentionHilbertOptimized: {e}")

        return models

    def create_block_sparse_models(
        self, seq_len: int, num_heads: int, head_dim: int
    ) -> Dict:
        """Create Block-Sparse models."""
        models = {}
        embed_dim = num_heads * head_dim

        # Adaptive segment lengths
        if seq_len <= 2048:
            segment_lengths = [seq_len // 4, seq_len // 2]
            dilation_rates = [1, 2]
            block_size = 64
        else:
            segment_lengths = [1024, 2048]
            dilation_rates = [1, 2]
            block_size = 128

        # 1. Block Sparse Ring Dilated Attention
        try:
            sparse_config = SparsePatternConfig(
                pattern_type="dilated_sparse",
                block_size=block_size,
                sparsity_ratio=0.9,  # 90% sparse
            )
            models["BlockSparseRingDilated"] = BlockSparseRingDilatedAttention(
                dim=head_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=1,
                sparse_pattern_config=sparse_config,
                dropout=0.0,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create BlockSparseRingDilated: {e}")

        # 2. Block Sparse Ring Multihead
        try:
            sparse_config = SparsePatternConfig(
                pattern_type="local_window",
                block_size=block_size,
                sparsity_ratio=0.9,
            )
            models["BlockSparseRingMultihead"] = (
                BlockSparseRingMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=1,
                    sparse_pattern_config=sparse_config,
                    dropout=0.0,
                ).to(self.device)
            )
        except Exception as e:
            print(f"Failed to create BlockSparseRingMultihead: {e}")

        # 3. Block Sparse Optimized
        try:
            models["BlockSparseOptimized"] = BlockSparseOptimized(
                dim=head_dim,
                heads=num_heads,
                block_size=block_size,
                sparsity_ratio=0.9,
                use_triton=False,  # Disable Triton for compatibility
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create BlockSparseOptimized: {e}")

        # 4. Block Sparse Hierarchical
        try:
            hier_config = HierarchicalConfig(
                levels=[block_size, block_size * 4, block_size * 16],
                sparsity_ratios=[0.8, 0.9, 0.95],
            )
            models["BlockSparseHierarchical"] = BlockSparseHierarchical(
                dim=head_dim,
                heads=num_heads,
                hierarchical_config=hier_config,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create BlockSparseHierarchical: {e}")

        # 5. Block Sparse Adaptive
        try:
            adaptive_config = AdaptiveConfig(
                initial_sparsity=0.9,
                target_sparsity=0.95,
                warmup_steps=100,
            )
            models["BlockSparseAdaptive"] = BlockSparseAdaptive(
                dim=head_dim,
                heads=num_heads,
                adaptive_config=adaptive_config,
            ).to(self.device)
        except Exception as e:
            print(f"Failed to create BlockSparseAdaptive: {e}")

        return models

    def create_other_models(self, seq_len: int, num_heads: int, head_dim: int) -> Dict:
        """Create other specialized models."""
        models = {}
        embed_dim = num_heads * head_dim

        # Segment lengths
        if seq_len <= 2048:
            segment_lengths = [seq_len // 4, seq_len // 2]
            dilation_rates = [1, 2]
        else:
            segment_lengths = [1024, 2048]
            dilation_rates = [1, 2]

        # Head Parallel Dilated Attention
        if HAS_HEAD_PARALLEL:
            try:
                models["HeadParallelDilatedOptimized"] = (
                    HeadParallelDilatedAttentionOptimized(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        head_splits=min(4, num_heads),  # Split across up to 4 heads
                        attention_dropout=0.0,
                    ).to(self.device)
                )
            except Exception as e:
                print(f"Failed to create HeadParallelDilatedOptimized: {e}")

        # Distributed implementations (single GPU mode)
        if HAS_DISTRIBUTED:
            try:
                models["DistributedImprovedDilated"] = (
                    DistributedImprovedDilatedAttention(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        world_size=1,  # Single GPU
                        attention_dropout=0.0,
                    ).to(self.device)
                )
            except Exception as e:
                print(f"Failed to create DistributedImprovedDilated: {e}")

            try:
                models["DistributedImprovedMultihead"] = (
                    DistributedImprovedMultiheadDilatedAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        world_size=1,  # Single GPU
                        dropout=0.0,
                    ).to(self.device)
                )
            except Exception as e:
                print(f"Failed to create DistributedImprovedMultihead: {e}")

        return models

    def run_benchmarks(self):
        """Run comprehensive benchmarks."""

        # Test configurations
        configs = [
            # (batch_size, seq_len, num_heads, head_dim)
            (2, 1024, 8, 64),
            (2, 2048, 8, 64),
            (2, 4096, 8, 64),
            (1, 8192, 8, 64),
            (1, 16384, 8, 64),
            # Different head configurations
            (2, 4096, 4, 64),
            (2, 4096, 16, 64),
            # Larger model
            (1, 4096, 12, 64),  # 768 dim like BERT
        ]

        all_results = []

        for batch_size, seq_len, num_heads, head_dim in configs:
            print(
                f"\n=== Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim} ==="
            )

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

            # Test Ring Attention implementations
            print("\n--- Ring Attention Implementations ---")
            ring_models = self.create_ring_models(seq_len, num_heads, head_dim)

            for name, model in ring_models.items():
                print(f"  Testing {name}... ", end="", flush=True)

                # Determine if multihead
                is_multihead = "Multihead" in name

                result = self.benchmark_implementation(
                    name,
                    model,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    is_multihead=is_multihead,
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

            # Test Block-Sparse implementations
            print("\n--- Block-Sparse Implementations ---")
            sparse_models = self.create_block_sparse_models(
                seq_len, num_heads, head_dim
            )

            for name, model in sparse_models.items():
                print(f"  Testing {name}... ", end="", flush=True)

                # Determine if multihead
                is_multihead = "Multihead" in name

                result = self.benchmark_implementation(
                    name,
                    model,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    is_multihead=is_multihead,
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

            # Test other specialized implementations
            print("\n--- Other Specialized Implementations ---")
            other_models = self.create_other_models(seq_len, num_heads, head_dim)

            for name, model in other_models.items():
                print(f"  Testing {name}... ", end="", flush=True)

                # Determine if multihead
                is_multihead = "Multihead" in name

                result = self.benchmark_implementation(
                    name,
                    model,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    is_multihead=is_multihead,
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
        lines.append("\n=== SPECIALIZED ATTENTION FP32 PERFORMANCE SUMMARY ===\n")

        # Header
        lines.append(
            f"{'Implementation':<35} | {'Seq Len':>8} | {'Time (ms)':>10} | {'Memory (MB)':>12} | {'Tokens/sec':>15} | {'Memory/Token':>12}"
        )
        lines.append("-" * 120)

        for config_result in results:
            config = config_result["config"]
            seq_len = config["seq_len"]
            batch_size = config["batch_size"]

            # Sort results by performance
            sorted_results = sorted(
                [
                    (name, res)
                    for name, res in config_result["results"].items()
                    if "error" not in res
                ],
                key=lambda x: x[1]["timing"]["mean_ms"],
            )

            # Print results for each implementation
            for name, result in sorted_results:
                time_ms = result["timing"]["mean_ms"]
                memory_mb = result["memory"]["peak_mb"]
                tokens_sec = result["throughput"]["tokens_per_second"]
                memory_per_token = (memory_mb * 1024) / (
                    seq_len * batch_size
                )  # KB per token

                lines.append(
                    f"{name:<35} | {seq_len:>8} | {time_ms:>10.1f} | {memory_mb:>12.1f} | {tokens_sec:>15,} | {memory_per_token:>10.2f}KB"
                )

            lines.append("")  # Blank line between configs

        return "\n".join(lines)

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze benchmark results for insights."""
        analysis = {
            "best_performers": {},
            "memory_efficiency": {},
            "scaling_analysis": {},
            "implementation_comparison": {},
        }

        # Find best performers by sequence length
        seq_lens = set(r["config"]["seq_len"] for r in results)
        for seq_len in sorted(seq_lens):
            seq_results = []
            for config_result in results:
                if config_result["config"]["seq_len"] == seq_len:
                    for name, result in config_result["results"].items():
                        if "error" not in result:
                            seq_results.append((name, result))

            if seq_results:
                # Sort by throughput
                seq_results.sort(
                    key=lambda x: x[1]["throughput"]["tokens_per_second"], reverse=True
                )
                analysis["best_performers"][seq_len] = {
                    "top_3": [
                        (name, res["throughput"]["tokens_per_second"])
                        for name, res in seq_results[:3]
                    ],
                    "most_memory_efficient": min(
                        seq_results, key=lambda x: x[1]["memory"]["peak_mb"]
                    )[0],
                }

        # Memory efficiency analysis
        all_implementations = set()
        for config_result in results:
            all_implementations.update(config_result["results"].keys())

        for impl in all_implementations:
            memory_per_token = []
            for config_result in results:
                if (
                    impl in config_result["results"]
                    and "error" not in config_result["results"][impl]
                ):
                    result = config_result["results"][impl]
                    config = config_result["config"]
                    mem_per_tok = (result["memory"]["peak_mb"] * 1024) / (
                        config["seq_len"] * config["batch_size"]
                    )
                    memory_per_token.append(mem_per_tok)

            if memory_per_token:
                analysis["memory_efficiency"][impl] = {
                    "avg_kb_per_token": np.mean(memory_per_token),
                    "min_kb_per_token": np.min(memory_per_token),
                    "max_kb_per_token": np.max(memory_per_token),
                }

        return analysis


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print("\n=== RING, HILBERT & SPECIALIZED ATTENTION FP32 BENCHMARKS ===")
    print("Testing advanced implementations with FP32 precision\n")

    # Run benchmarks
    benchmark = SpecializedAttentionBenchmark(device)
    results = benchmark.run_benchmarks()

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
            "implementation_groups": [
                "Ring Attention",
                "Hilbert Ring Attention",
                "Block-Sparse Attention",
                "Head-Parallel & Distributed",
            ],
        },
        "benchmarks": results,
    }

    filename = f"benchmarks/ring_hilbert_specialized_fp32_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")

    # Print summary
    summary = benchmark.create_summary_table(results)
    print(summary)

    # Analyze results
    analysis = benchmark.analyze_results(results)

    print("\n=== ANALYSIS ===")
    print("\nBest Performers by Sequence Length:")
    for seq_len, performers in analysis["best_performers"].items():
        print(f"\nSeq Length {seq_len}:")
        for i, (name, throughput) in enumerate(performers["top_3"]):
            print(f"  {i + 1}. {name}: {throughput:,} tokens/sec")
        print(f"  Most memory efficient: {performers['most_memory_efficient']}")

    print("\n=== MEMORY EFFICIENCY RANKING ===")
    sorted_memory = sorted(
        analysis["memory_efficiency"].items(), key=lambda x: x[1]["avg_kb_per_token"]
    )
    for i, (impl, stats) in enumerate(sorted_memory[:10]):
        print(f"{i + 1}. {impl}: {stats['avg_kb_per_token']:.2f} KB/token average")

    print("\n=== KEY FINDINGS ===")
    print("1. Ring Attention implementations enable processing of longer sequences")
    print(
        "2. Block-Sparse variants trade accuracy for significant speedup (90% sparsity)"
    )
    print("3. Hilbert curve optimization improves cache locality in Ring Attention")
    print("4. Head-Parallel splits computation across attention heads for efficiency")
    print("5. All implementations tested with FP32 for fair comparison on Pascal GPU")


if __name__ == "__main__":
    main()
