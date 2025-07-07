#!/usr/bin/env python3
"""
Benchmark all block-sparse implementations for performance comparison.

This script compares:
- Memory usage
- Forward pass speed
- Backward pass speed
- Sparsity effectiveness
"""

import torch
from typing import Dict, List, Tuple
import time
import gc
import sys
import os
from dataclasses import dataclass
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import (
    create_block_sparse_attention,
    get_block_sparse_preset,
)
from dilated_attention_pytorch import (
    SparsePatternConfig,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    batch_size: int = 2
    seq_lengths: List[int] = None
    num_heads: int = 12
    head_dim: int = 64
    num_warmup: int = 3
    num_iterations: int = 10
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [2048, 4096, 8192, 16384]


def measure_memory(func, *args, **kwargs):
    """Measure GPU memory usage of a function."""
    if not torch.cuda.is_available():
        return 0, func(*args, **kwargs)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Measure baseline
    baseline = torch.cuda.memory_allocated()

    # Run function
    result = func(*args, **kwargs)

    torch.cuda.synchronize()
    peak = torch.cuda.memory_allocated()

    memory_mb = (peak - baseline) / (1024 * 1024)
    return memory_mb, result


def benchmark_forward_pass(
    model,
    inputs: Tuple[torch.Tensor, ...],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Benchmark forward pass performance."""
    q, k, v = inputs

    # Warmup
    for _ in range(config.num_warmup):
        _ = model(q, k, v)

    # Measure memory
    torch.cuda.empty_cache()
    memory_mb, _ = measure_memory(model, q, k, v)

    # Measure time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(config.num_iterations):
            _ = model(q, k, v)
        end_event.record()

        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event)
    else:
        start_time = time.time()
        for _ in range(config.num_iterations):
            _ = model(q, k, v)
        total_time = (time.time() - start_time) * 1000

    avg_time = total_time / config.num_iterations

    return {
        "forward_time_ms": avg_time,
        "memory_mb": memory_mb,
    }


def benchmark_backward_pass(
    model,
    inputs: Tuple[torch.Tensor, ...],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Benchmark backward pass performance."""
    q, k, v = inputs

    # Enable gradients
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    # Warmup
    for _ in range(config.num_warmup):
        output = model(q, k, v)
        loss = output.mean()
        loss.backward()
        q.grad = None
        k.grad = None
        v.grad = None

    # Measure time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(config.num_iterations):
            output = model(q, k, v)
            loss = output.mean()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
        end_event.record()

        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event)
    else:
        start_time = time.time()
        for _ in range(config.num_iterations):
            output = model(q, k, v)
            loss = output.mean()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
        total_time = (time.time() - start_time) * 1000

    avg_time = total_time / config.num_iterations

    return {
        "backward_time_ms": avg_time,
    }


def create_test_inputs(
    seq_len: int,
    config: BenchmarkConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test input tensors."""
    shape = (config.batch_size, seq_len, config.num_heads, config.head_dim)

    q = torch.randn(shape, device=config.device, dtype=config.dtype)
    k = torch.randn(shape, device=config.device, dtype=config.dtype)
    v = torch.randn(shape, device=config.device, dtype=config.dtype)

    return q, k, v


def benchmark_implementation(
    name: str,
    model,
    seq_len: int,
    config: BenchmarkConfig,
) -> Dict[str, any]:
    """Benchmark a single implementation."""
    print(f"  Benchmarking {name} with seq_len={seq_len}...")

    # Create inputs
    inputs = create_test_inputs(seq_len, config)

    # Move model to device
    model = model.to(config.device)
    if hasattr(model, "eval"):
        model.eval()

    results = {
        "name": name,
        "seq_len": seq_len,
    }

    try:
        # Forward pass
        forward_results = benchmark_forward_pass(model, inputs, config)
        results.update(forward_results)

        # Backward pass
        backward_results = benchmark_backward_pass(model, inputs, config)
        results.update(backward_results)

        # Total time
        results["total_time_ms"] = (
            results["forward_time_ms"] + results["backward_time_ms"]
        )

        # Calculate throughput
        total_tokens = config.batch_size * seq_len * config.num_heads
        results["tokens_per_second"] = (
            total_tokens / results["forward_time_ms"]
        ) * 1000

    except Exception as e:
        print(f"    Error: {str(e)}")
        results["error"] = str(e)

    # Cleanup
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


def run_benchmarks(config: BenchmarkConfig) -> List[Dict]:
    """Run benchmarks for all implementations."""
    results = []

    # Define implementations to test
    implementations = [
        # Base implementation
        (
            "Base",
            lambda seq_len: create_block_sparse_attention(
                "base",
                sparse_config=SparsePatternConfig(
                    pattern_type="dilated_sparse",
                    sparsity_ratio=0.1,
                    block_size=64,
                ),
            ),
        ),
        # Hierarchical
        (
            "Hierarchical-Standard",
            lambda seq_len: create_block_sparse_attention(
                "hierarchical",
            ),
        ),
        (
            "Hierarchical-Long",
            lambda seq_len: get_block_sparse_preset("hierarchical_long"),
        ),
        # Adaptive (using fixed API)
        (
            "Adaptive",
            lambda seq_len: create_block_sparse_attention(
                "adaptive",
            ),
        ),
        # Multihead
        (
            "Multihead",
            lambda seq_len: create_block_sparse_attention(
                "multihead",
                embed_dim=config.num_heads * config.head_dim,
                num_heads=config.num_heads,
                sparsity_ratio=0.1,
            ),
        ),
        # Different sparsity levels
        (
            "Base-90%",
            lambda seq_len: create_block_sparse_attention(
                "base",
                sparsity_ratio=0.1,  # 90% sparse
            ),
        ),
        (
            "Base-95%",
            lambda seq_len: create_block_sparse_attention(
                "base",
                sparsity_ratio=0.05,  # 95% sparse
            ),
        ),
        (
            "Base-99%",
            lambda seq_len: create_block_sparse_attention(
                "base",
                sparsity_ratio=0.01,  # 99% sparse
            ),
        ),
        # Different patterns
        ("Local-Window", lambda seq_len: get_block_sparse_preset("local")),
        ("Global-Local", lambda seq_len: get_block_sparse_preset("global_local")),
    ]

    # Test each implementation at different sequence lengths
    for seq_len in config.seq_lengths:
        print(f"\nSequence length: {seq_len}")

        for name, create_fn in implementations:
            try:
                model = create_fn(seq_len)
                result = benchmark_implementation(name, model, seq_len, config)
                results.append(result)

                # Clean up
                del model

            except Exception as e:
                print(f"  Failed to create {name}: {str(e)}")
                results.append(
                    {
                        "name": name,
                        "seq_len": seq_len,
                        "error": str(e),
                    }
                )

    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze benchmark results."""
    analysis = {}

    # Group by sequence length
    by_seq_len = {}
    for result in results:
        seq_len = result["seq_len"]
        if seq_len not in by_seq_len:
            by_seq_len[seq_len] = []
        by_seq_len[seq_len].append(result)

    # Find best performers
    for seq_len, seq_results in by_seq_len.items():
        valid_results = [r for r in seq_results if "error" not in r]

        if valid_results:
            # Fastest forward
            fastest_forward = min(valid_results, key=lambda r: r["forward_time_ms"])

            # Lowest memory
            lowest_memory = min(valid_results, key=lambda r: r["memory_mb"])

            # Best throughput
            best_throughput = max(valid_results, key=lambda r: r["tokens_per_second"])

            analysis[seq_len] = {
                "fastest_forward": fastest_forward["name"],
                "fastest_forward_time": fastest_forward["forward_time_ms"],
                "lowest_memory": lowest_memory["name"],
                "lowest_memory_mb": lowest_memory["memory_mb"],
                "best_throughput": best_throughput["name"],
                "best_throughput_tokens": best_throughput["tokens_per_second"],
            }

    return analysis


def save_results(results: List[Dict], analysis: Dict, config: BenchmarkConfig):
    """Save benchmark results to file."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"block_sparse_benchmark_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "config": {
            "batch_size": config.batch_size,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "device": config.device,
            "dtype": str(config.dtype),
        },
        "results": results,
        "analysis": analysis,
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


def print_summary(results: List[Dict], analysis: Dict):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for seq_len in sorted(analysis.keys()):
        seq_analysis = analysis[seq_len]
        print(f"\nSequence Length: {seq_len}")
        print(
            f"  Fastest Forward: {seq_analysis['fastest_forward']} "
            f"({seq_analysis['fastest_forward_time']:.2f} ms)"
        )
        print(
            f"  Lowest Memory: {seq_analysis['lowest_memory']} "
            f"({seq_analysis['lowest_memory_mb']:.2f} MB)"
        )
        print(
            f"  Best Throughput: {seq_analysis['best_throughput']} "
            f"({seq_analysis['best_throughput_tokens']:.0f} tokens/sec)"
        )

    # Print detailed table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print(
        f"{'Implementation':<25} {'Seq Len':<10} {'Forward (ms)':<12} "
        f"{'Memory (MB)':<12} {'Tokens/sec':<15}"
    )
    print("-" * 80)

    for result in results:
        if "error" not in result:
            print(
                f"{result['name']:<25} {result['seq_len']:<10} "
                f"{result['forward_time_ms']:<12.2f} "
                f"{result['memory_mb']:<12.2f} "
                f"{result['tokens_per_second']:<15.0f}"
            )


def main():
    """Run block-sparse benchmarks."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU (will be slow).")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Configure benchmark
    config = BenchmarkConfig(
        batch_size=2,
        seq_lengths=[2048, 4096, 8192],  # Add 16384 if you have enough memory
        num_heads=12,
        head_dim=64,
        device=device,
        dtype=dtype,
    )

    print("Block-Sparse Attention Benchmarks")
    print(f"Config: {config}")

    # Run benchmarks
    results = run_benchmarks(config)

    # Analyze results
    analysis = analyze_results(results)

    # Print summary
    print_summary(results, analysis)

    # Save results
    save_results(results, analysis, config)


if __name__ == "__main__":
    main()
