#!/usr/bin/env python3
"""
Benchmark Flash Attention 3 performance and capabilities.

This script benchmarks FA3 vs FA2 vs standard attention across:
- Different sequence lengths
- Various sparsity patterns
- H100-specific optimizations
- FP8 precision (if available)
"""

import argparse
import json
import os
import sys
from datetime import datetime

from pathlib import Path
import torch

# Add parent directory to path

# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch import (
    BlockSparseRingDilatedAttention,
)
from dilated_attention_pytorch.core.constants import (
    GPU_TYPE,
    HAS_FLASH_ATTN,
    HAS_FLASH_ATTN_3,
)
from dilated_attention_pytorch.utils.attention_utils import get_flash_attention_info
from dilated_attention_pytorch.utils.flash_attention_3_utils import benchmark_fa3_vs_fa2


def print_system_info():
    """Print system and Flash Attention information."""
    print("=" * 80)
    print("FLASH ATTENTION 3 BENCHMARK")
    print("=" * 80)

    # System info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Type: {GPU_TYPE}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # Flash Attention info
    fa_info = get_flash_attention_info()
    print("\nFlash Attention Info:")
    print(f"  Has Flash Attention: {fa_info['has_flash_attn']}")
    print(f"  Has Flash Attention 3: {fa_info['has_flash_attn_3']}")
    print(f"  Version: {fa_info['version']}")
    print(f"  FA3 Optimized Hardware: {fa_info['fa3_optimized_hardware']}")

    print("-" * 80)


def benchmark_attention_backends(
    batch_size: int = 2,
    seq_len: int = 8192,
    num_heads: int = 8,
    head_dim: int = 64,
    num_runs: int = 100,
    use_fp16: bool = True,
) -> dict:
    """Benchmark different attention backends."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 else torch.float32

    results = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "num_runs": num_runs,
        },
        "backends": {},
    }

    # Create test tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"\nBenchmarking with seq_len={seq_len}, batch_size={batch_size}...")

    # 1. Standard PyTorch attention
    print("  Testing standard PyTorch attention...")
    try:
        # Warmup
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)
        end.record()

        torch.cuda.synchronize()
        results["backends"]["pytorch"] = {
            "time_ms": start.elapsed_time(end) / num_runs,
            "memory_mb": torch.cuda.memory_allocated() / 1024**2,
        }
    except Exception as e:
        results["backends"]["pytorch"] = {"error": str(e)}

    # 2. PyTorch SDPA
    print("  Testing PyTorch SDPA...")
    try:
        import torch.nn.functional as F  # noqa: N812

        # Reshape for SDPA
        q_sdpa = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)

        # Warmup
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        end.record()

        torch.cuda.synchronize()
        results["backends"]["sdpa"] = {
            "time_ms": start.elapsed_time(end) / num_runs,
            "memory_mb": torch.cuda.memory_allocated() / 1024**2,
        }
    except Exception as e:
        results["backends"]["sdpa"] = {"error": str(e)}

    # 3. Flash Attention 2/3 comparison
    if HAS_FLASH_ATTN:
        print("  Testing Flash Attention...")
        fa_results = benchmark_fa3_vs_fa2(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            num_runs=num_runs,
        )

        if "fa2_ms" in fa_results:
            results["backends"]["flash_attn_2"] = {
                "time_ms": fa_results["fa2_ms"],
                "memory_mb": torch.cuda.memory_allocated() / 1024**2,
            }

        if "fa3_ms" in fa_results:
            results["backends"]["flash_attn_3"] = {
                "time_ms": fa_results["fa3_ms"],
                "memory_mb": torch.cuda.memory_allocated() / 1024**2,
                "speedup_vs_fa2": fa_results.get("speedup", 1.0),
            }

    return results


def benchmark_sparse_patterns(
    seq_len: int = 16384,
    sparsity_ratios: list[float] | None = None,
    pattern_types: list[str] | None = None,
) -> dict:
    """Benchmark different sparse patterns with FA3."""
    if sparsity_ratios is None:
        sparsity_ratios = [0.9, 0.95, 0.99]
    if pattern_types is None:
        pattern_types = ["local_window", "dilated_sparse", "global_local"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_heads = 8
    head_dim = 64

    results = {"patterns": {}}

    print(f"\nBenchmarking sparse patterns (seq_len={seq_len})...")

    for pattern in pattern_types:
        results["patterns"][pattern] = {}

        for sparsity in sparsity_ratios:
            print(f"  Testing {pattern} with {sparsity * 100:.0f}% sparsity...")

            try:
                # Create sparse attention module
                attention = BlockSparseRingDilatedAttention(
                    segment_lengths=[seq_len],
                    dilation_rates=[1],
                    sparse_config={
                        "pattern_type": pattern,
                        "sparsity_ratio": 1 - sparsity,
                        "block_size": 128,
                    },
                    device=device,
                )

                # Create inputs
                q = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                # Warmup
                for _ in range(5):
                    _ = attention(q, k, v)

                # Benchmark
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                for _ in range(20):
                    _ = attention(q, k, v)
                end.record()

                torch.cuda.synchronize()

                results["patterns"][pattern][f"sparsity_{sparsity}"] = {
                    "time_ms": start.elapsed_time(end) / 20,
                    "memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "actual_sparsity": sparsity,
                }

            except Exception as e:
                results["patterns"][pattern][f"sparsity_{sparsity}"] = {"error": str(e)}

            torch.cuda.empty_cache()

    return results


def benchmark_sequence_scaling() -> dict:
    """Benchmark how performance scales with sequence length."""
    seq_lengths = [2048, 4096, 8192, 16384, 32768]
    if str(GPU_TYPE) in ["h100", "a100"]:
        seq_lengths.extend([65536, 131072])

    results = {"scaling": {}}

    print("\nBenchmarking sequence length scaling...")

    for seq_len in seq_lengths:
        print(f"  Testing seq_len={seq_len}...")

        try:
            backend_results = benchmark_attention_backends(
                seq_len=seq_len,
                num_runs=20 if seq_len <= 16384 else 10,
            )

            results["scaling"][seq_len] = backend_results["backends"]

        except Exception as e:
            results["scaling"][seq_len] = {"error": str(e)}

        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention 3")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--skip-sparse", action="store_true", help="Skip sparse pattern benchmarks"
    )
    parser.add_argument(
        "--skip-scaling", action="store_true", help="Skip sequence scaling benchmarks"
    )
    parser.add_argument("--output-file", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    # Print system info
    print_system_info()

    # Run benchmarks
    all_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H%M-UTC"),
        "system": {
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU",
            "gpu_type": str(GPU_TYPE),
            "has_fa2": HAS_FLASH_ATTN,
            "has_fa3": HAS_FLASH_ATTN_3,
        },
    }

    # 1. Backend comparison
    print("\n" + "=" * 60)
    print("BACKEND COMPARISON")
    print("=" * 60)
    backend_results = benchmark_attention_backends(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_runs=args.num_runs,
    )
    all_results["backend_comparison"] = backend_results

    # Print results
    print("\nResults:")
    for backend, metrics in backend_results["backends"].items():
        if "error" in metrics:
            print(f"  {backend}: ERROR - {metrics['error']}")
        else:
            print(
                f"  {backend}: {metrics['time_ms']:.2f} ms, {metrics.get('memory_mb', 0):.1f} MB"
            )

    # 2. Sparse patterns (if not skipped)
    if not args.skip_sparse:
        print("\n" + "=" * 60)
        print("SPARSE PATTERN BENCHMARKS")
        print("=" * 60)
        sparse_results = benchmark_sparse_patterns()
        all_results["sparse_patterns"] = sparse_results

    # 3. Sequence scaling (if not skipped)
    if not args.skip_scaling:
        print("\n" + "=" * 60)
        print("SEQUENCE SCALING BENCHMARKS")
        print("=" * 60)
        scaling_results = benchmark_sequence_scaling()
        all_results["sequence_scaling"] = scaling_results

    # Save results if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if HAS_FLASH_ATTN_3 and str(GPU_TYPE) in ["h100", "h800"]:
        print("✅ Flash Attention 3 is available and optimized for your hardware!")
        print("   Expected benefits:")
        print("   - 1.5-2x faster than Flash Attention 2")
        print("   - Native block-sparse support")
        print("   - FP8 precision for even faster computation")
        print("   - Asynchronous execution with warp specialization")
    elif HAS_FLASH_ATTN_3:
        print("⚠️  Flash Attention 3 is available but not optimized for your hardware")
        print("   Consider using H100/H800 GPUs for maximum performance")
    elif HAS_FLASH_ATTN:
        print("📊 Using Flash Attention 2")
        print("   Upgrade to Flash Attention 3 for better performance on H100")
    else:
        print("⚠️  Flash Attention not available")
        print("   Install flash-attn>=2.8.0 for significant speedups")


if __name__ == "__main__":
    main()

    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="flash-attention-3",
        parameters={}
    )
    
    # Add results
    output_manager.add_result("results", results)
    
    # Save results
    output_paths = output_manager.save_results()
    print(f"\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")
