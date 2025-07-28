#!/usr/bin/env python3
"""Comprehensive benchmark of all ring attention implementations."""

import torch
import torch.distributed as dist
import os
import sys
import gc
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import all ring attention implementations
from dilated_attention_pytorch import (
    RingDilatedAttentionSDPA,  # noqa: E402
    StandardRingAttention,  # noqa: E402
    HilbertRingAttention,  # noqa: E402
    DistributedRingAttention,  # noqa: E402
    BlockSparseRingDilatedAttention,  # noqa: E402
    BlockSparseRingMultiheadDilatedAttention,  # noqa: E402
    BlockSparseAdaptive,  # noqa: E402
    RingDilatedAttentionCorrect,  # noqa: E402
    RingDilatedAttentionHilbertGPUOptimized,  # noqa: E402
    EnterpriseDistributedDilatedAttention,  # noqa: E402
)
from dilated_attention_pytorch.utils import get_optimal_dtype  # noqa: E402

# Apply communication fixes for SDPA variant
from dilated_attention_pytorch.ring.utils import ring_communication_fix  # noqa: E402
from dilated_attention_pytorch.ring.base import ring_dilated_attention_sdpa  # noqa: E402

ring_dilated_attention_sdpa.ring_pass_kv_safe = (
    ring_communication_fix.ring_pass_kv_fixed
)


# Ring attention configurations to test
RING_IMPLEMENTATIONS = {
    "RingDilatedAttentionSDPA": {
        "class": RingDilatedAttentionSDPA,
        "kwargs": {},
        "description": "Ring attention with PyTorch SDPA",
        "multi_gpu": True,
    },
    "StandardRingAttention": {
        "class": StandardRingAttention,
        "kwargs": {},
        "description": "Standard ring attention implementation",
        "multi_gpu": True,
    },
    "RingDilatedAttentionCorrect": {
        "class": RingDilatedAttentionCorrect,
        "kwargs": {},
        "description": "Reference correct implementation",
        "multi_gpu": True,
    },
    "HilbertRingAttention": {
        "class": HilbertRingAttention,
        "kwargs": {},
        "description": "Ring attention with Hilbert optimization",
        "multi_gpu": True,
    },
    "RingDilatedAttentionHilbertGPUOptimized": {
        "class": RingDilatedAttentionHilbertGPUOptimized,
        "kwargs": {},
        "description": "GPU-optimized Hilbert ring attention",
        "multi_gpu": True,
    },
    "DistributedRingAttention": {
        "class": DistributedRingAttention,
        "kwargs": {"enable_monitoring": False},
        "description": "Enterprise distributed ring attention",
        "multi_gpu": True,
    },
    "BlockSparseRingDilatedAttention": {
        "class": BlockSparseRingDilatedAttention,
        "kwargs": {"pattern_type": "dilated_sparse", "sparsity_ratio": 0.25},
        "description": "Block-sparse ring attention",
        "multi_gpu": True,
    },
    "BlockSparseRingMultiheadDilatedAttention": {
        "class": BlockSparseRingMultiheadDilatedAttention,
        "kwargs": {"pattern_type": "local_window", "sparsity_ratio": 0.25},
        "description": "Multihead block-sparse ring attention",
        "multi_gpu": True,
    },
    "BlockSparseAdaptive": {
        "class": BlockSparseAdaptive,
        "kwargs": {},
        "description": "Adaptive block-sparse ring attention",
        "multi_gpu": True,
    },
    "EnterpriseDistributedDilatedAttention": {
        "class": EnterpriseDistributedDilatedAttention,
        "kwargs": {"enable_monitoring": False},
        "description": "Enterprise-grade distributed attention",
        "multi_gpu": True,
    },
}


def benchmark_implementation(
    impl_name: str,
    impl_config: Dict,
    seq_len: int,
    batch_size: int,
    embed_dim: int,
    num_heads: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 3,
) -> Optional[Dict]:
    """Benchmark a single ring attention implementation."""
    try:
        # Skip if multi-GPU required but not available
        if impl_config["multi_gpu"] and (
            not dist.is_initialized() or dist.get_world_size() == 1
        ):
            if "Distributed" in impl_name or "BlockSparseRingDistributed" in impl_name:
                print(f"  Skipping {impl_name} - requires multi-GPU")
                return None

        print(f"\n  Testing {impl_name}...")

        # Create model
        impl_class = impl_config["class"]
        kwargs = impl_config["kwargs"].copy()

        # Add common parameters
        kwargs.update(
            {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "segment_lengths": segment_lengths,
                "dilation_rates": dilation_rates,
                "dropout": 0.0,
                "device": device,
                "dtype": dtype,
            }
        )

        # Some implementations might not support all parameters
        try:
            model = impl_class(**kwargs)
        except TypeError as e:
            # Try without dtype/device for older implementations
            kwargs.pop("dtype", None)
            kwargs.pop("device", None)
            try:
                model = impl_class(**kwargs)
                model = model.to(device=device, dtype=dtype)
            except Exception:
                print(f"    Failed to initialize: {e}")
                return None

        model.eval()

        # Create input
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                try:
                    _ = model(x)
                except Exception as e:
                    print(f"    Forward pass failed: {e}")
                    return None

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Measure
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**2)

        # Time multiple iterations
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iters):
            with torch.no_grad():
                output = model(x)

        torch.cuda.synchronize()
        end_time = time.time()

        # Get stats
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        mem_used = peak_mem - mem_before
        avg_time = (end_time - start_time) / num_iters
        throughput = (batch_size * seq_len) / avg_time

        result = {
            "implementation": impl_name,
            "description": impl_config["description"],
            "time_per_iter": avg_time,
            "throughput": throughput,
            "peak_memory_mb": peak_mem,
            "memory_used_mb": mem_used,
            "memory_per_token_kb": mem_used / (batch_size * seq_len) * 1024,
            "output_shape": list(output.shape),
            "success": True,
        }

        print(
            f"    ✓ Time: {avg_time:.3f}s, Memory: {mem_used:.1f}MB, Throughput: {throughput:,.0f} tok/s"
        )

        # Cleanup
        del model, x, output
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "implementation": impl_name,
            "description": impl_config["description"],
            "success": False,
            "error": str(e),
        }


def main():
    # Configuration
    embed_dim = 512
    num_heads = 8
    batch_size = 2

    # Test configurations
    test_configs = [
        # (seq_len, segment_lengths, dilation_rates)
        (2048, [512, 1024], [1, 2]),
        (4096, [512, 1024, 2048], [1, 2, 4]),
        (8192, [1024, 2048, 4096], [1, 2, 4]),
    ]

    # Initialize distributed if available
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dtype = get_optimal_dtype(device)

    print(f"\n{'=' * 80}")
    print("Comprehensive Ring Attention Implementation Benchmark")
    print(f"{'=' * 80}")
    print(f"Rank: {rank}, World size: {world_size}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Batch size: {batch_size}")
    print(f"\nImplementations to test: {len(RING_IMPLEMENTATIONS)}")

    all_results = []

    for seq_len, segment_lengths, dilation_rates in test_configs:
        # Skip if not divisible
        if seq_len % max(segment_lengths) != 0:
            continue

        print(f"\n{'-' * 60}")
        print(f"Testing sequence length: {seq_len}")
        print(f"Segments: {segment_lengths}, Dilations: {dilation_rates}")

        config_results = []

        for impl_name, impl_config in RING_IMPLEMENTATIONS.items():
            result = benchmark_implementation(
                impl_name,
                impl_config,
                seq_len,
                batch_size,
                embed_dim,
                num_heads,
                segment_lengths,
                dilation_rates,
                device,
                dtype,
            )

            if result:
                result.update(
                    {
                        "sequence_length": seq_len,
                        "world_size": world_size,
                        "rank": rank,
                        "batch_size": batch_size,
                        "segment_lengths": segment_lengths,
                        "dilation_rates": dilation_rates,
                    }
                )
                config_results.append(result)
                all_results.append(result)

        # Print summary for this configuration
        if rank == 0 and config_results:
            print(f"\nSummary for seq_len={seq_len}:")
            print(
                f"{'Implementation':<40} {'Time(s)':<10} {'Memory(MB)':<12} {'Throughput':<15}"
            )
            print("-" * 80)

            # Sort by throughput
            config_results.sort(
                key=lambda x: x.get("throughput", 0) if x.get("success", False) else 0,
                reverse=True,
            )

            for r in config_results:
                if r.get("success", False):
                    print(
                        f"{r['implementation']:<40} "
                        f"{r['time_per_iter']:<10.3f} "
                        f"{r['memory_used_mb']:<12.1f} "
                        f"{r['throughput']:<15,.0f}"
                    )

    # Save results
    if rank == 0:
        output_dir = Path("benchmarks/results/ring/all_implementations")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_results_w{world_size}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_file}")

        # Overall summary
        successful_results = [r for r in all_results if r.get("success", False)]
        if successful_results:
            print(f"\nOverall Summary (World Size: {world_size}):")
            print(
                f"Successfully tested: {len(set(r['implementation'] for r in successful_results))} implementations"
            )

            # Find best performers
            by_seq_len = {}
            for r in successful_results:
                seq_len = r["sequence_length"]
                if seq_len not in by_seq_len:
                    by_seq_len[seq_len] = []
                by_seq_len[seq_len].append(r)

            print("\nBest performers by sequence length:")
            for seq_len in sorted(by_seq_len.keys()):
                results = by_seq_len[seq_len]
                best_throughput = max(results, key=lambda x: x["throughput"])
                best_memory = min(results, key=lambda x: x["memory_used_mb"])

                print(f"\n  Seq length {seq_len}:")
                print(
                    f"    Fastest: {best_throughput['implementation']} ({best_throughput['throughput']:,.0f} tok/s)"
                )
                print(
                    f"    Most memory efficient: {best_memory['implementation']} ({best_memory['memory_used_mb']:.1f} MB)"
                )

    # Cleanup distributed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
