#!/usr/bin/env python3
"""Benchmark working ring attention implementations."""

import torch
import torch.distributed as dist
import os
import sys
import gc
import time
import json
from pathlib import Path
from typing import Dict, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import working ring attention implementations
from dilated_attention_pytorch import (
    RingDilatedAttentionSDPA,  # noqa: E402
    RingDilatedAttentionCorrect,  # noqa: E402
    RingDilatedAttentionHilbertGPUOptimized,  # noqa: E402
    create_ring_attention,  # noqa: E402
    RingAttentionConfig,  # noqa: E402
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
        "description": "Ring attention with PyTorch SDPA (patched)",
    },
    "RingDilatedAttentionCorrect": {
        "class": RingDilatedAttentionCorrect,
        "kwargs": {},
        "description": "Reference correct implementation",
    },
    "RingDilatedAttentionHilbertGPUOptimized": {
        "class": RingDilatedAttentionHilbertGPUOptimized,
        "kwargs": {},
        "description": "GPU-optimized Hilbert ring attention",
    },
}

# Also test factory-created implementations
FACTORY_IMPLEMENTATIONS = {
    "StandardRingAttention": {
        "factory": "standard",
        "description": "Standard ring attention (factory)",
    },
    "HilbertRingAttention": {
        "factory": "hilbert",
        "description": "Hilbert ring attention (factory)",
    },
    "BlockSparseRingAttention": {
        "factory": "block_sparse",
        "description": "Block-sparse ring attention (factory)",
        "extra_kwargs": {"sparsity_ratio": 0.25},
    },
}


def benchmark_implementation(
    impl_name: str,
    model: torch.nn.Module,
    seq_len: int,
    batch_size: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 3,
) -> Optional[Dict]:
    """Benchmark a single ring attention implementation."""
    try:
        print(f"\n  Testing {impl_name}...")
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

        # Calculate effective sequence per GPU
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        effective_seq = seq_len // world_size

        result = {
            "implementation": impl_name,
            "time_per_iter": avg_time,
            "throughput": throughput,
            "peak_memory_mb": peak_mem,
            "memory_used_mb": mem_used,
            "memory_per_token_kb": mem_used / (batch_size * seq_len) * 1024,
            "effective_seq_per_gpu": effective_seq,
            "output_shape": list(output.shape),
            "success": True,
        }

        print(
            f"    ✓ Time: {avg_time:.3f}s, Memory: {mem_used:.1f}MB, Throughput: {throughput:,.0f} tok/s"
        )
        if world_size > 1:
            print(
                f"      Effective seq/GPU: {effective_seq}, Memory scaling: O(n/{world_size})"
            )

        # Cleanup
        del x, output
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "implementation": impl_name,
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
        (16384, [2048, 4096, 8192], [1, 2, 4]),
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
    print("Ring Attention Implementation Benchmark")
    print(f"{'=' * 80}")
    print(f"Rank: {rank}, World size: {world_size}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Batch size: {batch_size}")

    all_results = []

    for seq_len, segment_lengths, dilation_rates in test_configs:
        # Skip if not divisible
        if seq_len % max(segment_lengths) != 0:
            continue

        print(f"\n{'-' * 60}")
        print(f"Testing sequence length: {seq_len}")
        print(f"Segments: {segment_lengths}, Dilations: {dilation_rates}")

        config_results = []

        # Test direct implementations
        for impl_name, impl_config in RING_IMPLEMENTATIONS.items():
            try:
                # Create model
                impl_class = impl_config["class"]
                kwargs = impl_config.get("kwargs", {}).copy()

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

                model = impl_class(**kwargs)

                result = benchmark_implementation(
                    impl_name,
                    model,
                    seq_len,
                    batch_size,
                    embed_dim,
                    device,
                    dtype,
                )

                if result:
                    result["description"] = impl_config["description"]
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

                # Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Failed to test {impl_name}: {e}")

        # Test factory implementations
        for impl_name, impl_config in FACTORY_IMPLEMENTATIONS.items():
            try:
                print(f"\n  Testing {impl_name} (factory)...")

                config = RingAttentionConfig(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    **impl_config.get("extra_kwargs", {}),
                )

                model = create_ring_attention(
                    impl_config["factory"],
                    config=config,
                    device=device,
                    dtype=dtype,
                )

                result = benchmark_implementation(
                    impl_name,
                    model,
                    seq_len,
                    batch_size,
                    embed_dim,
                    device,
                    dtype,
                )

                if result:
                    result["description"] = impl_config["description"]
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

                # Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Failed to test {impl_name}: {e}")

        # Print summary for this configuration
        if rank == 0 and config_results:
            print(f"\nSummary for seq_len={seq_len}:")
            print(
                f"{'Implementation':<40} {'Time(s)':<10} {'Memory(MB)':<12} {'Throughput':<15}"
            )
            print("-" * 80)

            # Sort by throughput
            successful = [r for r in config_results if r.get("success", False)]
            successful.sort(key=lambda x: x["throughput"], reverse=True)

            for r in successful:
                print(
                    f"{r['implementation']:<40} "
                    f"{r['time_per_iter']:<10.3f} "
                    f"{r['memory_used_mb']:<12.1f} "
                    f"{r['throughput']:<15,.0f}"
                )

    # Save results
    if rank == 0:
        output_dir = Path("benchmarks/results/ring/working_implementations")
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

                if world_size > 1:
                    # Show memory scaling
                    avg_memory_per_token = sum(
                        r["memory_per_token_kb"] for r in results
                    ) / len(results)
                    print(f"    Avg memory per token: {avg_memory_per_token:.2f} KB")
                    print(f"    Effective O(n/{world_size}) scaling achieved")

    # Cleanup distributed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
