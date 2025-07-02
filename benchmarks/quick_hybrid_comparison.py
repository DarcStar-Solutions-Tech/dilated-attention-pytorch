#!/usr/bin/env python3
"""
Quick comparison of Hybrid vs V2 Collective on 2 GPUs.
Run with: torchrun --nproc_per_node=2 benchmarks/quick_hybrid_comparison.py
"""

import os
import gc
import time
import torch
import torch.distributed as dist
from datetime import datetime

from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def quick_benchmark(model, model_name, seq_len, device, num_iterations=5):
    """Quick benchmark with fewer iterations."""

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        _ = model(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Time iterations
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    # Memory stats
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Check output
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()

    return {
        "model": model_name,
        "seq_len": seq_len,
        "avg_time_ms": sum(times) / len(times) * 1000,
        "peak_mem_mb": peak_mem,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def main():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/quick_hybrid_comparison.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Quick Hybrid vs V2 Collective Comparison")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Test configurations (smaller for quick test)
    test_configs = [
        (1024, 512, 1),
        (2048, 1024, 1),
        (2048, 512, 2),
    ]

    for seq_len, segment_len, dilation_rate in test_configs:
        if rank == 0:
            print(
                f"\nSequence: {seq_len}, Segment: {segment_len}, Dilation: {dilation_rate}"
            )
            print("-" * 60)

        dist.barrier()

        # Test V2 Collective
        try:
            v2_model = RingDilatedAttentionV2Collective(
                segment_lengths=[segment_len],
                dilation_rates=[dilation_rate],
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
            )

            v2_result = quick_benchmark(v2_model, "V2 Collective", seq_len, device)

            # Gather results
            all_v2_results = [None] * world_size
            dist.all_gather_object(all_v2_results, v2_result)

            if rank == 0:
                avg_time = sum(r["avg_time_ms"] for r in all_v2_results) / len(
                    all_v2_results
                )
                max_mem = max(r["peak_mem_mb"] for r in all_v2_results)
                any_nan = any(r["has_nan"] for r in all_v2_results)

                print(f"V2 Collective: {avg_time:.2f}ms, {max_mem:.1f}MB peak", end="")
                if any_nan:
                    print(" ⚠️ NaN detected!", end="")
                print()

            del v2_model

        except Exception as e:
            if rank == 0:
                print(f"V2 Collective: Failed - {e}")

        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

        # Test Hybrid
        try:
            hybrid_model = RingDilatedAttentionHybrid(
                segment_lengths=[segment_len],
                dilation_rates=[dilation_rate],
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
                enable_memory_pool=True,
                use_pattern_cache=True,
                use_flash_attention=False,  # Disable for fair comparison
            )

            hybrid_result = quick_benchmark(hybrid_model, "Hybrid", seq_len, device)

            # Gather results
            all_hybrid_results = [None] * world_size
            dist.all_gather_object(all_hybrid_results, hybrid_result)

            if rank == 0:
                avg_time = sum(r["avg_time_ms"] for r in all_hybrid_results) / len(
                    all_hybrid_results
                )
                max_mem = max(r["peak_mem_mb"] for r in all_hybrid_results)
                any_nan = any(r["has_nan"] for r in all_hybrid_results)

                print(f"Hybrid:        {avg_time:.2f}ms, {max_mem:.1f}MB peak", end="")
                if any_nan:
                    print(" ⚠️ NaN detected!", end="")

                # Compare with V2
                if "all_v2_results" in locals():
                    v2_avg_time = sum(r["avg_time_ms"] for r in all_v2_results) / len(
                        all_v2_results
                    )
                    v2_max_mem = max(r["peak_mem_mb"] for r in all_v2_results)

                    time_ratio = avg_time / v2_avg_time
                    mem_ratio = max_mem / v2_max_mem

                    print(f" ({time_ratio:.2f}x time, {mem_ratio:.2f}x memory)")
                else:
                    print()

            del hybrid_model

        except Exception as e:
            if rank == 0:
                print(f"Hybrid: Failed - {e}")

        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Summary:")
        print("- V2 Collective: Uses all_gather (O(n) memory)")
        print("- Hybrid: True ring attention (O(n/p) memory) + V2 features")
        print("- Lower memory ratio = better (Hybrid should use less memory)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
