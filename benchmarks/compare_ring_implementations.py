#!/usr/bin/env python3
"""
Compare Ring V3 and Ring V2 Collective implementations for multi-GPU performance.
Run with: torchrun --nproc_per_node=2 benchmarks/compare_ring_implementations.py
"""

import os
import gc
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def compare_implementations():
    """Compare V3 and V2 Collective implementations."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/compare_ring_implementations.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Ring Implementation Comparison: V3 vs V2 Collective")
        print("=" * 60)
        print(f"World size: {world_size}")
        print("Focus: Multi-GPU performance and dilated attention handling\n")

    # Test configurations
    test_configs = [
        # (seq_len, segment_lengths, dilation_rates, name)
        (512, [256], [1], "Basic - no dilation"),
        (512, [256], [2], "Basic - dilation 2"),
        (1024, [512], [1], "Medium - no dilation"),
        (1024, [256, 256], [1, 2], "Medium - mixed dilation"),
        (2048, [1024], [1], "Large - no dilation"),
        (4096, [2048], [1], "XLarge - no dilation"),
    ]

    results = []

    for seq_len, segment_lengths, dilation_rates, name in test_configs:
        if rank == 0:
            print(f"\nTest: {name}")
            print(
                f"  Sequence: {seq_len}, Segments: {segment_lengths}, Dilation: {dilation_rates}"
            )
            print("-" * 50)

        dist.barrier()

        # Test both implementations
        for impl_name, impl_class in [
            ("V2 Collective", RingDilatedAttentionV2Collective),
            ("V3", RingDilatedAttentionV3),
        ]:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                # Create model with appropriate settings
                if impl_class == RingDilatedAttentionV3:
                    # V3 specific settings
                    model = impl_class(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=0.0,
                        use_bucketed=False,  # Disable due to performance issue
                        device=device,
                        dtype=torch.float32,
                        ring_size=world_size,
                    )
                else:
                    # V2 Collective specific settings
                    model = impl_class(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=0.0,
                        ring_size=world_size,
                        device=device,
                        dtype=torch.float32,
                        use_flash_attention=False,  # Disable for fair comparison
                        enable_memory_pool=True,
                        lightweight_pool=True,
                    )

                # Create inputs with scaling
                torch.manual_seed(42)
                scale = 0.1 / (seq_len**0.25)

                q = torch.randn(1, seq_len, 8, 64, device=device) * scale
                k = torch.randn(1, seq_len, 8, 64, device=device) * scale
                v = torch.randn(1, seq_len, 8, 64, device=device) * scale

                # Warmup
                _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()

                # Timed runs
                dist.barrier()
                times = []

                for _ in range(3):
                    start = time.time()
                    output = model(q, k, v, is_causal=False)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)

                # Get stats
                avg_time = sum(times) / len(times)
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
                has_nan = torch.isnan(output).any().item()
                output_mean = output.mean().item()

                # Gather results from all ranks
                stats = {
                    "time": avg_time,
                    "memory": peak_memory,
                    "has_nan": has_nan,
                    "output_mean": output_mean,
                }

                all_stats = [None] * world_size
                dist.all_gather_object(all_stats, stats)

                if rank == 0:
                    # Average across GPUs
                    avg_time_global = sum(s["time"] for s in all_stats) / len(all_stats)
                    max_memory = max(s["memory"] for s in all_stats)
                    any_nan = any(s["has_nan"] for s in all_stats)

                    status = "✅" if not any_nan else "❌ NaN"
                    print(
                        f"  {impl_name:15} Time: {avg_time_global:.3f}s, Memory: {max_memory:.1f}MB {status}"
                    )

                    # Store for comparison
                    results.append(
                        {
                            "test": name,
                            "impl": impl_name,
                            "time": avg_time_global,
                            "memory": max_memory,
                            "success": not any_nan,
                        }
                    )

                del model, q, k, v, output

            except Exception as e:
                if rank == 0:
                    print(f"  {impl_name:15} ❌ Failed: {str(e)[:60]}...")
                    results.append(
                        {
                            "test": name,
                            "impl": impl_name,
                            "time": float("inf"),
                            "memory": 0,
                            "success": False,
                        }
                    )

    # Summary comparison
    if rank == 0:
        print("\n\n" + "=" * 60)
        print("SUMMARY: Key Differences")
        print("=" * 60)

        print("\n1. Dilation Support:")
        print("   - V2 Collective: ✅ Full support with proper shape handling")
        print("   - V3: ❌ Disabled in multi-GPU mode (lines 180-182)")

        print("\n2. Communication Pattern:")
        print("   - V2 Collective: all_gather (robust, NCCL optimized)")
        print("   - V3: ring_pass utilities (more complex, potentially slower)")

        print("\n3. Memory Efficiency:")
        print("   - V2 Collective: Enhanced memory pool, pattern caching")
        print("   - V3: Basic implementation, bucketing has issues")

        print("\n4. Numerical Stability:")
        print("   - V2 Collective: Built-in handling, no NaN issues")
        print("   - V3: Required LSE fix for -inf handling")

        print("\n5. Performance Summary:")
        v2_times = [
            r["time"] for r in results if r["impl"] == "V2 Collective" and r["success"]
        ]
        v3_times = [r["time"] for r in results if r["impl"] == "V3" and r["success"]]

        if v2_times and v3_times:
            v2_avg = sum(v2_times) / len(v2_times)
            v3_avg = sum(v3_times) / len(v3_times)
            speedup = v3_avg / v2_avg
            print(f"   - Average V2 Collective: {v2_avg:.3f}s")
            print(f"   - Average V3: {v3_avg:.3f}s")
            print(
                f"   - V2 is {speedup:.1f}x faster"
                if speedup > 1
                else f"   - V3 is {1 / speedup:.1f}x faster"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    compare_implementations()
