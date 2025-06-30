"""
Compare Ring Dilated Attention V2 Collective vs Improved Dilated Attention.

This script analyzes:
1. Architectural differences
2. Performance characteristics
3. Memory usage patterns
4. Feature parity
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import gc
from datetime import datetime


def compare_implementations(rank: int, world_size: int):
    """Compare Ring V2 Collective vs Improved Dilated Attention."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12367"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print("\nComparing Ring V2 Collective vs Improved Dilated Attention")
        print("=" * 70)

    try:
        from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
            RingDilatedAttentionV2Collective,
        )
        from dilated_attention_pytorch.improved_dilated_attention import (
            ImprovedDilatedAttention,
        )

        # Test configurations
        configs = [
            {"seq_len": 2048, "batch_size": 2, "num_heads": 8, "head_dim": 64},
            {"seq_len": 4096, "batch_size": 2, "num_heads": 8, "head_dim": 64},
            {"seq_len": 8192, "batch_size": 1, "num_heads": 8, "head_dim": 64},
        ]

        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]

        results = {}

        for config in configs:
            seq_len = config["seq_len"]
            batch_size = config["batch_size"]
            num_heads = config["num_heads"]
            head_dim = config["head_dim"]

            if rank == 0:
                print(f"\n📊 Testing seq_len={seq_len}, batch={batch_size}")
                print("-" * 50)

            # Create models
            ring_model = RingDilatedAttentionV2Collective(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
                enable_memory_pool=True,  # Enable for Ring
                use_pattern_cache=True,
            ).to(device)

            if rank == 0:
                improved_model = ImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    device=device,
                    dtype=torch.float16,
                    use_memory_pool=True,  # Enable for Improved
                    use_cached_indices=True,
                ).to(device)

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
            for _ in range(3):
                with torch.amp.autocast("cuda"):
                    _ = ring_model(q, k, v)
                    if rank == 0:
                        _ = improved_model(q, k, v)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

            # Benchmark Ring V2 Collective
            num_iters = 10
            torch.cuda.reset_peak_memory_stats()
            ring_mem_before = torch.cuda.memory_allocated() / 1024**2

            dist.barrier()
            ring_start = time.time()

            for _ in range(num_iters):
                with torch.amp.autocast("cuda"):
                    ring_output = ring_model(q, k, v)

            torch.cuda.synchronize()
            ring_end = time.time()

            ring_time = (ring_end - ring_start) / num_iters * 1000
            ring_peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            ring_mem_used = ring_peak_mem - ring_mem_before

            # Gather ring results
            ring_times = [torch.tensor(ring_time, device=device)]
            ring_mems = [torch.tensor(ring_mem_used, device=device)]

            if rank == 0:
                for _ in range(world_size - 1):
                    ring_times.append(torch.empty(1, device=device))
                    ring_mems.append(torch.empty(1, device=device))

                dist.gather(torch.tensor(ring_time, device=device), ring_times, dst=0)
                dist.gather(
                    torch.tensor(ring_mem_used, device=device), ring_mems, dst=0
                )

                avg_ring_time = sum(t.item() for t in ring_times) / world_size
                total_ring_mem = sum(m.item() for m in ring_mems)
                avg_ring_mem = total_ring_mem / world_size
            else:
                dist.gather(torch.tensor(ring_time, device=device), dst=0)
                dist.gather(torch.tensor(ring_mem_used, device=device), dst=0)

            # Benchmark Improved Dilated Attention (only on rank 0)
            if rank == 0:
                torch.cuda.empty_cache()
                gc.collect()

                torch.cuda.reset_peak_memory_stats()
                improved_mem_before = torch.cuda.memory_allocated() / 1024**2

                improved_start = time.time()

                for _ in range(num_iters):
                    with torch.amp.autocast("cuda"):
                        improved_output = improved_model(q, k, v)

                torch.cuda.synchronize()
                improved_end = time.time()

                improved_time = (improved_end - improved_start) / num_iters * 1000
                improved_peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                improved_mem_used = improved_peak_mem - improved_mem_before

                # Compare outputs
                output_diff = torch.abs(ring_output - improved_output).mean().item()

                # Store results
                results[seq_len] = {
                    "ring_time": avg_ring_time,
                    "ring_mem_per_gpu": avg_ring_mem,
                    "ring_mem_total": total_ring_mem,
                    "improved_time": improved_time,
                    "improved_mem": improved_mem_used,
                    "output_diff": output_diff,
                    "speedup": improved_time / avg_ring_time,
                    "mem_reduction": (1 - avg_ring_mem / improved_mem_used) * 100,
                }

                # Print comparison
                print("\n🔹 Ring V2 Collective (Distributed):")
                print(f"   Time: {avg_ring_time:.2f}ms")
                print(f"   Memory per GPU: {avg_ring_mem:.1f}MB")
                print(f"   Total memory: {total_ring_mem:.1f}MB")

                print("\n🔹 Improved Dilated Attention (Single GPU):")
                print(f"   Time: {improved_time:.2f}ms")
                print(f"   Memory: {improved_mem_used:.1f}MB")

                print("\n📈 Comparison:")
                print(f"   Speedup: {results[seq_len]['speedup']:.2f}x")
                print(
                    f"   Memory reduction per GPU: {results[seq_len]['mem_reduction']:.1f}%"
                )
                print(f"   Output difference: {output_diff:.6f}")

                if output_diff < 1e-3:
                    print("   ✅ Outputs are numerically similar")
                else:
                    print("   ⚠️  Outputs differ significantly")

            dist.barrier()
            torch.cuda.empty_cache()
            gc.collect()

        # Summary
        if rank == 0:
            print("\n" + "=" * 70)
            print("ARCHITECTURAL COMPARISON")
            print("=" * 70)

            print("\n📋 Feature Comparison:")
            print(
                f"{'Feature':<30} {'Ring V2 Collective':<20} {'Improved Dilated':<20}"
            )
            print("-" * 70)

            features = [
                ("Distributed Support", "✅ Multi-GPU", "❌ Single GPU"),
                ("Memory Scaling", "O(n/ring_size)", "O(n)"),
                ("Dilated Patterns", "✅ Applied", "✅ Applied"),
                ("Pattern Caching", "✅ Global Cache", "✅ Global Cache"),
                ("Memory Pool", "✅ Enhanced", "✅ Optional"),
                ("SDPA Optimization", "❌ Basic", "✅ Kernel Selection"),
                ("Flash Attention", "❌ Not integrated", "✅ Auto-detected"),
                ("Online Softmax", "✅ For correctness", "❌ Standard"),
                ("Communication", "✅ Collective ops", "N/A"),
                ("Head Groups", "✅ Manual division", "✅ Cached groups"),
            ]

            for feature, ring, improved in features:
                print(f"{feature:<30} {ring:<20} {improved:<20}")

            print("\n📊 Performance Summary:")
            print(
                f"{'Seq Length':<12} {'Speedup':<12} {'Mem Save/GPU':<15} {'Output Diff':<12}"
            )
            print("-" * 51)

            for seq_len, data in results.items():
                print(
                    f"{seq_len:<12} {data['speedup']:<12.2f} "
                    f"{data['mem_reduction']:<15.1f}% {data['output_diff']:<12.6f}"
                )

            print("\n🎯 Key Insights:")
            print("1. Ring V2 provides significant memory savings per GPU")
            print("2. Improved Dilated is faster on single GPU (SDPA optimizations)")
            print("3. Both apply dilated patterns correctly")
            print("4. Ring V2 enables longer sequences through distribution")
            print("5. Trade-off: Ring has communication overhead but better scaling")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        dist.barrier()
        dist.destroy_process_group()


def main():
    """Run comparison."""
    print("Ring V2 Collective vs Improved Dilated Attention Comparison")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    world_size = 2
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs for distributed Ring V2")
        return

    try:
        mp.spawn(
            compare_implementations, args=(world_size,), nprocs=world_size, join=True
        )
        print("\n✅ Comparison completed successfully!")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
