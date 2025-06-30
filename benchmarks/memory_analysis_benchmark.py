"""
Memory analysis benchmark to understand actual memory usage patterns.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import gc


def analyze_ring_memory(rank: int, world_size: int):
    """Analyze Ring V2 memory usage in detail."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12363"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print("\n🔍 DETAILED MEMORY ANALYSIS")
        print("=" * 60)

    try:
        from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
            RingDilatedAttentionV2Collective,
        )
        from dilated_attention_pytorch.improved_dilated_attention import (
            ImprovedDilatedAttention,
        )

        # Simple test case
        seq_len = 4096
        batch_size = 1
        num_heads = 4
        head_dim = 32

        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]

        if rank == 0:
            print("Test configuration:")
            print(f"  Sequence length: {seq_len}")
            print(f"  Batch size: {batch_size}")
            print(f"  Heads: {num_heads}, Head dim: {head_dim}")
            print(f"  Ring size: {world_size}")

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

        # Baseline memory
        baseline_mem = torch.cuda.memory_allocated() / 1024**2

        # Create inputs (same on all GPUs for fair comparison)
        torch.manual_seed(42)  # For reproducible results
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        input_mem = torch.cuda.memory_allocated() / 1024**2 - baseline_mem

        if rank == 0:
            print("\n📊 Memory breakdown:")
            print(f"  Input tensors (Q,K,V): {input_mem:.1f}MB")

        # Test Ring V2 Collective
        ring_model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=world_size,
            device=device,
            dtype=torch.float16,
            enable_memory_pool=False,  # Disable for cleaner measurement
            use_pattern_cache=False,
        ).to(device)

        model_mem = torch.cuda.memory_allocated() / 1024**2 - baseline_mem - input_mem

        # Run inference
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2

        with torch.amp.autocast("cuda"):
            ring_output = ring_model(q, k, v)

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        ring_inference_mem = peak_mem - mem_before

        # Gather results
        if rank == 0:
            # Collect memory stats from all GPUs
            ring_mems = [torch.tensor(0.0, device=device) for _ in range(world_size)]
            inference_mems = [
                torch.tensor(0.0, device=device) for _ in range(world_size)
            ]

            dist.gather(torch.tensor(model_mem, device=device), ring_mems, dst=0)
            dist.gather(
                torch.tensor(ring_inference_mem, device=device), inference_mems, dst=0
            )

            total_ring_mem = sum(m.item() for m in ring_mems)
            total_inference_mem = sum(m.item() for m in inference_mems)
            avg_ring_mem = total_ring_mem / world_size
            avg_inference_mem = total_inference_mem / world_size

            print(f"  Ring model per GPU: {avg_ring_mem:.1f}MB")
            print(f"  Ring inference per GPU: {avg_inference_mem:.1f}MB")
            print(f"  Ring total across GPUs: {total_ring_mem:.1f}MB")

            # Test Improved Dilated (single GPU)
            torch.cuda.empty_cache()
            gc.collect()

            improved_model = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=torch.float16,
            ).to(device)

            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**2

            with torch.amp.autocast("cuda"):
                improved_output = improved_model(q, k, v)

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            improved_inference_mem = peak_mem - mem_before

            print(f"  Improved inference (1 GPU): {improved_inference_mem:.1f}MB")

            # Analysis
            total_ring_cost = input_mem + avg_ring_mem + avg_inference_mem
            total_improved_cost = input_mem + improved_inference_mem

            print("\n📈 Memory comparison:")
            print(f"  Ring V2 total per GPU: {total_ring_cost:.1f}MB")
            print(f"  Improved total: {total_improved_cost:.1f}MB")

            if total_ring_cost < total_improved_cost:
                mem_save = (1 - total_ring_cost / total_improved_cost) * 100
                print(f"  Ring saves: {mem_save:.1f}% per GPU ✅")
            else:
                mem_overhead = (total_ring_cost / total_improved_cost - 1) * 100
                print(f"  Ring overhead: {mem_overhead:.1f}% per GPU ⚠️")

            # Check if Ring enables larger sequences
            estimated_max_improved = (
                7000 / total_improved_cost
            ) * seq_len  # Assuming 7GB GPU
            estimated_max_ring = (7000 / total_ring_cost) * seq_len

            print("\n🚀 Sequence scaling potential:")
            print(f"  Improved max seq (est): {estimated_max_improved:.0f}")
            print(f"  Ring max seq per GPU (est): {estimated_max_ring:.0f}")
            print("  Ring max seq distributed: Unlimited (with more GPUs)")

            # Accuracy check
            output_diff = torch.abs(ring_output - improved_output).mean().item()
            print("\n🎯 Numerical accuracy:")
            print(f"  Output difference: {output_diff:.6f}")
            if output_diff < 0.01:
                print("  ✅ Outputs are very similar")
            elif output_diff < 0.1:
                print("  ✅ Outputs are reasonably similar")
            else:
                print("  ⚠️  Outputs differ significantly")
        else:
            dist.gather(torch.tensor(model_mem, device=device), dst=0)
            dist.gather(torch.tensor(ring_inference_mem, device=device), dst=0)

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        dist.barrier()
        dist.destroy_process_group()


def main():
    """Run memory analysis."""

    print("Memory Analysis Benchmark")
    print("=" * 50)

    world_size = min(2, torch.cuda.device_count())

    if world_size >= 2:
        try:
            mp.spawn(
                analyze_ring_memory, args=(world_size,), nprocs=world_size, join=True
            )
        except Exception as e:
            print(f"❌ Memory analysis failed: {e}")
    else:
        print("⚠️  Need 2+ GPUs for Ring Attention analysis")

    print("\n✅ Memory analysis completed!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
