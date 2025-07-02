#!/usr/bin/env python3
"""
Verify if V2 Collective achieves true ring attention memory scaling.
Focus on memory patterns rather than performance.
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def check_memory_pattern():
    """Check if memory scales as O(n/p) for ring attention."""

    # Get rank and world size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        print("This test requires multiple GPUs. Run with:")
        print("torchrun --nproc_per_node=2 verify_ring_attention_memory.py")
        return

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Test configuration
    seq_len = 16384
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    if rank == 0:
        print("=" * 60)
        print("Ring Attention Memory Pattern Verification")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Sequence length: {seq_len:,}")
        print(f"Configuration: batch={batch_size}, heads={num_heads}, dim={head_dim}")

    # Synchronize before starting
    dist.barrier()

    # Clear memory and get baseline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    baseline_mem = torch.cuda.memory_allocated() / (1024**2)

    # Create model
    model = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        dtype=torch.float16,
        ring_size=world_size,
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    mem_after_alloc = torch.cuda.memory_allocated() / (1024**2)
    input_memory = mem_after_alloc - baseline_mem

    # Run forward pass
    output = model(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    _ = torch.cuda.memory_allocated() / (1024**2)

    # Calculate memory usage
    computation_memory = peak_memory - mem_after_alloc

    # Gather memory info from all ranks
    all_peak_memories = [None] * world_size
    all_input_memories = [None] * world_size
    all_computation_memories = [None] * world_size

    dist.all_gather_object(all_peak_memories, peak_memory)
    dist.all_gather_object(all_input_memories, input_memory)
    dist.all_gather_object(all_computation_memories, computation_memory)

    if rank == 0:
        print("\nMemory Usage by GPU:")
        print("-" * 60)
        print(f"{'GPU':<5} {'Input':<10} {'Compute':<10} {'Peak':<10} {'Total':<10}")
        print("-" * 60)

        total_peak = 0
        total_input = 0
        total_compute = 0

        for i in range(world_size):
            print(
                f"{i:<5} {all_input_memories[i]:<10.1f} "
                f"{all_computation_memories[i]:<10.1f} "
                f"{all_peak_memories[i]:<10.1f} "
                f"{all_peak_memories[i]:<10.1f}"
            )
            total_peak += all_peak_memories[i]
            total_input += all_input_memories[i]
            total_compute += all_computation_memories[i]

        print("-" * 60)
        print(
            f"{'Total':<5} {total_input:<10.1f} {total_compute:<10.1f} "
            f"{total_peak:<10.1f}"
        )

        avg_peak = total_peak / world_size

        # Theoretical memory for true ring attention
        # Each GPU should hold:
        # - Full Q: batch * seq * heads * dim * 2 bytes
        # - 1/p of K and V: 2 * (batch * seq/p * heads * dim * 2 bytes)
        bytes_per_element = 2  # float16

        q_memory_mb = (
            batch_size * seq_len * num_heads * head_dim * bytes_per_element
        ) / (1024**2)
        kv_memory_mb = (
            2
            * (
                batch_size
                * (seq_len / world_size)
                * num_heads
                * head_dim
                * bytes_per_element
            )
            / (1024**2)
        )
        theoretical_per_gpu = q_memory_mb + kv_memory_mb

        print("\nTheoretical memory for true ring attention:")
        print(f"  Q (full): {q_memory_mb:.1f} MB")
        print(f"  K,V (1/{world_size}): {kv_memory_mb:.1f} MB")
        print(f"  Total per GPU: {theoretical_per_gpu:.1f} MB")

        print(f"\nActual average per GPU: {avg_peak:.1f} MB")

        # Check if we have true ring attention
        # In all-gather pattern, each GPU would have full K,V
        all_gather_kv_memory = (
            2
            * (batch_size * seq_len * num_heads * head_dim * bytes_per_element)
            / (1024**2)
        )
        all_gather_total = q_memory_mb + all_gather_kv_memory

        print(f"\nAll-gather pattern would use: {all_gather_total:.1f} MB per GPU")

        # Determine pattern
        ratio = avg_peak / all_gather_total

        print(f"\nMemory ratio (actual/all-gather): {ratio:.2f}")

        if ratio < 0.7:
            print("✅ TRUE RING ATTENTION ACHIEVED!")
            print(f"   Memory scaling: O(n/{world_size})")
        elif ratio < 0.9:
            print("⚠️  PARTIAL RING ATTENTION")
            print("   Some memory savings but not optimal")
        else:
            print("❌ ALL-GATHER PATTERN DETECTED")
            print("   Memory scaling: O(n)")

        # Test larger sequence
        print("\n" + "=" * 60)
        print("Testing maximum sequence length...")

    # Clean up
    del q, k, v, output, model
    torch.cuda.empty_cache()

    # Test maximum sequence length
    if rank == 0:
        dist.barrier()

    max_seq_len = 65536
    while max_seq_len <= 1048576:  # Up to 1M tokens
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            # Adjust segment lengths
            if max_seq_len <= 32768:
                test_segments = [4096, 8192]
            elif max_seq_len <= 131072:
                test_segments = [8192, 16384]
            else:
                test_segments = [16384, 32768]

            model = RingDilatedAttentionV2Collective(
                segment_lengths=test_segments,
                dilation_rates=[1, 2],
                device=device,
                dtype=torch.float16,
                ring_size=world_size,
            )

            q = torch.randn(1, max_seq_len, 8, 64, device=device, dtype=torch.float16)
            k = torch.randn(1, max_seq_len, 8, 64, device=device, dtype=torch.float16)
            v = torch.randn(1, max_seq_len, 8, 64, device=device, dtype=torch.float16)

            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            # Gather success from all ranks
            all_success = [None] * world_size
            dist.all_gather_object(all_success, True)

            if rank == 0 and all(all_success):
                print(f"✓ {max_seq_len:,} tokens: {peak_mb:.0f} MB per GPU")

            del q, k, v, output, model
            torch.cuda.empty_cache()

            max_seq_len *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Gather failure from all ranks
                all_success = [None] * world_size
                dist.all_gather_object(all_success, False)

                if rank == 0:
                    print(f"✗ {max_seq_len:,} tokens: OOM")
                break
            else:
                raise

    if rank == 0:
        print("\n" + "=" * 60)
        print("CONCLUSION:")
        print("=" * 60)

        if world_size == 2:
            print("With 2 GPUs:")
            print("- Single GPU max: ~65K tokens")
            print(f"- Two GPU max: {max_seq_len // 2:,} tokens")
            print(f"- Scaling factor: {(max_seq_len // 2) / 65536:.1f}x")


def main():
    """Main function."""
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        check_memory_pattern()
        dist.destroy_process_group()
    else:
        print("This script must be run with torchrun:")
        print("torchrun --nproc_per_node=2 verify_ring_attention_memory.py")


if __name__ == "__main__":
    main()
