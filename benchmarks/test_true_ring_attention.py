#!/usr/bin/env python3
"""
Test True Ring Dilated Attention to verify O(n/p) memory scaling.
Run with: torchrun --nproc_per_node=<num_gpus> test_true_ring_attention.py
"""

import os
import gc
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_true import (
    TrueRingDilatedAttention,
    TrueRingMultiheadDilatedAttention,
)


def test_memory_scaling():
    """Test that memory scales as O(n/p) with true ring attention."""

    # Get rank and world size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        print("This test is designed for multi-GPU. Single GPU test:")
        world_size = 1

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
        print("True Ring Dilated Attention Test")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Sequence length: {seq_len:,}")
        print(f"Configuration: batch={batch_size}, heads={num_heads}, dim={head_dim}")

    # Synchronize
    if world_size > 1:
        dist.barrier()

    # Clear memory and get baseline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    baseline_mem = torch.cuda.memory_allocated() / (1024**2)

    try:
        # Create model
        model = TrueRingDilatedAttention(
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
        computation_memory = peak_memory - mem_after_alloc

        if world_size > 1:
            # Gather memory info from all ranks
            all_peak_memories = [None] * world_size
            dist.all_gather_object(all_peak_memories, peak_memory)

            if rank == 0:
                print("\nMemory Usage by GPU:")
                print("-" * 60)
                total_peak = sum(all_peak_memories)
                avg_peak = total_peak / world_size

                for i, mem in enumerate(all_peak_memories):
                    print(f"GPU {i}: {mem:.1f} MB")

                print(f"\nAverage per GPU: {avg_peak:.1f} MB")

                # Compare with theoretical
                bytes_per_element = 2  # float16

                # True ring: Each GPU holds full Q + 1/p of K,V
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
                theoretical_ring = q_memory_mb + kv_memory_mb

                # All-gather: Each GPU holds full Q,K,V
                theoretical_allgather = 3 * q_memory_mb

                print("\nTheoretical memory:")
                print(
                    f"  True ring (O(n/{world_size})): {theoretical_ring:.1f} MB per GPU"
                )
                print(f"  All-gather (O(n)): {theoretical_allgather:.1f} MB per GPU")
                print(f"\nActual: {avg_peak:.1f} MB per GPU")

                ratio = avg_peak / theoretical_allgather
                print(f"\nMemory ratio (actual/all-gather): {ratio:.2f}")

                if ratio < 0.7:
                    print("✅ TRUE RING ATTENTION ACHIEVED!")
                else:
                    print("⚠️  Memory usage higher than expected")
        else:
            print("\nSingle GPU test:")
            print(f"Peak memory: {peak_memory:.1f} MB")
            print(f"Input memory: {input_memory:.1f} MB")
            print(f"Computation memory: {computation_memory:.1f} MB")

        success = True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        success = False

    # Clean up
    if "model" in locals():
        del model
    if "q" in locals():
        del q, k, v
    if "output" in locals():
        del output
    torch.cuda.empty_cache()
    gc.collect()

    return success


def test_multihead_wrapper():
    """Test the multihead wrapper."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Multihead Wrapper")
        print("=" * 60)

    try:
        # Create multihead model
        model = TrueRingMultiheadDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=device,
            dtype=torch.float16,
            ring_size=world_size,
        )

        # Test input
        batch_size = 2
        seq_len = 4096
        x = torch.randn(batch_size, seq_len, 512, device=device, dtype=torch.float16)

        # Forward pass
        output, _ = model(x, x, x, is_causal=False)

        assert output.shape == x.shape

        if rank == 0:
            print("✅ Multihead wrapper test passed")
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {output.shape}")

        return True

    except Exception as e:
        if rank == 0:
            print(f"❌ Multihead wrapper test failed: {e}")
        return False


def test_scaling_limits():
    """Test maximum sequence lengths with true ring attention."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Scaling Limits")
        print("=" * 60)

    # Test increasing sequence lengths
    test_lengths = [8192, 16384, 32768, 65536, 131072, 262144]

    max_working = 0

    for seq_len in test_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Adjust segment lengths
            if seq_len <= 16384:
                segment_lengths = [2048, 4096]
            elif seq_len <= 65536:
                segment_lengths = [4096, 8192]
            else:
                segment_lengths = [8192, 16384]

            model = TrueRingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=[1, 2],
                device=device,
                dtype=torch.float16,
                ring_size=world_size,
            )

            # Small batch for testing limits
            q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)
            k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)
            v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)

            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            if world_size > 1:
                all_success = [None] * world_size
                dist.all_gather_object(all_success, True)

                if rank == 0 and all(all_success):
                    print(f"✓ {seq_len:,} tokens: {peak_mb:.0f} MB per GPU")
                    max_working = seq_len
            else:
                print(f"✓ {seq_len:,} tokens: {peak_mb:.0f} MB")
                max_working = seq_len

            del q, k, v, output, model

        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"✗ {seq_len:,} tokens: OOM")
                break
            else:
                raise

    if rank == 0:
        print(
            f"\nMaximum sequence length with {world_size} GPU(s): {max_working:,} tokens"
        )

        if world_size > 1:
            _ = 65536 * world_size  # Based on single GPU limit
            actual_scaling = max_working / 65536
            print(f"Expected scaling: {world_size}x")
            print(f"Actual scaling: {actual_scaling:.1f}x")


def main():
    """Main test function."""
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")

        # Run tests
        test_memory_scaling()
        test_multihead_wrapper()
        test_scaling_limits()

        dist.destroy_process_group()
    else:
        print("Running single GPU tests...")
        test_memory_scaling()
        test_multihead_wrapper()
        test_scaling_limits()


if __name__ == "__main__":
    main()
