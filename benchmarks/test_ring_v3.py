#!/usr/bin/env python3
"""
Test Ring Dilated Attention V3 with proper utilities.
Run with: torchrun --nproc_per_node=<num_gpus> test_ring_v3.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import (
    RingDilatedAttentionV3,
    RingMultiheadDilatedAttentionV3,
)


def test_ring_v3():
    """Test Ring V3 implementation."""

    # Initialize distributed if available
    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Test configuration
    seq_len = 8192  # Start smaller
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048]
    dilation_rates = [1, 2]

    if rank == 0:
        print("=" * 60)
        print("Ring Dilated Attention V3 Test")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Sequence length: {seq_len:,}")

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # Create model
        model = RingDilatedAttentionV3(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ring_size=world_size,
        )

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )

        # Get memory before forward
        if device.type == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**2)

        # Forward pass
        output = model(q, k, v, is_causal=False)

        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()

        # Get memory after forward
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            mem_after = torch.cuda.memory_allocated() / (1024**2)

            print(f"\n[Rank {rank}] Memory usage:")
            print(f"  Before forward: {mem_before:.1f} MB")
            print(f"  After forward: {mem_after:.1f} MB")
            print(f"  Peak: {peak_memory:.1f} MB")

        # Verify output shape
        assert output.shape == q.shape, (
            f"Output shape mismatch: {output.shape} vs {q.shape}"
        )

        if rank == 0:
            print("\n✅ Basic test passed!")
            print(f"   Output shape: {output.shape}")

        # Test causal
        output_causal = model(q, k, v, is_causal=True)
        assert output_causal.shape == q.shape

        if rank == 0:
            print("✅ Causal test passed!")

        # Clean up
        del q, k, v, output, output_causal, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        success = True

    except Exception as e:
        print(f"\n[Rank {rank}] ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        success = False

    return success


def test_multihead_v3():
    """Test multihead wrapper."""
    if "RANK" in os.environ:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Multihead V3")
        print("=" * 60)

    try:
        model = RingMultiheadDilatedAttentionV3(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ring_size=world_size,
        )

        # Test input
        batch_size = 2
        seq_len = 2048
        x = torch.randn(batch_size, seq_len, 512, device=device, dtype=model.dtype)

        # Forward pass
        output = model(x, x, x, is_causal=False)

        assert output.shape == x.shape

        if rank == 0:
            print("✅ Multihead test passed!")
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {output.shape}")

        return True

    except Exception as e:
        if rank == 0:
            print(f"❌ Multihead test failed: {e}")
        return False


def test_memory_scaling():
    """Test memory scaling with different sequence lengths."""
    if "RANK" in os.environ:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Memory Scaling")
        print("=" * 60)

    test_lengths = [2048, 4096, 8192, 16384]

    for seq_len in test_lengths:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            # Adjust segment lengths
            if seq_len <= 4096:
                segment_lengths = [512, 1024]
            else:
                segment_lengths = [1024, 2048]

            model = RingDilatedAttentionV3(
                segment_lengths=segment_lengths,
                dilation_rates=[1, 2],
                device=device,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                ring_size=world_size,
            )

            # Small batch for memory testing
            q = torch.randn(1, seq_len, 8, 64, device=device, dtype=model.dtype)
            k = torch.randn(1, seq_len, 8, 64, device=device, dtype=model.dtype)
            v = torch.randn(1, seq_len, 8, 64, device=device, dtype=model.dtype)

            output = model(q, k, v, is_causal=False)

            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

                if world_size > 1:
                    # Gather memory from all ranks
                    all_memories = [None] * world_size
                    dist.all_gather_object(all_memories, peak_mb)

                    if rank == 0:
                        avg_memory = sum(all_memories) / world_size
                        print(f"✓ {seq_len:,} tokens: {avg_memory:.0f} MB avg per GPU")
                        print(
                            f"  Individual GPUs: {[f'{m:.0f}' for m in all_memories]}"
                        )
                else:
                    print(f"✓ {seq_len:,} tokens: {peak_mb:.0f} MB")

            del q, k, v, output, model

        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"✗ {seq_len:,} tokens: OOM")
                break
            else:
                raise


def main():
    """Main test function."""
    # Run tests
    test_ring_v3()
    test_multihead_v3()
    test_memory_scaling()

    # Clean up distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
