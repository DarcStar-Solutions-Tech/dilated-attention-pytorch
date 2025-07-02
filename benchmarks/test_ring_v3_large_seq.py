#!/usr/bin/env python3
"""
Test Ring V3 with very large sequences using bucketing and gradient checkpointing.
"""

import os
import gc
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import (
    RingDilatedAttentionV3,
    RingMultiheadDilatedAttentionV3,
)


def test_large_sequence():
    """Test with large sequences using memory optimizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("Skipping test (no CUDA)")
        return

    print("Testing Large Sequence Processing")
    print("=" * 50)

    # Test configurations - start small and increase
    test_configs = [
        # (seq_len, bucket_size, grad_checkpoint, dtype_str)
        (4096, 512, False, "float16"),
        (8192, 512, False, "float16"),
        (8192, 256, True, "float16"),  # With gradient checkpointing
        (16384, 256, True, "float16"),  # Very large
    ]

    for seq_len, bucket_size, grad_checkpoint, dtype_str in test_configs:
        print(
            f"\nTesting seq_len={seq_len:,}, bucket_size={bucket_size}, "
            f"grad_checkpoint={grad_checkpoint}, dtype={dtype_str}"
        )

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dtype = torch.float16 if dtype_str == "float16" else torch.float32

        try:
            # Create model
            model = RingDilatedAttentionV3(
                segment_lengths=[2048],
                dilation_rates=[1],
                bucket_size=bucket_size,
                use_bucketed=True,
                grad_checkpoint_buckets=grad_checkpoint,
                device=device,
                dtype=dtype,
            )

            # Set training mode if using gradient checkpointing
            if grad_checkpoint:
                model.train()

            # Create inputs (minimal size for memory)
            batch_size = 1
            num_heads = 8
            head_dim = 64

            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )

            # Get memory before forward
            mem_before = torch.cuda.memory_allocated() / (1024**2)

            # Forward pass
            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Get memory stats
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            mem_after = torch.cuda.memory_allocated() / (1024**2)

            print("  ✅ Success!")
            print(f"     Memory before: {mem_before:.1f} MB")
            print(f"     Memory after: {mem_after:.1f} MB")
            print(f"     Peak memory: {peak_memory:.1f} MB")
            print(f"     Output shape: {output.shape}")
            print(f"     Output mean: {output.float().mean().item():.6f}")

            # Clean up
            del q, k, v, output, model

        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"  ❌ OOM - Peak memory before failure: {peak_memory:.1f} MB")
            else:
                print(f"  ❌ Error: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")


def test_multihead_large():
    """Test multihead wrapper with large sequences."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("\nSkipping multihead test (no CUDA)")
        return

    print("\n\nTesting Multihead with Large Sequences")
    print("=" * 50)

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Create multihead model
        model = RingMultiheadDilatedAttentionV3(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            bucket_size=256,
            use_bucketed=True,
            grad_checkpoint_buckets=True,
            device=device,
            dtype=torch.float16,
        )

        model.train()  # For gradient checkpointing

        # Test input
        seq_len = 4096
        batch_size = 1
        x = torch.randn(batch_size, seq_len, 512, device=device, dtype=torch.float16)

        print(f"Input shape: {x.shape}")
        print(f"Input memory: {x.numel() * 2 / (1024**2):.1f} MB")

        # Forward pass
        output = model(x, x, x, is_causal=False)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

        print("✅ Success!")
        print(f"   Output shape: {output.shape}")
        print(f"   Peak memory: {peak_memory:.1f} MB")
        print(f"   Output mean: {output.float().mean().item():.6f}")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_multi_gpu_bucketed():
    """Test bucketed processing with multiple GPUs."""
    if "RANK" not in os.environ:
        print("\n\nSkipping multi-GPU test (not in distributed mode)")
        print("Run with: torchrun --nproc_per_node=2 test_ring_v3_large_seq.py")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("\n\nTesting Multi-GPU with Bucketed Processing")
        print("=" * 50)
        print(f"World size: {world_size}")

    # Create model
    model = RingDilatedAttentionV3(
        segment_lengths=[1024],
        dilation_rates=[1],
        bucket_size=256,
        use_bucketed=True,
        grad_checkpoint_buckets=False,
        device=device,
        dtype=torch.float16,
        ring_size=world_size,
    )

    # Test with medium sequence
    seq_len = 2048  # Must be divisible by world_size
    q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16)

    try:
        output = model(q, k, v, is_causal=False)

        if rank == 0:
            print("✅ Multi-GPU forward pass succeeded!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output mean: {output.float().mean().item():.6f}")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_large_sequence()
    test_multihead_large()
    test_multi_gpu_bucketed()
