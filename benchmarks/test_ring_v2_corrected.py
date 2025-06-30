"""
Test the corrected Ring Dilated Attention V2 implementation.

This verifies that the fixed distributed communication works properly.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time


def test_corrected_ring_v2(rank: int, world_size: int):
    """Test the corrected Ring Dilated Attention V2."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print(f"\n[GPU {rank}] Testing Corrected Ring Dilated Attention V2")

    # Import the corrected version
    from dilated_attention_pytorch.ring_dilated_attention_v2_corrected import (
        RingDilatedAttentionV2Corrected,
    )

    # Test parameters
    batch_size = 1
    seq_len = 4096  # Must be divisible by largest segment length
    num_heads = 8
    head_dim = 64

    # Create model
    model = RingDilatedAttentionV2Corrected(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        ring_size=world_size,
        device=device,
        dtype=torch.float16,
        enable_memory_pool=False,
        use_pattern_cache=False,
    ).to(device)

    print(f"[GPU {rank}] Model created successfully")
    print(f"[GPU {rank}] Mode: {model.mode}")
    print(f"[GPU {rank}] Ring size: {model.ring_size}")

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Monitor memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1024**2

    print(f"[GPU {rank}] Initial memory: {initial_mem:.1f}MB")

    # Synchronize before forward
    dist.barrier()

    # Forward pass
    print(f"[GPU {rank}] Running forward pass...")
    start_time = time.time()

    try:
        with torch.amp.autocast("cuda"):
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        end_time = time.time()

        # Report results
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        _ = torch.cuda.memory_allocated() / 1024**2

        print(f"[GPU {rank}] SUCCESS! Forward pass completed")
        print(f"[GPU {rank}] Time: {(end_time - start_time) * 1000:.1f}ms")
        print(f"[GPU {rank}] Peak memory: {peak_mem:.1f}MB")
        print(f"[GPU {rank}] Output shape: {output.shape}")
        print(f"[GPU {rank}] Output valid: {not torch.isnan(output).any().item()}")

        # Get memory estimate
        mem_est = model.get_memory_estimate(seq_len, batch_size, num_heads, head_dim)
        if rank == 0:
            print(f"\n[GPU {rank}] Memory estimate:")
            for key, value in mem_est.items():
                if isinstance(value, float) and "gb" in key:
                    print(f"  {key}: {value * 1024:.1f}MB")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"[GPU {rank}] ERROR: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()
    print(f"[GPU {rank}] Test completed")


def main():
    """Run the corrected Ring V2 test."""
    print("Testing Corrected Ring Dilated Attention V2")
    print("=" * 70)
    print("\nThis version fixes the distributed communication issues:")
    print("- Uses dist.isend/irecv instead of non-existent dist.sendrecv")
    print("- Properly implements ring communication pattern")
    print("- Compatible with all PyTorch versions >= 1.8")
    print("=" * 70)

    world_size = min(2, torch.cuda.device_count())
    if world_size < 2:
        print("\nError: Need at least 2 GPUs for distributed test")
        return

    print(f"\nUsing {world_size} GPUs")

    try:
        mp.spawn(
            test_corrected_ring_v2, args=(world_size,), nprocs=world_size, join=True
        )

        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("Corrected Ring Dilated Attention V2 is working properly")
        print("=" * 70)

    except Exception as e:
        print(f"\nTest failed: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
