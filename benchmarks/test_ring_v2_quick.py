"""
Quick test of corrected Ring V2 implementation.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def quick_test(rank: int, world_size: int):
    """Quick test of Ring V2 collective."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12370"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
            RingDilatedAttentionV2Collective,
        )

        print(f"[GPU {rank}] Testing seq_len=1024 (small)")

        # Small test case
        model = RingDilatedAttentionV2Collective(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            ring_size=world_size,
            device=device,
            dtype=torch.float16,
            enable_memory_pool=False,
            use_pattern_cache=False,
        ).to(device)

        # Small inputs
        batch_size = 1
        seq_len = 1024
        num_heads = 4
        head_dim = 32

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Single forward pass
        with torch.amp.autocast("cuda"):
            output = model(q, k, v)

        if rank == 0:
            print(f"✓ Success! Output shape: {output.shape}")
            print(f"✓ Output valid: {not torch.isnan(output).any().item()}")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        dist.barrier()
        dist.destroy_process_group()


def main():
    """Run quick test."""
    print("Quick Ring V2 Collective Test")
    print("=" * 50)

    world_size = 2
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs")
        return

    try:
        mp.spawn(quick_test, args=(world_size,), nprocs=world_size, join=True)
        print("\n✅ Quick test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
