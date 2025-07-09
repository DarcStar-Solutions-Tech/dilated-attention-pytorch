#!/usr/bin/env python3
"""
Test GPU memory availability and basic operations.
"""

import torch
import gc


def test_gpu_memory():
    """Test GPU memory with simple operations."""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    # Get GPU info
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")

        # Set device
        torch.cuda.set_device(i)
        device = torch.device(f"cuda:{i}")

        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Get memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3

        print(f"  Total memory: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Available: {total_memory - reserved:.2f} GB")

        # Test allocation
        try:
            # Try to allocate a reasonable tensor
            test_size = 4096  # 4K tokens
            test_tensor = torch.randn(
                1, test_size, 768, device=device, dtype=torch.float32
            )
            print(f"  ✓ Can allocate tensor (1, {test_size}, 768)")

            # Check memory after allocation
            allocated_after = torch.cuda.memory_allocated(device) / 1024**2
            print(f"  Memory used by test tensor: {allocated_after:.1f} MB")

            # Try QKV projection
            qkv = torch.randn(
                1, test_size, 3, 12, 64, device=device, dtype=torch.float32
            )
            print("  ✓ Can allocate QKV tensor")

            # Check total
            total_allocated = torch.cuda.memory_allocated(device) / 1024**2
            print(f"  Total allocated: {total_allocated:.1f} MB")

            del test_tensor, qkv

        except torch.cuda.OutOfMemoryError:
            print("  ✗ OOM during test allocation")

        # Clear again
        gc.collect()
        torch.cuda.empty_cache()


def test_distributed_memory():
    """Test memory with distributed setup."""
    import os

    if "RANK" in os.environ:
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"\nDistributed test with {world_size} GPUs")

        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()

        # Check available memory
        props = torch.cuda.get_device_properties(device)
        _ = props.total_memory / 1024**3

        # Synchronize
        dist.barrier()

        # Try allocation
        try:
            local_seq = 8192 // world_size
            x = torch.randn(1, local_seq, 768, device=device, dtype=torch.float32)

            mem_mb = torch.cuda.memory_allocated(device) / 1024**2

            if rank == 0:
                print(f"Rank {rank}: Allocated {local_seq} tokens, {mem_mb:.1f} MB")

            del x

        except Exception as e:
            print(f"Rank {rank}: Error - {e}")

        dist.destroy_process_group()
    else:
        print("\nNot in distributed mode. Run with torchrun for distributed test.")


if __name__ == "__main__":
    print("=" * 60)
    print("GPU Memory Test")
    print("=" * 60)

    test_gpu_memory()
    test_distributed_memory()
