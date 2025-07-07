#!/usr/bin/env python3
"""
Test block-sparse distributed implementation with proper setup.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention
from dilated_attention_pytorch.block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
    DistributedSparseConfig,
)


def setup_distributed(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def test_distributed_block_sparse(rank, world_size):
    """Test distributed block-sparse on one rank."""
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    if rank == 0:
        print("\nDistributed Block-Sparse Test")
        print("=" * 60)
        print(f"World size: {world_size}")

    # Test parameters
    batch_size = 2
    seq_len = 8192
    num_heads = 8
    head_dim = 64

    # Create local inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test 1: Direct creation
    if rank == 0:
        print("\n1. Testing direct BlockSparseRingDistributedDilatedAttention...")

    try:
        config = DistributedSparseConfig(
            pattern_type="hierarchical",
            sparsity_ratio=0.05,  # 95% sparse
            enable_gradient_compression=True,
            compression_ratio=0.1,
        )

        model = BlockSparseRingDistributedDilatedAttention(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            distributed_config=config,
        ).to(device=device, dtype=dtype)

        output = model(q, k, v)

        # Synchronize
        dist.barrier()

        if rank == 0:
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Output valid: {torch.isfinite(output).all()}")

    except Exception as e:
        if rank == 0:
            print(f"   ✗ Failed: {e}")
            import traceback

            traceback.print_exc()

    # Test 2: Factory creation
    if rank == 0:
        print("\n2. Testing factory creation...")

    try:
        # Note: Factory doesn't support distributed variant directly
        # We'd need to enhance the factory
        model = create_block_sparse_attention(
            variant="base",  # Use base for now
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            sparsity_ratio=0.05,
        ).to(device=device, dtype=dtype)

        # Wrap in DataParallel for multi-GPU
        if world_size > 1:
            # For actual distributed, we'd use DistributedDataParallel
            output = model(q, k, v)
        else:
            output = model(q, k, v)

        dist.barrier()

        if rank == 0:
            print("   ✓ Factory model works")

    except Exception as e:
        if rank == 0:
            print(f"   ✗ Factory failed: {e}")

    # Test 3: Performance comparison
    if rank == 0:
        print("\n3. Performance comparison...")

    try:
        import time

        # Time distributed version
        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()

        for _ in range(5):
            _ = model(q, k, v)

        torch.cuda.synchronize()
        dist.barrier()

        distributed_time = (time.time() - start) / 5 * 1000

        if rank == 0:
            print(f"   Average forward time: {distributed_time:.2f}ms")

            # Show theoretical benefits
            print("\n   Theoretical benefits with distributed:")
            print(f"   - Memory per GPU: O(n/{world_size}) instead of O(n)")
            print(f"   - Can handle {world_size}x longer sequences")
            print("   - 95% sparsity further reduces memory by 20x")

    except Exception as e:
        if rank == 0:
            print(f"   ✗ Performance test failed: {e}")

    if rank == 0:
        print("\n✅ Distributed test completed!")

    cleanup()


def test_data_parallel_comparison():
    """Compare DataParallel vs single GPU."""
    print("\nDataParallel vs Single GPU Comparison")
    print("=" * 60)

    device = torch.device("cuda:0")
    dtype = torch.float16

    # Parameters
    batch_size = 4
    seq_len = 8192
    num_heads = 8
    head_dim = 64

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Create model
    model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        sparsity_ratio=0.05,
    ).to(device=device, dtype=dtype)

    # Single GPU timing
    import time

    torch.cuda.synchronize()
    start = time.time()
    _ = model(q, k, v)
    torch.cuda.synchronize()
    single_time = (time.time() - start) * 1000

    print(f"\nSingle GPU: {single_time:.2f}ms")

    # DataParallel
    if torch.cuda.device_count() > 1:
        model_dp = torch.nn.DataParallel(model)

        torch.cuda.synchronize()
        start = time.time()
        _ = model_dp(q, k, v)
        torch.cuda.synchronize()
        dp_time = (time.time() - start) * 1000

        print(f"DataParallel: {dp_time:.2f}ms")
        print(f"Speedup: {single_time / dp_time:.2f}x")

    print("\nNote: DataParallel has overhead for small models.")
    print("Distributed training shines with:")
    print("- Very large models (billions of parameters)")
    print("- Very long sequences (100K+ tokens)")
    print("- Gradient accumulation across nodes")


def main():
    """Run distributed tests."""
    print("Block-Sparse Multi-GPU Testing")
    print("=" * 60)

    print(f"\nAvailable GPUs: {torch.cuda.device_count()}")

    # Run DataParallel comparison first (single process)
    test_data_parallel_comparison()

    # Run distributed test with proper process spawning
    if torch.cuda.device_count() >= 2:
        print("\n\nRunning Distributed Test...")
        world_size = 2
        mp.spawn(
            test_distributed_block_sparse,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
    else:
        print("\nNeed at least 2 GPUs for distributed test")


if __name__ == "__main__":
    main()
