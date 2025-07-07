#!/usr/bin/env python3
"""
Test that the distributed block-sparse initialization is fixed.
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


def test_direct_initialization():
    """Test direct initialization of BlockSparseRingDistributedDilatedAttention."""
    print("Testing direct initialization...")

    try:
        config = DistributedSparseConfig(
            pattern_type="hierarchical",
            sparsity_ratio=0.05,
        )

        # This should work now with embed_dim and num_heads
        model = BlockSparseRingDistributedDilatedAttention(
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            distributed_config=config,
        )

        print("✓ Direct initialization successful!")
        print(f"  Model type: {type(model).__name__}")

        # Test forward pass
        batch_size = 2
        seq_len = 4096
        num_heads = 12
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            q, k, v = q.cuda(), k.cuda(), v.cuda()

        output = model(q, k, v)
        print(f"  Output shape: {output.shape}")
        print(f"  Output valid: {torch.isfinite(output).all()}")

        return True

    except Exception as e:
        print(f"✗ Direct initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factory_initialization():
    """Test factory initialization of distributed variant."""
    print("\nTesting factory initialization...")

    try:
        # Using factory with distributed variant
        model = create_block_sparse_attention(
            variant="distributed",
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            distributed_config=DistributedSparseConfig(
                sparsity_ratio=0.05,
                pattern_type="hierarchical",
            ),
        )

        print("✓ Factory initialization successful!")
        print(f"  Model type: {type(model).__name__}")

        # Test with default embed_dim and num_heads
        _ = create_block_sparse_attention(
            variant="distributed",
            segment_lengths=[2048],
            dilation_rates=[1],
        )

        print("✓ Factory with defaults successful!")

        return True

    except Exception as e:
        print(f"✗ Factory initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def setup_distributed(rank, world_size):
    """Setup distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def test_distributed_usage(rank, world_size):
    """Test in actual distributed environment."""
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("\nTesting in distributed environment...")

    try:
        # Create model
        model = create_block_sparse_attention(
            variant="distributed",
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048],
            dilation_rates=[1],
        ).to(device)

        # Test forward pass
        q = torch.randn(1, 2048, 12, 64, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        output = model(q, k, v)

        dist.barrier()

        if rank == 0:
            print("✓ Distributed test successful!")
            print(f"  Output shape: {output.shape}")

    except Exception as e:
        if rank == 0:
            print(f"✗ Distributed test failed: {e}")
            import traceback

            traceback.print_exc()

    cleanup()


def main():
    """Run all initialization tests."""
    print("Distributed Block-Sparse Initialization Tests")
    print("=" * 60)

    # Test 1: Direct initialization
    direct_success = test_direct_initialization()

    # Test 2: Factory initialization
    factory_success = test_factory_initialization()

    # Test 3: Distributed environment (if 2+ GPUs available)
    if torch.cuda.device_count() >= 2:
        print("\nRunning distributed test...")
        mp.spawn(test_distributed_usage, args=(2,), nprocs=2, join=True)
    else:
        print("\nSkipping distributed test (need 2+ GPUs)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Direct initialization: {'✓ PASS' if direct_success else '✗ FAIL'}")
    print(f"  Factory initialization: {'✓ PASS' if factory_success else '✗ FAIL'}")

    if direct_success and factory_success:
        print("\n✅ All initialization tests passed!")
    else:
        print("\n❌ Some tests failed")


if __name__ == "__main__":
    main()
