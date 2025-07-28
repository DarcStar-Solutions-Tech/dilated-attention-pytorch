#!/usr/bin/env python3
"""
Verify that ring attention implementations properly split sequences across GPUs.
"""

import torch
import torch.distributed as dist
import os

from dilated_attention_pytorch import (
    StandardRingAttention,
    DistributedRingAttention,
    HilbertRingAttention,
    RingBlockSparseAttention,
    RingAttentionConfig,
)
from dilated_attention_pytorch.utils import get_optimal_dtype


def verify_implementation(impl_name: str, impl_class):
    """Verify a single implementation."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    print(f"\n[Rank {rank}] Verifying {impl_name}")
    print(f"[Rank {rank}] World size: {world_size}")

    # Test parameters
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    # Create config
    config = RingAttentionConfig(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        dropout=0.0,
    )

    # Select optimal dtype based on GPU architecture
    dtype = get_optimal_dtype(device)

    # Create model
    if impl_name == "BlockSparse":
        model = impl_class(
            config=config,
            sparsity_ratio=0.1,
            device=device,
            dtype=dtype,
        )
    else:
        model = impl_class(config=config, device=device, dtype=dtype)

    # Expected local sequence length
    expected_local_seq = seq_len // world_size if world_size > 1 else seq_len
    print(f"[Rank {rank}] Expected local sequence length: {expected_local_seq}")

    # Create FULL sequence tensors (this is what user would pass)
    torch.manual_seed(42)  # Same seed on all ranks
    q_full = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k_full = torch.randn_like(q_full)
    v_full = torch.randn_like(q_full)

    # Monitor memory before
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**2)  # MB
        print(f"[Rank {rank}] Memory before forward: {mem_before:.2f} MB")

    # Forward pass
    try:
        # Test 1: Pass full sequence (implementation should split internally)
        print(f"[Rank {rank}] Test 1: Passing full sequence ({seq_len} tokens)")
        output = model(q_full, k_full, v_full)
        print(f"[Rank {rank}] ✓ Output shape: {output.shape}")

        # Check if output is correct size
        if output.shape[1] != seq_len:
            print(
                f"[Rank {rank}] ✗ ERROR: Output has wrong sequence length! Expected {seq_len}, got {output.shape[1]}"
            )

        # Memory check
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            mem_used = peak_mem - mem_before
            print(
                f"[Rank {rank}] Peak memory: {peak_mem:.2f} MB (used: {mem_used:.2f} MB)"
            )

            # Rough check: if truly splitting, memory should be ~1/world_size
            if world_size > 1:
                _ = 1.0 / world_size
                # Allow some overhead
                if mem_used > (mem_before * 2):  # If using more than 2x initial memory
                    print(
                        f"[Rank {rank}] ⚠️  WARNING: Memory usage suggests full sequence processing!"
                    )

        # Test 2: Check if model has already_split parameter
        if (
            hasattr(model.forward, "__code__")
            and "already_split" in model.forward.__code__.co_varnames
        ):
            print(f"\n[Rank {rank}] Test 2: Model supports already_split parameter")

            # Manually split sequences
            if world_size > 1:
                chunk_size = seq_len // world_size
                start_idx = rank * chunk_size
                end_idx = start_idx + chunk_size

                q_local = q_full[:, start_idx:end_idx].contiguous()
                k_local = k_full[:, start_idx:end_idx].contiguous()
                v_local = v_full[:, start_idx:end_idx].contiguous()

                print(f"[Rank {rank}] Passing pre-split sequence ({chunk_size} tokens)")
                output_local = model(q_local, k_local, v_local, already_split=True)
                print(f"[Rank {rank}] ✓ Local output shape: {output_local.shape}")

    except Exception as e:
        print(f"[Rank {rank}] ✗ ERROR during forward pass: {str(e)}")
        import traceback

        traceback.print_exc()

    # Cleanup
    del model, q_full, k_full, v_full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Synchronize before next test
    if dist.is_initialized():
        dist.barrier()


def check_source_code():
    """Check source code for proper sequence splitting."""
    print("\n" + "=" * 60)
    print("Checking source code for sequence splitting patterns...")
    print("=" * 60)

    # Import implementations to check their forward methods
    from dilated_attention_pytorch.ring.base import StandardRingAttention

    # Check if StandardRingAttention splits sequences
    import inspect

    source = inspect.getsource(StandardRingAttention.forward)

    # Look for splitting patterns
    splitting_patterns = [
        "chunk",
        "split",
        "local_seq",
        "start_idx",
        "end_idx",
        "world_size",
        "get_rank",
    ]

    found_patterns = []
    for pattern in splitting_patterns:
        if pattern in source:
            found_patterns.append(pattern)

    if found_patterns:
        print(f"✓ Found splitting patterns in StandardRingAttention: {found_patterns}")
    else:
        print("✗ No obvious splitting patterns found in StandardRingAttention")

    # Check for ring communication
    ring_patterns = ["ring_pass", "isend", "irecv", "send", "recv"]
    found_ring = [p for p in ring_patterns if p in source]

    if found_ring:
        print(f"✓ Found ring communication patterns: {found_ring}")
    else:
        print("✗ No ring communication patterns found")


def main():
    """Main verification routine."""
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        print(f"[Rank {rank}] Distributed initialized")
    else:
        rank = 0
        print("Running in single GPU mode")

    # First check source code (only on rank 0)
    if rank == 0:
        check_source_code()

    if dist.is_initialized():
        dist.barrier()

    # Test implementations
    implementations = {
        "Standard": StandardRingAttention,
        "Distributed": DistributedRingAttention,
        "Hilbert": HilbertRingAttention,
        "BlockSparse": RingBlockSparseAttention,
    }

    for name, cls in implementations.items():
        try:
            verify_implementation(name, cls)
        except Exception as e:
            print(f"\n[Rank {rank}] Failed to verify {name}: {str(e)}")

    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"\n[Rank {rank}] Verification complete")


if __name__ == "__main__":
    main()
