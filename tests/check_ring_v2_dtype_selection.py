#!/usr/bin/env python3
"""
Check if RingDilatedAttentionV2Collective properly selects dtype based on GPU architecture.
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def test_dtype_selection():
    """Test dtype selection in both single and distributed mode."""

    # Setup distributed if available
    is_distributed = "WORLD_SIZE" in os.environ

    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Testing dtype selection in Ring Attention V2")
        print(f"World size: {world_size}")
        print(f"{'=' * 60}\n")

    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        compute_capability = torch.cuda.get_device_capability(device)
        if rank == 0:
            print(f"GPU {device}: {gpu_name}")
            print(f"Compute capability: {compute_capability}")
            print(
                f"Expected dtype: {'float32' if compute_capability[0] < 7 else 'float16'}\n"
            )

    # Test 1: Default dtype selection (no dtype specified)
    if rank == 0:
        print("Test 1: Default dtype selection")

    model1 = RingDilatedAttentionV2Collective(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        ring_size=world_size,
        device=device,
        # dtype not specified - should auto-select
    )

    if rank == 0:
        print(f"  Model dtype: {model1.dtype}")
        print(f"  Device: {model1.device}")

    # Test 2: Check if gpu_utils import worked
    if rank == 0:
        print("\nTest 2: Checking gpu_utils availability")

    try:
        from dilated_attention_pytorch.utils.gpu_utils import get_optimal_dtype

        gpu_utils_available = True
        optimal_dtype = get_optimal_dtype(device, prefer_fp16=True, warn_pascal=False)
        if rank == 0:
            print("  gpu_utils available: Yes")
            print(f"  get_optimal_dtype returned: {optimal_dtype}")
    except ImportError as e:
        gpu_utils_available = False
        if rank == 0:
            print("  gpu_utils available: No")
            print(f"  Import error: {e}")

    # Test 3: Force dtype and check
    if rank == 0:
        print("\nTest 3: Forcing different dtypes")

    for dtype in [torch.float16, torch.float32]:
        model = RingDilatedAttentionV2Collective(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            ring_size=world_size,
            device=device,
            dtype=dtype,
        )
        if rank == 0:
            print(f"  Requested: {dtype}, Got: {model.dtype}")

    # Test 4: Create input and check computation dtype
    if rank == 0:
        print("\nTest 4: Checking computation dtype")

    # Create model with auto dtype
    model = RingDilatedAttentionV2Collective(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        ring_size=world_size,
        device=device,
    )

    # Create input
    seq_len = 4096
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Test with different input dtypes
    for input_dtype in [torch.float16, torch.float32]:
        x = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=input_dtype
        )

        try:
            with torch.no_grad():
                output = model(x, x, x, is_causal=False)

            if rank == 0:
                print(f"  Input dtype: {input_dtype}, Output dtype: {output.dtype}")
                print(f"  Model internal dtype: {model.dtype}")
        except Exception as e:
            if rank == 0:
                print(f"  Error with input dtype {input_dtype}: {e}")

    # Test 5: Check if dtype affects memory usage
    if rank == 0:
        print("\nTest 5: Memory usage with different dtypes")

    torch.cuda.empty_cache()

    for dtype in [torch.float32, torch.float16]:
        torch.cuda.reset_peak_memory_stats()

        model = RingDilatedAttentionV2Collective(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            ring_size=world_size,
            device=device,
            dtype=dtype,
        )

        x = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        with torch.no_grad():
            output = model(x, x, x, is_causal=False)

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        if rank == 0:
            print(f"  dtype={dtype}: Peak memory = {peak_memory_mb:.1f} MB")

    # Summary
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY:")
        print(f"{'=' * 60}")

        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(device)
            expected_dtype = torch.float32 if cc[0] < 7 else torch.float16

            print(f"GPU Architecture: Compute {cc[0]}.{cc[1]}")
            print(f"Expected dtype: {expected_dtype}")
            print(f"Actual dtype selected: {model1.dtype}")

            if gpu_utils_available:
                print("GPU utils available: YES")
            else:
                print("GPU utils available: NO (falling back to simple logic)")

            if model1.dtype == expected_dtype:
                print("\n✓ Dtype selection is correct for this GPU architecture")
            else:
                print("\n✗ Dtype selection is incorrect!")
                print(f"  Expected {expected_dtype} but got {model1.dtype}")

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_dtype_selection()
