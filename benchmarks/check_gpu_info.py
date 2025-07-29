#!/usr/bin/env python3
"""Check GPU info and constraints."""

import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(device)}")

    # Get memory info
    props = torch.cuda.get_device_properties(device)
    print("\nMemory Constraints:")
    print(f"  Total Global Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Shared Memory Per Block: {props.shared_memory_per_block:,} bytes")
    print(f"  Max Threads Per Block: {props.max_threads_per_block}")
    print(f"  Warp Size: {props.warp_size}")
    print(f"  Max Block Dimensions: {props.max_block_dim}")
    print(f"  Max Grid Dimensions: {props.max_grid_dim}")

    # Calculate optimal block sizes based on shared memory
    head_dim = 64
    dtype_size = 4  # float32

    print(f"\nOptimal Block Sizes (for head_dim={head_dim}):")
    for block_m in [32, 64, 128]:
        for block_n in [32, 64, 128]:
            # Estimate shared memory usage
            # Need to store tiles of Q, K, V
            shared_mem = 3 * block_m * head_dim * dtype_size
            shared_mem += 3 * block_n * head_dim * dtype_size

            if shared_mem <= props.max_shared_memory_per_block:
                print(f"  BLOCK_M={block_m}, BLOCK_N={block_n}: {shared_mem:,} bytes ✓")
            else:
                print(
                    f"  BLOCK_M={block_m}, BLOCK_N={block_n}: {shared_mem:,} bytes ✗ (exceeds limit)"
                )
else:
    print("No CUDA device available")
