#!/usr/bin/env python3
"""
Debug dtype selection in hybrid implementation.
"""

import torch
import torch.distributed as dist
import os


def debug_dtype():
    # Initialize distributed if needed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {rank}] Device: {device}")

    # Check GPU info
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability(device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[Rank {rank}] GPU: {gpu_name}")
        print(f"[Rank {rank}] Compute Capability: {cc}")

    # Test gpu_utils
    try:
        from dilated_attention_pytorch.utils.gpu_utils import get_optimal_dtype

        optimal = get_optimal_dtype(device, prefer_fp16=True, warn_pascal=False)
        print(f"[Rank {rank}] get_optimal_dtype returned: {optimal}")
    except Exception as e:
        print(f"[Rank {rank}] gpu_utils error: {e}")

    # Create hybrid model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    model = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[1],
        device=device,
        # dtype not specified - should auto-select
    )

    print(f"[Rank {rank}] Model dtype: {model.dtype}")

    # Test with inputs
    q = torch.randn(1, 512, 4, 32, device=device, dtype=model.dtype) * 0.1
    k = torch.randn(1, 512, 4, 32, device=device, dtype=model.dtype) * 0.1
    v = torch.randn(1, 512, 4, 32, device=device, dtype=model.dtype) * 0.1

    try:
        output = model(q, k, v, is_causal=False)
        print(f"[Rank {rank}] Forward pass successful!")
        print(f"[Rank {rank}] Output dtype: {output.dtype}")
    except Exception as e:
        print(f"[Rank {rank}] Forward pass failed: {e}")

    if "RANK" in os.environ:
        dist.destroy_process_group()


if __name__ == "__main__":
    debug_dtype()
