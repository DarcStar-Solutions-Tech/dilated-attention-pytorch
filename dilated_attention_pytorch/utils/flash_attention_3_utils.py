"""
Flash Attention 3 specific utilities and optimizations.

This module provides FA3-specific features including:
- Block-sparse FA3 patterns
- H100 Tensor Core optimization
- FP8 precision support
- Asynchronous computation management
"""

from typing import Any

import torch
from torch import Tensor

from ..core.constants import GPU_TYPE, HAS_FLASH_ATTN_3


def create_fa3_block_sparse_mask(
    seq_len: int,
    block_size: int = 128,
    sparsity_ratio: float = 0.9,
    pattern_type: str = "dilated_sparse",
    device: torch.device | None = None,
) -> Tensor:
    """
    Create optimized block-sparse mask for Flash Attention 3.

    FA3 supports native block-sparse patterns with significant speedups.

    Args:
        seq_len: Sequence length (must be divisible by block_size)
        block_size: Block size for sparsity (FA3 optimized for 128)
        sparsity_ratio: Fraction of blocks to keep sparse (0.9 = 90% sparse)
        pattern_type: Type of sparse pattern
        device: Device to create mask on

    Returns:
        Block sparse mask tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_blocks = seq_len // block_size

    if pattern_type == "dilated_sparse":
        # Create dilated pattern optimized for FA3
        mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)

        # Local attention within blocks
        for i in range(num_blocks):
            mask[i, max(0, i - 1) : min(num_blocks, i + 2)] = True

        # Dilated attention with exponentially increasing stride
        stride = 2
        while stride < num_blocks:
            for i in range(0, num_blocks, stride):
                if i < num_blocks:
                    mask[i, ::stride] = True
                    mask[::stride, i] = True
            stride *= 2

    elif pattern_type == "local_window":
        # Local window pattern
        mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)
        window_blocks = max(1, int((1 - sparsity_ratio) * num_blocks))

        for i in range(num_blocks):
            start = max(0, i - window_blocks // 2)
            end = min(num_blocks, i + window_blocks // 2 + 1)
            mask[i, start:end] = True

    else:
        # Fixed sparsity pattern
        mask = torch.rand(num_blocks, num_blocks, device=device) > sparsity_ratio

    return mask


def get_fa3_config(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    use_fp8: bool = False,
    enable_async: bool = True,
) -> dict[str, Any]:
    """
    Get optimal Flash Attention 3 configuration for given parameters.

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        use_fp8: Whether to use FP8 precision (H100 only)
        enable_async: Whether to enable async computation

    Returns:
        FA3 configuration dict
    """
    config = {
        "block_size": 128,  # FA3 optimized for 128
        "use_fp8": False,
        "enable_async": enable_async,
        "warp_specialized": True,
        "persistent_kernel": seq_len >= 8192,
        "num_splits": 1,
    }

    # H100-specific optimizations
    if str(GPU_TYPE) == "h100":
        config["use_fp8"] = use_fp8
        config["block_size"] = 256 if seq_len >= 16384 else 128
        config["num_splits"] = 2 if seq_len >= 32768 else 1
        config["tensor_core_enabled"] = True

    # Adjust for memory constraints
    if seq_len * num_heads * head_dim > 1e9:  # >1B elements
        config["persistent_kernel"] = False
        config["num_splits"] = 4

    return config


def apply_fa3_optimization(
    module: torch.nn.Module,
    seq_len: int,
    use_fp8: bool = False,
) -> None:
    """
    Apply Flash Attention 3 optimizations to a module in-place.

    Args:
        module: PyTorch module to optimize
        seq_len: Expected sequence length
        use_fp8: Whether to use FP8 precision
    """
    if not HAS_FLASH_ATTN_3:
        return

    # Set FA3-specific attributes
    if hasattr(module, "attention"):
        module.attention.use_flash_attention_3 = True
        module.attention.fa3_config = get_fa3_config(
            seq_len=seq_len,
            num_heads=getattr(module, "num_heads", 8),
            head_dim=getattr(module, "head_dim", 64),
            use_fp8=use_fp8,
        )

    # Enable TF32 for H100
    if str(GPU_TYPE) == "h100":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def benchmark_fa3_vs_fa2(
    batch_size: int = 2,
    seq_len: int = 8192,
    num_heads: int = 8,
    head_dim: int = 64,
    num_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark Flash Attention 3 vs Flash Attention 2.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of heads
        head_dim: Head dimension
        num_runs: Number of benchmark runs

    Returns:
        Dict with timing results
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    dtype = torch.float16

    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    results = {}

    # Import HAS_FLASH_ATTN from constants
    from ..core.constants import HAS_FLASH_ATTN

    # Benchmark FA2
    if HAS_FLASH_ATTN:
        try:
            from flash_attn import flash_attn_func

            # Warmup
            for _ in range(10):
                _ = flash_attn_func(q, k, v, causal=True)

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(num_runs):
                _ = flash_attn_func(q, k, v, causal=True)
            end.record()

            torch.cuda.synchronize()
            results["fa2_ms"] = start.elapsed_time(end) / num_runs

        except Exception as e:
            results["fa2_error"] = str(e)

    # Benchmark FA3
    if HAS_FLASH_ATTN_3:
        try:
            from flash_attn_interface import flash_attn_func_v3

            # Warmup
            for _ in range(10):
                _ = flash_attn_func_v3(q, k, v, causal=True)

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(num_runs):
                _ = flash_attn_func_v3(q, k, v, causal=True)
            end.record()

            torch.cuda.synchronize()
            results["fa3_ms"] = start.elapsed_time(end) / num_runs

            # Calculate speedup
            if "fa2_ms" in results and "fa3_ms" in results:
                results["speedup"] = results["fa2_ms"] / results["fa3_ms"]

        except Exception as e:
            results["fa3_error"] = str(e)

    # Add configuration info
    results["config"] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "gpu": torch.cuda.get_device_name(0),
    }

    return results
