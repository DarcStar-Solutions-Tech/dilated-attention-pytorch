"""Data generation utilities for benchmarks."""

from typing import Dict, List, Optional, Tuple
import torch


def generate_qkv_data(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate standard QKV tensors for attention.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        device: Device to create tensors on
        seed: Random seed for reproducibility

    Returns:
        Tuple of (query, key, value) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)

    return q, k, v


def generate_attention_mask(
    batch_size: int,
    seq_len: int,
    mask_type: str = "none",
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
) -> Optional[torch.Tensor]:
    """Generate attention mask.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_type: Type of mask ("none", "causal", "random")
        dtype: Data type
        device: Device to create mask on

    Returns:
        Attention mask tensor or None
    """
    if mask_type == "none":
        return None
    elif mask_type == "causal":
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask.expand(batch_size, 1, seq_len, seq_len)
    elif mask_type == "random":
        # Random mask with 10% masked positions
        mask = torch.rand(batch_size, 1, seq_len, seq_len, device=device) < 0.1
        return mask.masked_fill(mask, float("-inf"))
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def get_standard_configs() -> Dict[str, Dict[str, List[int]]]:
    """Get standard benchmark configurations.

    Returns:
        Dictionary of configuration presets
    """
    return {
        "tiny": {
            "segment_lengths": [512],
            "dilation_rates": [1],
            "seq_lengths": [1024, 2048],
            "batch_sizes": [1, 2],
            "num_heads": [8],
            "head_dims": [64],
        },
        "small": {
            "segment_lengths": [1024, 2048],
            "dilation_rates": [1, 2],
            "seq_lengths": [4096, 8192],
            "batch_sizes": [1, 2, 4],
            "num_heads": [8, 16],
            "head_dims": [64, 128],
        },
        "medium": {
            "segment_lengths": [2048, 4096, 8192],
            "dilation_rates": [1, 2, 4],
            "seq_lengths": [16384, 32768, 65536],
            "batch_sizes": [1, 2],
            "num_heads": [16, 32],
            "head_dims": [64, 128],
        },
        "large": {
            "segment_lengths": [4096, 8192, 16384, 32768],
            "dilation_rates": [1, 2, 4, 8],
            "seq_lengths": [131072, 262144],
            "batch_sizes": [1],
            "num_heads": [32],
            "head_dims": [128],
        },
        "extreme": {
            "segment_lengths": [8192, 16384, 32768, 65536],
            "dilation_rates": [1, 2, 4, 8],
            "seq_lengths": [524288, 1048576],
            "batch_sizes": [1],
            "num_heads": [32],
            "head_dims": [128],
        },
    }


def get_model_configs() -> Dict[str, Dict[str, int]]:
    """Get standard model configurations.

    Returns:
        Dictionary of model presets
    """
    return {
        "gpt2": {
            "num_heads": 12,
            "head_dim": 64,
            "embed_dim": 768,
        },
        "gpt3_small": {
            "num_heads": 12,
            "head_dim": 96,
            "embed_dim": 1152,
        },
        "gpt3_medium": {
            "num_heads": 16,
            "head_dim": 96,
            "embed_dim": 1536,
        },
        "gpt3_large": {
            "num_heads": 16,
            "head_dim": 128,
            "embed_dim": 2048,
        },
        "llama_7b": {
            "num_heads": 32,
            "head_dim": 128,
            "embed_dim": 4096,
        },
        "llama_13b": {
            "num_heads": 40,
            "head_dim": 128,
            "embed_dim": 5120,
        },
        "llama_70b": {
            "num_heads": 64,
            "head_dim": 128,
            "embed_dim": 8192,
        },
    }


def create_test_batch(
    config_name: str,
    seq_len: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Create test batch based on model configuration.

    Args:
        config_name: Name of model configuration
        seq_len: Sequence length
        batch_size: Batch size
        dtype: Data type
        device: Device
        seed: Random seed

    Returns:
        Dictionary with q, k, v tensors
    """
    configs = get_model_configs()
    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(configs.keys())}"
        )

    config = configs[config_name]
    q, k, v = generate_qkv_data(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=config["num_heads"],
        head_dim=config["head_dim"],
        dtype=dtype,
        device=device,
        seed=seed,
    )

    return {"q": q, "k": k, "v": v, "config": config}
