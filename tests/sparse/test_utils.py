"""
Shared test utilities for block-sparse attention tests.

This module provides common test functions and fixtures to reduce duplication
across different test files.
"""

import torch
from typing import Dict, Any, Optional, Tuple
import pytest


# Common test configurations
TEST_CONFIGS = {
    "tiny": {
        "batch_size": 2,
        "seq_len": 256,
        "num_heads": 4,
        "head_dim": 16,
        "embed_dim": 64,
    },
    "small": {
        "batch_size": 2,
        "seq_len": 1024,
        "num_heads": 4,
        "head_dim": 32,
        "embed_dim": 128,
    },
    "medium": {
        "batch_size": 4,
        "seq_len": 4096,
        "num_heads": 8,
        "head_dim": 64,
        "embed_dim": 512,
    },
    "large": {
        "batch_size": 2,
        "seq_len": 16384,
        "num_heads": 16,
        "head_dim": 64,
        "embed_dim": 1024,
    },
}


def get_device_for_test(prefer_cuda: bool = True) -> torch.device:
    """Get appropriate device for testing, considering memory constraints."""
    if prefer_cuda and torch.cuda.is_available():
        # Check available memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        if available_memory < 8 * 1024**3:  # Less than 8GB
            # For small GPUs, might want to use CPU for some tests
            return torch.device("cuda")
        return torch.device("cuda")
    return torch.device("cpu")


def create_test_tensors(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V tensors for testing.

    Args:
        config: Test configuration dict with batch_size, seq_len, num_heads, head_dim
        device: Device to create tensors on (default: auto-select)
        dtype: Data type for tensors

    Returns:
        Tuple of (q, k, v) tensors with shape [batch, seq_len, num_heads, head_dim]
    """
    if device is None:
        device = get_device_for_test()

    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    return q, k, v


def assert_valid_attention_output(
    output: torch.Tensor,
    expected_shape: Tuple[int, ...],
    check_finite: bool = True,
    check_normalized: bool = False,
) -> None:
    """Assert that attention output is valid.

    Args:
        output: Attention output tensor
        expected_shape: Expected shape of output
        check_finite: Whether to check for NaN/Inf values
        check_normalized: Whether to check if attention weights sum to 1
    """
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )

    if check_finite:
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    if check_normalized:
        # For checking attention weights normalization
        # This would be used when testing attention weights directly
        weight_sum = output.sum(dim=-1)
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6), (
            "Attention weights do not sum to 1"
        )


def run_standard_forward_pass_test(
    attention_module: torch.nn.Module,
    config_name: str = "small",
    device: Optional[torch.device] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run a standard forward pass test on an attention module.

    Args:
        attention_module: The attention module to test
        config_name: Name of test configuration to use
        device: Device to run test on (default: auto-select)
        is_causal: Whether to use causal masking

    Returns:
        Output tensor from forward pass
    """
    config = TEST_CONFIGS[config_name]

    # Create test tensors
    q, k, v = create_test_tensors(config, device)

    # Move module to device if needed
    if device is not None:
        attention_module = attention_module.to(device)

    # Run forward pass
    if hasattr(attention_module, "forward"):
        # Check what parameters the forward method accepts
        import inspect

        sig = inspect.signature(attention_module.forward)
        params = sig.parameters

        if "is_causal" in params:
            output = attention_module(q, k, v, is_causal=is_causal)
        else:
            output = attention_module(q, k, v)
    else:
        raise AttributeError(f"{type(attention_module).__name__} has no forward method")

    # Validate output
    assert_valid_attention_output(output, q.shape)

    return output


def skip_if_insufficient_memory(config_name: str, min_gpu_memory_gb: float = 8.0):
    """Decorator to skip test if insufficient GPU memory for config."""

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                available_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )

                # Estimate memory needed based on config
                config = TEST_CONFIGS[config_name]
                seq_len = config["seq_len"]
                embed_dim = config["embed_dim"]
                batch_size = config["batch_size"]

                # Rough estimate: 4 bytes per float32 element
                # Q, K, V, output, and intermediate tensors
                estimated_gb = (5 * batch_size * seq_len * embed_dim * 4) / (1024**3)

                if available_gb < max(
                    min_gpu_memory_gb, estimated_gb * 1.5
                ):  # 1.5x safety factor
                    pytest.skip(f"Insufficient GPU memory for {config_name} config")

            return test_func(*args, **kwargs)

        return wrapper

    return decorator


# Parameterized test decorators for common test patterns
def parametrize_sparsity_ratios():
    """Common sparsity ratios for testing."""
    return pytest.mark.parametrize("sparsity_ratio", [0.1, 0.25, 0.5, 0.75])


def parametrize_test_configs(exclude_large: bool = True):
    """Common test configurations."""
    configs = ["tiny", "small", "medium"]
    if not exclude_large:
        configs.append("large")
    return pytest.mark.parametrize("config_name", configs)


def parametrize_pattern_types():
    """Common sparse pattern types."""
    return pytest.mark.parametrize(
        "pattern_type", ["local_window", "dilated_sparse", "global_local"]
    )
