"""
GPU utilities for device detection and optimization.

This module provides utilities for detecting GPU architectures and
selecting optimal configurations based on hardware capabilities.
"""

import warnings
from typing import Optional, Tuple

import torch


def get_gpu_compute_capability(device: torch.device) -> Optional[Tuple[int, int]]:
    """
    Get the compute capability of a GPU device.

    Args:
        device: PyTorch device

    Returns:
        Tuple of (major, minor) compute capability or None if not available
    """
    if device.type != "cuda":
        return None

    try:
        # Get device properties
        props = torch.cuda.get_device_properties(device)
        return (props.major, props.minor)
    except Exception:
        return None


def is_pascal_or_older(device: torch.device) -> bool:
    """
    Check if GPU is Pascal architecture or older (compute < 7.0).

    Pascal GPUs have limited FP16 performance and no Tensor Cores.

    Args:
        device: PyTorch device

    Returns:
        True if Pascal or older, False otherwise
    """
    capability = get_gpu_compute_capability(device)
    if capability is None:
        return False

    major, minor = capability
    return major < 7


def get_optimal_dtype(
    device: torch.device, prefer_fp16: bool = True, warn_pascal: bool = True
) -> torch.dtype:
    """
    Get optimal dtype for a device based on its architecture.

    Pascal and older GPUs often perform worse with FP16 due to:
    - No Tensor Core support
    - Limited FP16 compute units
    - Overhead from FP16/FP32 conversions

    Args:
        device: PyTorch device
        prefer_fp16: Whether to prefer FP16 on capable devices
        warn_pascal: Whether to warn about Pascal FP16 performance

    Returns:
        Optimal dtype for the device (torch.float32 or torch.float16)
    """
    if device.type != "cuda":
        return torch.float32

    # Check compute capability
    capability = get_gpu_compute_capability(device)
    if capability is None:
        return torch.float32

    major, minor = capability

    # Pascal and older (compute < 7.0)
    if major < 7:
        if prefer_fp16 and warn_pascal:
            props = torch.cuda.get_device_properties(device)
            warnings.warn(
                f"GPU {props.name} (compute {major}.{minor}) has limited FP16 performance. "
                f"Using FP32 for optimal performance. Pascal GPUs can be 5-10x slower with FP16.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.float32

    # Volta and newer
    if prefer_fp16:
        # Check if bfloat16 is available (Ampere+)
        if major >= 8 and hasattr(torch, "bfloat16") and torch.cuda.is_bf16_supported():
            # Could return torch.bfloat16 here, but FP16 is more widely supported
            return torch.float16
        else:
            return torch.float16
    else:
        return torch.float32


def warn_suboptimal_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """
    Warn if using a suboptimal dtype for the device.

    Args:
        device: PyTorch device
        dtype: Current dtype being used
    """
    if device.type != "cuda":
        return

    capability = get_gpu_compute_capability(device)
    if capability is None:
        return

    major, minor = capability

    # Warn if using FP16 on Pascal
    if major < 7 and dtype in [torch.float16, torch.bfloat16]:
        props = torch.cuda.get_device_properties(device)
        warnings.warn(
            f"Using {dtype} on Pascal GPU {props.name} may result in poor performance. "
            f"Consider using torch.float32 instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Warn if using FP32 on modern GPUs when FP16 would be faster
    elif major >= 7 and dtype == torch.float32:
        props = torch.cuda.get_device_properties(device)
        warnings.warn(
            f"GPU {props.name} supports efficient FP16. "
            f"Consider using torch.float16 for better performance.",
            RuntimeWarning,
            stacklevel=2,
        )
