"""GPU utility functions for architecture detection and optimization."""

import warnings
from typing import Optional

import torch


def get_gpu_compute_capability(device: torch.device) -> Optional[tuple[int, int]]:
    """
    Get the compute capability of a GPU device.

    Args:
        device: PyTorch device

    Returns:
        Tuple of (major, minor) compute capability, or None if not CUDA
    """
    if device.type != "cuda":
        return None

    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def is_pascal_or_older(device: torch.device) -> bool:
    """
    Check if GPU is Pascal generation or older (compute capability < 7.0).

    Pascal GPUs (compute 6.x) have limited FP16 performance without Tensor Cores.

    Args:
        device: PyTorch device

    Returns:
        True if Pascal or older, False otherwise
    """
    capability = get_gpu_compute_capability(device)
    if capability is None:
        return False

    major, _ = capability
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

    # Volta and newer (compute >= 7.0) - have Tensor Cores
    if prefer_fp16:
        return torch.float16
    else:
        return torch.float32


def warn_suboptimal_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """
    Warn if using a potentially suboptimal dtype for the device.

    Args:
        device: PyTorch device
        dtype: Data type being used
    """
    if device.type != "cuda":
        return

    capability = get_gpu_compute_capability(device)
    if capability is None:
        return

    major, minor = capability
    props = torch.cuda.get_device_properties(device)

    # Warn if using FP16 on Pascal or older
    if major < 7 and dtype in (torch.float16, torch.bfloat16):
        warnings.warn(
            f"Using {dtype} on {props.name} (compute {major}.{minor}) may result in "
            f"significantly reduced performance. Pascal GPUs can be 5-10x slower with FP16. "
            f"Consider using dtype=torch.float32 for optimal performance.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Inform if not using FP16 on Volta+ (just info, not warning)
    elif major >= 7 and dtype == torch.float32 and dtype != torch.bfloat16:
        # Only inform, don't warn - FP32 is still valid choice
        pass  # Could add logging here if desired
