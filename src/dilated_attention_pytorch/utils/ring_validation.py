"""Validation utilities for ring attention setup and configuration."""

import torch
import torch.distributed as dist
from typing import List
import warnings


def validate_ring_setup(world_size: int, rank: int) -> None:
    """Validate ring attention setup.

    Args:
        world_size: Total number of processes
        rank: Current process rank

    Raises:
        ValueError: If setup is invalid
    """
    if world_size < 1:
        raise ValueError(f"World size must be positive, got {world_size}")

    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Rank {rank} is invalid for world size {world_size}. "
            f"Must be in range [0, {world_size - 1}]"
        )

    if world_size == 1:
        warnings.warn(
            "Ring attention running with world_size=1. "
            "This provides no memory benefit over standard attention."
        )


def validate_sequence_split(
    seq_len: int, world_size: int, segment_lengths: List[int]
) -> None:
    """Validate that sequence can be properly split for ring attention.

    Args:
        seq_len: Total sequence length
        world_size: Number of processes
        segment_lengths: Configured segment lengths

    Raises:
        ValueError: If sequence cannot be properly split
    """
    # Check divisibility by world size
    if seq_len % world_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by world size {world_size}"
        )

    # Check divisibility by largest segment
    max_segment = max(segment_lengths)
    if seq_len % max_segment != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by largest segment {max_segment}"
        )

    # Check that local sequence is large enough
    local_seq_len = seq_len // world_size
    if local_seq_len < max_segment:
        warnings.warn(
            f"Local sequence length {local_seq_len} is smaller than largest segment {max_segment}. "
            f"This may lead to inefficient computation."
        )


def validate_ring_communication() -> bool:
    """Validate that ring communication is working properly.

    Returns:
        True if communication test passes
    """
    if not dist.is_available() or not dist.is_initialized():
        return False

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create test tensor
        test_tensor = torch.tensor([rank], dtype=torch.float32)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()

        # Ring pass test
        src = (rank - 1) % world_size
        dst = (rank + 1) % world_size

        recv_tensor = torch.empty_like(test_tensor)

        # Non-blocking send/recv
        send_op = dist.isend(test_tensor, dst=dst)
        recv_op = dist.irecv(recv_tensor, src=src)

        send_op.wait()
        recv_op.wait()

        # Verify received correct value
        expected = (rank - 1) % world_size
        if recv_tensor.item() != expected:
            warnings.warn(
                f"Ring communication test failed. "
                f"Rank {rank} expected {expected}, got {recv_tensor.item()}"
            )
            return False

        # Synchronize all ranks
        dist.barrier()

        return True

    except Exception as e:
        warnings.warn(f"Ring communication validation failed: {e}")
        return False


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    world_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Estimate memory usage for ring attention.

    Args:
        batch_size: Batch size
        seq_len: Total sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        world_size: Number of processes for ring
        dtype: Data type

    Returns:
        Dictionary with memory estimates in MB
    """
    # Get bytes per element
    if dtype in [torch.float32, torch.int32]:
        bytes_per_elem = 4
    elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
        bytes_per_elem = 2
    elif dtype in [torch.float64, torch.int64]:
        bytes_per_elem = 8
    else:
        bytes_per_elem = 4  # Default

    # Local sequence length
    local_seq_len = seq_len // world_size

    # QKV memory (per process)
    qkv_memory = 3 * batch_size * local_seq_len * num_heads * head_dim * bytes_per_elem

    # Attention scores memory (per process)
    # Only need to store scores for local_seq_len queries
    scores_memory = (
        batch_size * num_heads * local_seq_len * local_seq_len * bytes_per_elem
    )

    # Communication buffers (K and V)
    comm_memory = 2 * batch_size * local_seq_len * num_heads * head_dim * bytes_per_elem

    # Output memory
    output_memory = batch_size * local_seq_len * num_heads * head_dim * bytes_per_elem

    # Total per process
    total_per_process = qkv_memory + scores_memory + comm_memory + output_memory

    # Standard attention comparison (no ring)
    std_qkv = 3 * batch_size * seq_len * num_heads * head_dim * bytes_per_elem
    std_scores = batch_size * num_heads * seq_len * seq_len * bytes_per_elem
    std_output = batch_size * seq_len * num_heads * head_dim * bytes_per_elem
    std_total = std_qkv + std_scores + std_output

    return {
        "ring_qkv_mb": qkv_memory / (1024**2),
        "ring_scores_mb": scores_memory / (1024**2),
        "ring_comm_mb": comm_memory / (1024**2),
        "ring_output_mb": output_memory / (1024**2),
        "ring_total_mb": total_per_process / (1024**2),
        "standard_total_mb": std_total / (1024**2),
        "memory_reduction": 1 - (total_per_process / std_total),
        "scaling_factor": world_size,
    }


def check_hardware_compatibility() -> dict:
    """Check hardware compatibility for ring attention.

    Returns:
        Dictionary with hardware information and recommendations
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "distributed_available": torch.distributed.is_available()
        if hasattr(torch, "distributed")
        else False,
        "recommended_backend": None,
        "warnings": [],
    }

    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(info["gpu_count"])
        ]

        # Check NCCL availability
        try:
            import torch.distributed as dist

            info["nccl_available"] = dist.is_nccl_available()
            info["recommended_backend"] = "nccl" if info["nccl_available"] else "gloo"
        except Exception:
            info["nccl_available"] = False
            info["recommended_backend"] = "gloo"

        # Check compute capability
        for i in range(info["gpu_count"]):
            major, minor = torch.cuda.get_device_capability(i)
            if major < 7:
                info["warnings"].append(
                    f"GPU {i} has compute capability {major}.{minor}. "
                    f"Ring attention works best with compute capability 7.0+"
                )

    else:
        info["gpu_count"] = 0
        info["recommended_backend"] = "gloo"
        info["warnings"].append(
            "No CUDA GPUs available. Ring attention will run on CPU with limited performance."
        )

    return info


def suggest_optimal_ring_size(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    available_gpus: int,
    target_memory_gb: float = 16.0,
) -> int:
    """Suggest optimal ring size based on sequence length and memory constraints.

    Args:
        seq_len: Total sequence length
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Dimension per head
        available_gpus: Number of available GPUs
        target_memory_gb: Target memory usage per GPU in GB

    Returns:
        Suggested ring size
    """
    # Start with all available GPUs
    best_ring_size = available_gpus

    # Check memory usage for different ring sizes
    for ring_size in range(available_gpus, 0, -1):
        # Skip if sequence not divisible
        if seq_len % ring_size != 0:
            continue

        # Estimate memory
        mem_info = estimate_memory_usage(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            world_size=ring_size,
            dtype=torch.float16,
        )

        # Check if within target
        if mem_info["ring_total_mb"] / 1024 <= target_memory_gb:
            best_ring_size = ring_size
            break

    return best_ring_size
