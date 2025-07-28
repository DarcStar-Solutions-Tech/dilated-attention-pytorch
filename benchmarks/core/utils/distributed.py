"""Distributed computing utilities for benchmarks."""

import os
from typing import Callable, List, Any
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    port: str = "12356",
    master_addr: str = "localhost",
) -> None:
    """Initialize distributed environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend (nccl, gloo)
        port: Communication port
        master_addr: Master node address
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = port

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_distributed_test(
    test_func: Callable,
    world_size: int,
    *args,
    backend: str = "nccl",
    port: str = "12356",
    **kwargs,
) -> List[Any]:
    """Run a test function across multiple GPUs.

    Args:
        test_func: Function to run on each GPU. Should accept (rank, world_size, *args, **kwargs)
        world_size: Number of GPUs to use
        *args: Additional positional arguments for test_func
        backend: Distributed backend
        port: Communication port
        **kwargs: Additional keyword arguments for test_func

    Returns:
        List of results from each rank
    """

    def wrapped_test(rank: int):
        init_distributed(rank, world_size, backend, port)
        try:
            result = test_func(rank, world_size, *args, **kwargs)
            return result
        finally:
            cleanup_distributed()

    if world_size == 1:
        # Single GPU - run directly
        return [test_func(0, 1, *args, **kwargs)]
    else:
        # Multi-GPU - use multiprocessing
        mp.spawn(wrapped_test, args=(), nprocs=world_size, join=True)
        return []  # Results handled internally


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast an object from source rank to all ranks.

    Args:
        obj: Object to broadcast (only needed on source rank)
        src: Source rank

    Returns:
        Broadcasted object on all ranks
    """
    if not dist.is_initialized():
        return obj

    import pickle

    if get_rank() == src:
        data = pickle.dumps(obj)
        size = torch.tensor(len(data), dtype=torch.long)
    else:
        size = torch.tensor(0, dtype=torch.long)

    # Broadcast size
    dist.broadcast(size, src=src)

    # Prepare buffer
    if get_rank() != src:
        data = bytearray(size.item())

    # Broadcast data
    tensor = torch.tensor(list(data), dtype=torch.uint8)
    dist.broadcast(tensor, src=src)

    # Deserialize
    data = bytes(tensor.cpu().numpy())
    return pickle.loads(data)
