"""Base classes for benchmarks."""

import gc
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        """Initialize base benchmark.

        Args:
            device: Device to run benchmarks on
            dtype: Data type for tensors
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or torch.float32
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def measure_memory(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure peak memory usage of a function.

        Args:
            func: Function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (function result, peak memory in MB)
        """
        self.cleanup_memory()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        result = func(*args, **kwargs)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            peak_mb = 0.0

        return result, peak_mb

    def time_operation(
        self,
        func: Callable,
        *args,
        warmup: Optional[int] = None,
        iterations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Time an operation with warmup.

        Args:
            func: Function to time
            *args: Positional arguments for func
            warmup: Number of warmup iterations (default: self.warmup_iterations)
            iterations: Number of timing iterations (default: self.benchmark_iterations)
            **kwargs: Keyword arguments for func

        Returns:
            Dict with timing statistics (mean, std, min, max in seconds)
        """
        warmup = warmup or self.warmup_iterations
        iterations = iterations or self.benchmark_iterations

        # Warmup
        for _ in range(warmup):
            _ = func(*args, **kwargs)

        # Time iterations
        times = []
        for _ in range(iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            start = time.perf_counter()
            _ = func(*args, **kwargs)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            end = time.perf_counter()
            times.append(end - start)

        times_tensor = torch.tensor(times)
        return {
            "mean": times_tensor.mean().item(),
            "std": times_tensor.std().item(),
            "min": times_tensor.min().item(),
            "max": times_tensor.max().item(),
            "median": times_tensor.median().item(),
        }

    def generate_test_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate standard QKV test data.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: Data type (default: self.dtype)
            device: Device (default: self.device)

        Returns:
            Tuple of (query, key, value) tensors
        """
        dtype = dtype or self.dtype
        device = device or self.device

        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(*shape, dtype=dtype, device=device)
        k = torch.randn(*shape, dtype=dtype, device=device)
        v = torch.randn(*shape, dtype=dtype, device=device)

        return q, k, v

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        pass


class BaseDistributedBenchmark(BaseBenchmark):
    """Base class for distributed benchmarks."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        port: str = "12356",
        **kwargs,
    ):
        """Initialize distributed benchmark.

        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Distributed backend
            port: Communication port
            **kwargs: Additional arguments for BaseBenchmark
        """
        # Set device based on rank
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        super().__init__(device=device, **kwargs)

        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.port = port

    def setup_distributed(self) -> None:
        """Initialize distributed environment."""
        import os

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.port

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend, rank=self.rank, world_size=self.world_size
            )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.rank)

    def cleanup_distributed(self) -> None:
        """Clean up distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    def gather_results(self, local_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather results from all ranks.

        Args:
            local_result: Local result dictionary

        Returns:
            List of results from all ranks (only on rank 0)
        """
        if not dist.is_initialized():
            return [local_result]

        # Convert to tensor for gathering
        import pickle

        serialized = pickle.dumps(local_result)
        size = torch.tensor(len(serialized), device=self.device)

        # Gather sizes
        size_list = [torch.zeros_like(size) for _ in range(self.world_size)]
        dist.all_gather(size_list, size)

        # Prepare receive buffers
        max_size = max(s.item() for s in size_list)
        tensor = torch.zeros(max_size, dtype=torch.uint8, device=self.device)
        tensor[: len(serialized)] = torch.tensor(
            list(serialized), dtype=torch.uint8, device=self.device
        )

        # Gather data
        if self.rank == 0:
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gathered, dst=0)

            # Deserialize
            results = []
            for i, (t, s) in enumerate(zip(gathered, size_list)):
                data = bytes(t[: s.item()].cpu().numpy())
                results.append(pickle.loads(data))
            return results
        else:
            dist.gather(tensor, dst=0)
            return []
