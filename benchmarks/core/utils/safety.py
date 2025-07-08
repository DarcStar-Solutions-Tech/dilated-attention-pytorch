"""
Safety utilities for benchmarking to prevent system lockups.

This module provides memory checks, progressive testing, and automatic cleanup
to prevent GPU memory exhaustion during benchmarks.
"""

import torch
import gc
import os
import psutil
from typing import Tuple, Optional, Dict, Any
import warnings


class SafetyConfig:
    """Configuration for safety limits."""

    def __init__(
        self,
        max_memory_fraction: float = 0.8,  # Max 80% of GPU memory
        min_free_memory_gb: float = 2.0,  # Keep at least 2GB free
        progressive_steps: int = 5,  # Number of progressive steps
        cleanup_threshold: float = 0.7,  # Cleanup if >70% memory used
        enable_cpu_check: bool = True,  # Also check CPU memory
        max_cpu_memory_fraction: float = 0.9,  # Max 90% of CPU memory
    ):
        self.max_memory_fraction = max_memory_fraction
        self.min_free_memory_gb = min_free_memory_gb
        self.progressive_steps = progressive_steps
        self.cleanup_threshold = cleanup_threshold
        self.enable_cpu_check = enable_cpu_check
        self.max_cpu_memory_fraction = max_cpu_memory_fraction


class MemorySafetyChecker:
    """Check memory availability before allocations."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_cuda = torch.cuda.is_available()

    def get_gpu_memory_info(self) -> Tuple[float, float, float]:
        """Get GPU memory info in GB (used, free, total)."""
        if not self.has_cuda:
            return 0.0, 0.0, 0.0

        # Force synchronization and cache clear
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Calculate free memory conservatively
        used = max(allocated, reserved)
        free = total - used

        return used, free, total

    def get_cpu_memory_info(self) -> Tuple[float, float, float]:
        """Get CPU memory info in GB (used, free, total)."""
        mem = psutil.virtual_memory()
        used = mem.used / 1e9
        free = mem.available / 1e9
        total = mem.total / 1e9
        return used, free, total

    def estimate_tensor_memory(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        num_tensors: int = 1,
    ) -> float:
        """Estimate memory required for tensors in GB."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        memory_gb = (num_elements * element_size * num_tensors) / 1e9

        # Add overhead for gradients if training
        if torch.is_grad_enabled():
            memory_gb *= 2  # Conservative estimate for gradient storage

        return memory_gb

    def check_memory_available(
        self, required_gb: float, device: Optional[torch.device] = None
    ) -> Tuple[bool, str]:
        """Check if required memory is available."""
        device = device or self.device

        if device.type == "cuda" and self.has_cuda:
            used, free, total = self.get_gpu_memory_info()

            # Check absolute free memory
            if free < self.config.min_free_memory_gb:
                return (
                    False,
                    f"Insufficient GPU memory: {free:.2f}GB free, need {self.config.min_free_memory_gb:.2f}GB minimum",
                )

            # Check if allocation would exceed fraction limit
            new_used = used + required_gb
            if new_used / total > self.config.max_memory_fraction:
                return (
                    False,
                    f"Would exceed GPU memory limit: {new_used / total:.1%} > {self.config.max_memory_fraction:.1%}",
                )

            # Check if we have enough free memory
            if required_gb > free:
                return (
                    False,
                    f"Insufficient GPU memory: need {required_gb:.2f}GB, have {free:.2f}GB free",
                )

        # Also check CPU memory if enabled
        if self.config.enable_cpu_check:
            cpu_used, cpu_free, cpu_total = self.get_cpu_memory_info()
            cpu_new_used = cpu_used + required_gb

            if cpu_new_used / cpu_total > self.config.max_cpu_memory_fraction:
                return (
                    False,
                    f"Would exceed CPU memory limit: {cpu_new_used / cpu_total:.1%} > {self.config.max_cpu_memory_fraction:.1%}",
                )

        return True, "Memory check passed"

    def cleanup_if_needed(self) -> bool:
        """Cleanup memory if usage exceeds threshold."""
        if self.has_cuda:
            used, free, total = self.get_gpu_memory_info()
            usage_fraction = used / total

            if usage_fraction > self.config.cleanup_threshold:
                self.force_cleanup()
                return True
        return False

    def force_cleanup(self):
        """Force memory cleanup."""
        # Clear Python garbage
        gc.collect()

        # Clear PyTorch cache
        if self.has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear any cached allocations
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()


class ProgressiveTester:
    """Test with progressively larger inputs to find limits safely."""

    def __init__(
        self,
        safety_checker: Optional[MemorySafetyChecker] = None,
        config: Optional[SafetyConfig] = None,
    ):
        self.safety_checker = safety_checker or MemorySafetyChecker(config)
        self.config = self.safety_checker.config

    def generate_progressive_sizes(
        self,
        target_size: int,
        min_size: Optional[int] = None,
        scale_factor: float = 2.0,
    ) -> list:
        """Generate progressive sizes leading up to target."""
        if min_size is None:
            min_size = min(1024, target_size // 16)

        sizes = []
        current = min_size

        while current < target_size:
            sizes.append(current)
            current = int(current * scale_factor)

        sizes.append(target_size)
        return sizes

    def test_with_safety(
        self,
        test_fn,
        test_params: Dict[str, Any],
        size_param_name: str = "seq_len",
        target_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        estimate_memory_fn=None,
    ):
        """Run test with progressive sizes and safety checks."""
        if target_size is None:
            target_size = test_params.get(size_param_name, 8192)

        sizes = self.generate_progressive_sizes(target_size)
        successful_size = None

        for size in sizes:
            # Update test parameters
            current_params = test_params.copy()
            current_params[size_param_name] = size

            # Estimate memory requirement
            if estimate_memory_fn:
                required_gb = estimate_memory_fn(current_params)
            else:
                # Default estimation for attention
                batch_size = current_params.get("batch_size", 1)
                num_heads = current_params.get("num_heads", 8)
                head_dim = current_params.get("head_dim", 64)

                # Q, K, V tensors
                shape = (batch_size, size, num_heads, head_dim)
                required_gb = self.safety_checker.estimate_tensor_memory(
                    shape, dtype, num_tensors=3
                )

                # Add overhead for attention computation
                required_gb *= 1.5

            # Check memory availability
            can_run, message = self.safety_checker.check_memory_available(required_gb)

            if not can_run:
                print(f"Skipping size {size}: {message}")
                break

            try:
                # Run cleanup if needed
                self.safety_checker.cleanup_if_needed()

                # Run the test
                print(f"Testing with size {size}...")
                result = test_fn(**current_params)
                successful_size = size

                # If this is not the target size, continue
                if size < target_size:
                    print(f"Success at size {size}, continuing...")
                else:
                    print(f"Successfully completed target size {size}")
                    return result

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"Failed at size {size}: {e}")
                self.safety_checker.force_cleanup()
                break
            except Exception as e:
                print(f"Unexpected error at size {size}: {e}")
                self.safety_checker.force_cleanup()
                raise

        # Return result from largest successful size
        if successful_size is not None:
            warnings.warn(
                f"Could not reach target size {target_size}, "
                f"stopped at {successful_size}"
            )
            # Re-run with successful size
            current_params = test_params.copy()
            current_params[size_param_name] = successful_size
            return test_fn(**current_params)
        else:
            raise RuntimeError("Could not run test at any size")


class SafeBenchmarkRunner:
    """Safe benchmark runner with automatic limits."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.safety_checker = MemorySafetyChecker(self.config)
        self.progressive_tester = ProgressiveTester(self.safety_checker)

    def run_benchmark(
        self,
        benchmark_fn,
        configs: list,
        estimate_memory_fn=None,
        size_param_name: str = "seq_len",
    ):
        """Run benchmarks with safety checks."""
        results = []

        for i, config in enumerate(configs):
            print(f"\n=== Benchmark {i + 1}/{len(configs)} ===")
            print(f"Config: {config}")

            # Check current memory state
            if self.safety_checker.has_cuda:
                used, free, total = self.safety_checker.get_gpu_memory_info()
                print(
                    f"GPU Memory: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total"
                )

            try:
                # Run with progressive testing
                result = self.progressive_tester.test_with_safety(
                    benchmark_fn,
                    config,
                    size_param_name=size_param_name,
                    estimate_memory_fn=estimate_memory_fn,
                )
                results.append(result)

            except Exception as e:
                print(f"Benchmark failed: {e}")
                results.append(None)

            # Always cleanup after each benchmark
            self.safety_checker.force_cleanup()

        return results


# Convenience functions
def check_memory_before_allocation(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    num_tensors: int = 1,
    device: Optional[torch.device] = None,
) -> bool:
    """Quick check if allocation is safe."""
    checker = MemorySafetyChecker()
    required_gb = checker.estimate_tensor_memory(shape, dtype, num_tensors)
    can_allocate, message = checker.check_memory_available(required_gb, device)

    if not can_allocate:
        print(f"Warning: {message}")

    return can_allocate


def run_with_memory_limit(fn, *args, max_memory_gb: float = None, **kwargs):
    """Run function with memory limit."""
    original_limit = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    if max_memory_gb is not None:
        # Set PyTorch memory limit
        max_split_size_mb = int(max_memory_gb * 1024)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size_mb}"

    try:
        return fn(*args, **kwargs)
    finally:
        # Restore original setting
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original_limit
