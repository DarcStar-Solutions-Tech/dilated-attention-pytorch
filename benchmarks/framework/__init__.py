"""Benchmark framework for dilated attention implementations."""

from .base import BaseBenchmark, BenchmarkResult
from .config import BenchmarkConfig, BenchmarkSuite
from .utils import (
    setup_device_and_dtype,
    reset_gpu_memory,
    get_peak_memory_mb,
    create_attention_inputs,
    measure_with_warmup,
    format_results_table,
    save_results,
)

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkSuite",
    "setup_device_and_dtype",
    "reset_gpu_memory",
    "get_peak_memory_mb",
    "create_attention_inputs",
    "measure_with_warmup",
    "format_results_table",
    "save_results",
]
