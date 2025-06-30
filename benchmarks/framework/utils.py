"""Utility functions for benchmark framework."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tabulate import tabulate


def setup_device_and_dtype(
    device_str: str = "cuda", use_fp16: bool = True
) -> Tuple[torch.device, torch.dtype]:
    """Setup device and data type for benchmarking.

    Args:
        device_str: Device string ("cuda" or "cpu")
        use_fp16: Whether to use float16 on CUDA

    Returns:
        Tuple of (device, dtype)
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"

    device = torch.device(device_str)

    # Select appropriate dtype
    if device.type == "cuda":
        if use_fp16:
            # Check if GPU supports float16 efficiently
            if torch.cuda.get_device_capability()[0] >= 7:
                dtype = torch.float16
            else:
                print("GPU doesn't support efficient float16, using float32")
                dtype = torch.float32
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    return device, dtype


def reset_gpu_memory(device: torch.device):
    """Reset GPU memory statistics and clear cache.

    Args:
        device: The torch device
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory_mb(device: torch.device) -> float:
    """Get peak memory usage in MB.

    Args:
        device: The torch device

    Returns:
        Peak memory usage in MB
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def get_current_memory_mb(device: torch.device) -> float:
    """Get current memory usage in MB.

    Args:
        device: The torch device

    Returns:
        Current memory usage in MB
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def create_attention_inputs(
    batch_size: int,
    seq_length: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    is_multihead: bool = False,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input tensors for attention benchmarks.

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to create tensors on
        dtype: Data type for tensors
        is_multihead: If True, create (batch, seq, embed_dim) tensors
        requires_grad: If True, enable gradients

    Returns:
        Tuple of (query, key, value) tensors
    """
    if is_multihead:
        # For nn.MultiheadAttention style modules
        shape = (batch_size, seq_length, num_heads * head_dim)
    else:
        # For core attention modules
        shape = (batch_size, seq_length, num_heads, head_dim)

    # Create tensors with small values to avoid numerical issues
    q = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        * 0.02
    )
    k = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        * 0.02
    )
    v = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        * 0.02
    )

    return q, k, v


def create_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask.

    Args:
        seq_length: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask tensor
    """
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    return mask.bool()


def measure_with_warmup(
    func: Callable,
    warmup_steps: int = 3,
    measure_steps: int = 10,
    sync_cuda: bool = True,
) -> Tuple[float, float]:
    """Measure function execution time with warmup.

    Args:
        func: Function to measure
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations
        sync_cuda: Whether to synchronize CUDA

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup_steps):
        func()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(measure_steps):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        func()

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    times_ms = [t * 1000 for t in times]
    mean_time = sum(times_ms) / len(times_ms)

    # Calculate standard deviation
    if len(times_ms) > 1:
        variance = sum((t - mean_time) ** 2 for t in times_ms) / (len(times_ms) - 1)
        std_time = variance**0.5
    else:
        std_time = 0.0

    return mean_time, std_time


def profile_memory_usage(func: Callable, device: torch.device) -> Dict[str, float]:
    """Profile memory usage of a function.

    Args:
        func: Function to profile
        device: Device to profile on

    Returns:
        Dictionary with memory statistics
    """
    reset_gpu_memory(device)

    # Get baseline memory
    baseline_memory = get_current_memory_mb(device)

    # Run function
    func()

    # Get peak memory
    peak_memory = get_peak_memory_mb(device)
    current_memory = get_current_memory_mb(device)

    return {
        "baseline_mb": baseline_memory,
        "peak_mb": peak_memory,
        "current_mb": current_memory,
        "allocated_mb": peak_memory - baseline_memory,
    }


def format_results_table(
    results: List[Dict[str, Any]], headers: Optional[List[str]] = None
) -> str:
    """Format benchmark results as a table.

    Args:
        results: List of result dictionaries
        headers: Optional list of headers to include

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display"

    # Auto-detect headers if not provided
    if headers is None:
        headers = list(results[0].keys())

    # Extract rows
    rows = []
    for result in results:
        row = [result.get(h, "N/A") for h in headers]
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid")


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    prefix: str,
    save_json: bool = True,
    save_csv: bool = True,
    save_plot: bool = True,
) -> Dict[str, Path]:
    """Save benchmark results in multiple formats.

    Args:
        results: List of result dictionaries
        output_dir: Output directory
        prefix: Filename prefix
        save_json: Whether to save JSON
        save_csv: Whether to save CSV
        save_plot: Whether to save plot

    Returns:
        Dictionary of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # Save JSON
    if save_json:
        json_path = output_dir / f"{prefix}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        saved_files["json"] = json_path

    # Save CSV
    if save_csv and results:
        csv_path = output_dir / f"{prefix}_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        saved_files["csv"] = csv_path

    # Save plot
    if save_plot and results:
        plot_path = output_dir / f"{prefix}_{timestamp}.png"
        create_performance_plot(results, str(plot_path))
        saved_files["plot"] = plot_path

    return saved_files


def create_performance_plot(
    results: List[Dict[str, Any]], output_path: str, metric: str = "throughput"
):
    """Create a performance comparison plot.

    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
        metric: Metric to plot (throughput, time_ms, memory_mb)
    """
    if not results:
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Filter successful results
    df = df[df.get("success", True)]

    if df.empty:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by implementation
    implementations = df["implementation"].unique()

    # Plot bars for each sequence length
    seq_lengths = sorted(df["seq_length"].unique())
    x = range(len(seq_lengths))
    width = 0.8 / len(implementations)

    for i, impl in enumerate(implementations):
        impl_data = df[df["implementation"] == impl]
        values = []

        for seq_len in seq_lengths:
            seq_data = impl_data[impl_data["seq_length"] == seq_len]
            if not seq_data.empty:
                values.append(seq_data[metric].mean())
            else:
                values.append(0)

        offset = (i - len(implementations) / 2) * width + width / 2
        _ = ax.bar([xi + offset for xi in x], values, width, label=impl)

    # Customize plot
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Performance Comparison - {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compare_implementations(
    results: List[Dict[str, Any]], baseline: str = "standard"
) -> pd.DataFrame:
    """Compare implementations against a baseline.

    Args:
        results: List of result dictionaries
        baseline: Name of baseline implementation

    Returns:
        DataFrame with comparison metrics
    """
    df = pd.DataFrame(results)

    # Filter successful results
    df = df[df.get("success", True)]

    if df.empty or baseline not in df["implementation"].values:
        return pd.DataFrame()

    # Get baseline data
    baseline_df = df[df["implementation"] == baseline]

    # Calculate relative performance
    comparison_data = []

    for impl in df["implementation"].unique():
        if impl == baseline:
            continue

        impl_df = df[df["implementation"] == impl]

        # Match configurations
        for _, baseline_row in baseline_df.iterrows():
            config_match = impl_df[
                (impl_df["batch_size"] == baseline_row["batch_size"])
                & (impl_df["seq_length"] == baseline_row["seq_length"])
                & (impl_df["num_heads"] == baseline_row["num_heads"])
            ]

            if not config_match.empty:
                impl_row = config_match.iloc[0]

                comparison_data.append(
                    {
                        "implementation": impl,
                        "batch_size": baseline_row["batch_size"],
                        "seq_length": baseline_row["seq_length"],
                        "num_heads": baseline_row["num_heads"],
                        "speedup": baseline_row["time_ms"] / impl_row["time_ms"],
                        "memory_ratio": impl_row["memory_mb"]
                        / baseline_row["memory_mb"],
                        "throughput_ratio": impl_row["throughput"]
                        / baseline_row["throughput"],
                    }
                )

    return pd.DataFrame(comparison_data)
