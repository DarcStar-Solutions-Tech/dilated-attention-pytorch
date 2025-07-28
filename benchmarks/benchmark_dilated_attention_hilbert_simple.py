#!/usr/bin/env python3
"""
Simple benchmark for dilated attention with Hilbert Triton kernel.
"""

import argparse
import gc
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Import implementations
from dilated_attention_pytorch import DilatedAttention, MultiheadDilatedAttention
from dilated_attention_pytorch.kernels import HilbertAttentionCore


def get_device_info():
    """Get device information."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "CPU"}

    device = torch.cuda.current_device()
    return {
        "device": f"cuda:{device}",
        "name": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
    }


class Timer:
    """Simple timer for benchmarking."""

    def __init__(self, device="cuda"):
        self.device = device

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.elapsed = (time.perf_counter() - self.start) * 1000  # ms


def benchmark_attention(
    module: nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_warmup: int = 5,
    num_iterations: int = 20,
    test_backward: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Benchmark a single attention module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move module to device and set dtype
    module = module.to(device)
    if dtype == torch.float16:
        module = module.half()

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(x)
        if device == "cuda":
            torch.cuda.synchronize()

    # Measure forward pass
    forward_times = []
    for _ in range(num_iterations):
        with Timer(device) as timer:
            with torch.no_grad():
                output = module(x)
        forward_times.append(timer.elapsed)

    # Measure backward pass
    backward_times = []
    if test_backward:
        x.requires_grad = True
        for _ in range(num_iterations):
            # Forward
            output = module(x)
            loss = output.mean()

            # Backward
            with Timer(device) as timer:
                loss.backward()
            backward_times.append(timer.elapsed)

            # Clear gradients
            module.zero_grad()
            x.grad = None

    # Measure memory
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = module(x)
        forward_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

        if test_backward:
            torch.cuda.reset_peak_memory_stats()
            output = module(x)
            loss = output.mean()
            loss.backward()
            total_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        else:
            total_memory = forward_memory
    else:
        forward_memory = 0
        total_memory = 0

    return {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "backward_mean": np.mean(backward_times) if backward_times else 0,
        "backward_std": np.std(backward_times) if backward_times else 0,
        "forward_memory_mb": forward_memory,
        "total_memory_mb": total_memory,
        "throughput_tokens_per_sec": (batch_size * seq_len)
        / (np.mean(forward_times) / 1000),
    }


def create_attention_module(
    impl_type: str,
    hidden_dim: int,
    num_heads: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    seq_len: int,
) -> Optional[nn.Module]:
    """Create attention module based on type."""
    head_dim = hidden_dim // num_heads

    if impl_type == "standard":
        # Standard PyTorch attention
        return nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )

    elif impl_type == "dilated":
        # Check if sequence length is compatible
        if seq_len % max(segment_lengths) != 0:
            return None
        return DilatedAttention(
            dim=head_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            causal=False,
        )

    elif impl_type == "multihead_dilated":
        # Check if sequence length is compatible
        if seq_len % max(segment_lengths) != 0:
            return None
        return MultiheadDilatedAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            batch_first=True,
        )

    elif impl_type == "hilbert":
        # Hilbert attention with Triton kernel
        # Use single segment for Hilbert
        segment_size = min(segment_lengths[0], seq_len)
        dilation_rate = dilation_rates[0]

        return HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )

    else:
        raise ValueError(f"Unknown implementation type: {impl_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--seq-lengths", nargs="+", type=int, default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[768])
    parser.add_argument("--num-heads", nargs="+", type=int, default=[12])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    device_info = get_device_info()
    print(f"Running on: {device_info['name']}")

    # Configurations to test
    segment_configs = [
        ([512], [1]),
        ([512, 1024], [1, 2]),
        ([512, 1024, 2048], [1, 2, 4]),
        ([1024, 2048, 4096], [1, 2, 4]),
    ]

    implementations = ["standard", "dilated", "multihead_dilated", "hilbert"]

    results = []
    dtype = torch.float16 if args.fp16 else torch.float32

    print(f"\nBenchmarking with dtype: {dtype}")
    print("-" * 80)

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lengths:
            for hidden_dim in args.hidden_dims:
                for num_heads in args.num_heads:
                    print(
                        f"\nConfig: batch={batch_size}, seq={seq_len}, "
                        f"hidden={hidden_dim}, heads={num_heads}"
                    )

                    for impl in implementations:
                        for seg_lengths, dil_rates in segment_configs:
                            # Skip incompatible configurations
                            if impl == "standard" and (
                                seg_lengths != [512] or dil_rates != [1]
                            ):
                                continue

                            # Create module
                            try:
                                module = create_attention_module(
                                    impl,
                                    hidden_dim,
                                    num_heads,
                                    seg_lengths,
                                    dil_rates,
                                    seq_len,
                                )
                                if module is None:
                                    continue
                            except Exception as e:
                                print(f"  Failed to create {impl}: {e}")
                                continue

                            # Benchmark
                            try:
                                config_str = f"seg={seg_lengths}, dil={dil_rates}"
                                print(
                                    f"  Testing {impl} ({config_str})...",
                                    end="",
                                    flush=True,
                                )

                                metrics = benchmark_attention(
                                    module,
                                    batch_size,
                                    seq_len,
                                    hidden_dim,
                                    args.warmup,
                                    args.iterations,
                                    not args.no_backward,
                                    dtype,
                                )

                                print(
                                    f" Forward: {metrics['forward_mean']:.2f}ms, "
                                    f"Memory: {metrics['forward_memory_mb']:.1f}MB"
                                )

                                results.append(
                                    {
                                        "implementation": impl,
                                        "batch_size": batch_size,
                                        "seq_len": seq_len,
                                        "hidden_dim": hidden_dim,
                                        "num_heads": num_heads,
                                        "segment_lengths": seg_lengths,
                                        "dilation_rates": dil_rates,
                                        **metrics,
                                    }
                                )

                            except Exception as e:
                                print(f" Failed: {e}")

                            # Clean up
                            del module
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find best performing configurations
    for seq_len in args.seq_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        if not seq_results:
            continue

        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        # Sort by forward time
        seq_results.sort(key=lambda x: x["forward_mean"])

        # Show top 5
        for i, r in enumerate(seq_results[:5]):
            impl = r["implementation"]
            seg = r["segment_lengths"]
            dil = r["dilation_rates"]
            fwd = r["forward_mean"]
            mem = r["forward_memory_mb"]

            print(f"{i + 1}. {impl} (seg={seg}, dil={dil}): {fwd:.2f}ms, {mem:.1f}MB")

    # Calculate speedups vs standard attention
    print("\n" + "=" * 80)
    print("SPEEDUP vs STANDARD ATTENTION")
    print("=" * 80)

    for seq_len in args.seq_lengths:
        standard_results = [
            r
            for r in results
            if r["implementation"] == "standard" and r["seq_len"] == seq_len
        ]
        if not standard_results:
            continue

        standard_time = standard_results[0]["forward_mean"]

        print(f"\nSequence Length: {seq_len} (baseline: {standard_time:.2f}ms)")
        print("-" * 40)

        for impl in ["dilated", "multihead_dilated", "hilbert"]:
            impl_results = [
                r
                for r in results
                if r["implementation"] == impl and r["seq_len"] == seq_len
            ]

            for r in impl_results:
                speedup = standard_time / r["forward_mean"]
                seg = r["segment_lengths"]
                dil = r["dilation_rates"]

                print(f"{impl} (seg={seg}, dil={dil}): {speedup:.2f}x speedup")

    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dilated_attention_hilbert_benchmark_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(
                {"device_info": device_info, "args": vars(args), "results": results},
                f,
                indent=2,
            )
        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
