#!/usr/bin/env python3
"""
Benchmark Hilbert-optimized dilated attention implementation.
Tests the integration of Hilbert Triton kernels with dilated attention patterns.
"""

import gc
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedAttentionWithHilbert(nn.Module):
    """Dilated attention implementation that uses Hilbert kernel."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        use_hilbert: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.use_hilbert = use_hilbert
        self.use_triton = use_triton

        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Import Hilbert kernel if using Triton
        if use_triton:
            try:
                from dilated_attention_pytorch.kernels import HilbertAttentionCore

                self.hilbert_kernel = HilbertAttentionCore(
                    hidden_dim=embed_dim,
                    num_heads=num_heads,
                    segment_size=segment_lengths[0],
                    dilation_rate=dilation_rates[0],
                )
            except ImportError:
                print("Failed to import HilbertAttentionCore, falling back to PyTorch")
                self.use_triton = False
                self.hilbert_kernel = None
        else:
            self.hilbert_kernel = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dilated attention pattern."""
        batch_size, seq_len, _ = x.shape

        # Project to QKV
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Process each segment with appropriate dilation
        outputs = []
        for seg_len, dil_rate in zip(self.segment_lengths, self.dilation_rates):
            if seq_len < seg_len:
                continue

            # Process segments
            for start in range(0, seq_len - seg_len + 1, seg_len):
                end = start + seg_len

                if (
                    self.use_triton
                    and self.hilbert_kernel is not None
                    and seg_len == self.segment_lengths[0]
                ):
                    # Use Hilbert kernel for first segment size
                    seg_input = x[:, start:end, :]
                    seg_output = self.hilbert_kernel(
                        seg_input, use_hilbert=self.use_hilbert
                    )
                    outputs.append(seg_output)
                else:
                    # Fallback to PyTorch implementation
                    q_seg = q[:, :, start:end:dil_rate, :]
                    k_seg = k[:, :, start:end:dil_rate, :]
                    v_seg = v[:, :, start:end:dil_rate, :]

                    scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) / (
                        self.head_dim**0.5
                    )
                    attn_weights = F.softmax(scores, dim=-1)
                    seg_output = torch.matmul(attn_weights, v_seg)

                    # Reshape back
                    seg_output = seg_output.transpose(1, 2).contiguous()
                    seg_output = seg_output.view(batch_size, -1, self.embed_dim)

                    # Upsample if dilated
                    if dil_rate > 1:
                        seg_output = seg_output.repeat_interleave(dil_rate, dim=1)

                    outputs.append(seg_output[:, :seg_len, :])

        # Combine outputs
        if outputs:
            output = torch.cat(outputs, dim=1)[:, :seq_len, :]
        else:
            output = torch.zeros_like(x)

        # Output projection
        output = self.out_proj(output)
        return output


def benchmark_configuration(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    embed_dim: int,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark a specific configuration."""
    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    # Time forward pass
    forward_times = []
    for _ in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model(x)

        if device == "cuda":
            torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    # Time backward pass
    x.requires_grad = True
    model.zero_grad()

    backward_times = []
    for _ in range(num_iterations):
        # Forward
        output = model(x)
        loss = output.mean()

        # Time backward
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        if device == "cuda":
            torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)

        model.zero_grad()

    # Memory usage
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(x)
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
    else:
        peak_memory = 0

    return {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "backward_mean": np.mean(backward_times),
        "backward_std": np.std(backward_times),
        "memory_mb": peak_memory,
        "throughput_tokens_sec": (batch_size * seq_len)
        / (np.mean(forward_times) / 1000),
    }


def main():
    print("=" * 80)
    print("Hilbert-Optimized Dilated Attention Benchmark")
    print("=" * 80)

    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("Running on CPU")

    # Test configurations
    embed_dim = 768
    num_heads = 12
    batch_size = 2

    # Different segment configurations
    configs = [
        # (name, segment_lengths, dilation_rates)
        ("Standard (seg=512)", [512], [1]),
        ("Dilated (seg=512, dil=2)", [512], [2]),
        ("Multi-scale (seg=[512,1024], dil=[1,2])", [512, 1024], [1, 2]),
        ("Large segments (seg=1024)", [1024], [1]),
        ("Large dilated (seg=1024, dil=2)", [1024], [2]),
    ]

    sequence_lengths = [1024, 2048, 4096, 8192]

    results = []

    print(
        f"\nConfiguration: embed_dim={embed_dim}, num_heads={num_heads}, batch_size={batch_size}"
    )
    print("-" * 80)

    for seq_len in sequence_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        seq_results = []

        for name, seg_lengths, dil_rates in configs:
            # Skip if incompatible
            if seq_len % max(seg_lengths) != 0:
                print(f"  {name}: SKIPPED (incompatible sequence length)")
                continue

            # Test with and without Hilbert
            for use_hilbert in [False, True]:
                for use_triton in [False, True]:
                    if not use_triton and use_hilbert:
                        continue  # Skip invalid combination

                    variant = f"{'Triton+' if use_triton else ''}{'Hilbert' if use_hilbert else 'Standard'}"
                    print(f"  {name} ({variant})...", end="", flush=True)

                    try:
                        # Create model
                        model = DilatedAttentionWithHilbert(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=seg_lengths,
                            dilation_rates=dil_rates,
                            use_hilbert=use_hilbert,
                            use_triton=use_triton,
                        ).to(device)

                        # Benchmark
                        metrics = benchmark_configuration(
                            model,
                            seq_len,
                            batch_size,
                            embed_dim,
                            num_warmup=5,
                            num_iterations=20,
                            device=device,
                        )

                        print(
                            f" {metrics['forward_mean']:.2f}ms "
                            f"(±{metrics['forward_std']:.2f}ms), "
                            f"{metrics['memory_mb']:.1f}MB"
                        )

                        result = {
                            "seq_len": seq_len,
                            "config_name": name,
                            "segment_lengths": seg_lengths,
                            "dilation_rates": dil_rates,
                            "use_hilbert": use_hilbert,
                            "use_triton": use_triton,
                            "variant": variant,
                            **metrics,
                        }
                        seq_results.append(result)
                        results.append(result)

                    except Exception as e:
                        print(f" FAILED: {e}")

                    # Cleanup
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

        # Show best for this sequence length
        if seq_results:
            seq_results.sort(key=lambda x: x["forward_mean"])
            best = seq_results[0]
            print(
                f"\n  Best: {best['config_name']} ({best['variant']}) - {best['forward_mean']:.2f}ms"
            )

    # Summary analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Compare Hilbert vs non-Hilbert
    print("\nHilbert Optimization Impact:")
    print("-" * 40)

    for seq_len in sequence_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]

        # Find matching pairs
        for config_name in set(r["config_name"] for r in seq_results):
            triton_standard = next(
                (
                    r
                    for r in seq_results
                    if r["config_name"] == config_name
                    and r["use_triton"]
                    and not r["use_hilbert"]
                ),
                None,
            )
            triton_hilbert = next(
                (
                    r
                    for r in seq_results
                    if r["config_name"] == config_name
                    and r["use_triton"]
                    and r["use_hilbert"]
                ),
                None,
            )

            if triton_standard and triton_hilbert:
                speedup = (
                    triton_standard["forward_mean"] / triton_hilbert["forward_mean"]
                )
                print(
                    f"  seq={seq_len}, {config_name}: {speedup:.2f}x speedup with Hilbert"
                )

    # Memory efficiency
    print("\nMemory Efficiency:")
    print("-" * 40)

    for seq_len in sequence_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        if seq_results:
            min_mem = min(r["memory_mb"] for r in seq_results)
            max_mem = max(r["memory_mb"] for r in seq_results)
            print(f"  seq={seq_len}: {min_mem:.1f}MB - {max_mem:.1f}MB")

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Performance vs sequence length
    plt.subplot(2, 2, 1)
    for config_name in set(r["config_name"] for r in results):
        for variant in ["Standard", "Triton+Standard", "Triton+Hilbert"]:
            data = [
                (r["seq_len"], r["forward_mean"])
                for r in results
                if r["config_name"] == config_name and r["variant"] == variant
            ]
            if data:
                x, y = zip(*data)
                plt.plot(x, y, marker="o", label=f"{config_name} ({variant})")

    plt.xlabel("Sequence Length")
    plt.ylabel("Forward Pass Time (ms)")
    plt.title("Performance vs Sequence Length")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Plot 2: Memory usage
    plt.subplot(2, 2, 2)
    seq_lens = sorted(set(r["seq_len"] for r in results))
    memory_data = []
    for seq_len in seq_lens:
        seq_results = [r["memory_mb"] for r in results if r["seq_len"] == seq_len]
        if seq_results:
            memory_data.append((seq_len, np.mean(seq_results)))

    if memory_data:
        x, y = zip(*memory_data)
        plt.plot(x, y, "bo-", linewidth=2, markersize=8)
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Scaling")
        plt.grid(True, alpha=0.3)

    # Plot 3: Speedup from Hilbert
    plt.subplot(2, 2, 3)
    speedup_data = []
    for seq_len in sequence_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        for config_name in set(r["config_name"] for r in seq_results):
            standard = next(
                (
                    r
                    for r in seq_results
                    if r["config_name"] == config_name
                    and r["use_triton"]
                    and not r["use_hilbert"]
                ),
                None,
            )
            hilbert = next(
                (
                    r
                    for r in seq_results
                    if r["config_name"] == config_name
                    and r["use_triton"]
                    and r["use_hilbert"]
                ),
                None,
            )

            if standard and hilbert:
                speedup = standard["forward_mean"] / hilbert["forward_mean"]
                speedup_data.append((seq_len, speedup, config_name))

    if speedup_data:
        for config_name in set(d[2] for d in speedup_data):
            data = [(d[0], d[1]) for d in speedup_data if d[2] == config_name]
            if data:
                x, y = zip(*data)
                plt.plot(x, y, marker="o", label=config_name)

        plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("Sequence Length")
        plt.ylabel("Speedup Factor")
        plt.title("Hilbert Optimization Speedup")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 4: Throughput
    plt.subplot(2, 2, 4)
    for variant in ["Standard", "Triton+Standard", "Triton+Hilbert"]:
        data = [
            (r["seq_len"], r["throughput_tokens_sec"])
            for r in results
            if r["variant"] == variant
        ]
        if data:
            x, y = zip(*data)
            plt.plot(x, y, marker="o", label=variant)

    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Processing Throughput")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hilbert_dilated_attention_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to: hilbert_dilated_attention_benchmark.png")


if __name__ == "__main__":
    main()
