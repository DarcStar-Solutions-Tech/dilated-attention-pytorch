#!/usr/bin/env python3
"""
Visualize FP32 benchmark results for all implementations.
"""

import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import numpy as np
from datetime import datetime

# Load the results
with open("benchmarks/all_implementations_fp32_2025-07-06-2315-UTC.json", "r") as f:
    data = json.load(f)

# Extract data for plotting
implementations = [
    "DilatedAttention",
    "MultiheadDilatedAttention",
    "ImprovedMultiheadDilatedAttention",
    "Factory-Auto",
]
seq_lengths = []
throughput_by_impl = {impl: [] for impl in implementations}
memory_by_impl = {impl: [] for impl in implementations}

for benchmark in data["benchmarks"]:
    seq_len = benchmark["config"]["seq_len"]
    if seq_len not in seq_lengths and benchmark["config"]["num_heads"] == 8:
        seq_lengths.append(seq_len)

        for impl in implementations:
            if (
                impl in benchmark["results"]
                and "error" not in benchmark["results"][impl]
            ):
                result = benchmark["results"][impl]
                throughput_by_impl[impl].append(
                    result["throughput"]["tokens_per_second"]
                )
                memory_by_impl[impl].append(result["memory"]["peak_mb"])
            else:
                throughput_by_impl[impl].append(0)
                memory_by_impl[impl].append(0)

# Sort by sequence length
sorted_indices = np.argsort(seq_lengths)
seq_lengths = [seq_lengths[i] for i in sorted_indices]
for impl in implementations:
    throughput_by_impl[impl] = [throughput_by_impl[impl][i] for i in sorted_indices]
    memory_by_impl[impl] = [memory_by_impl[impl][i] for i in sorted_indices]

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Colors for each implementation
colors = {
    "DilatedAttention": "#1f77b4",
    "MultiheadDilatedAttention": "#ff7f0e",
    "ImprovedMultiheadDilatedAttention": "#2ca02c",
    "Factory-Auto": "#d62728",
}

# Plot 1: Throughput vs Sequence Length
for impl in implementations:
    if any(throughput_by_impl[impl]):  # Only plot if we have data
        ax1.plot(
            seq_lengths,
            throughput_by_impl[impl],
            "o-",
            label=impl,
            color=colors[impl],
            linewidth=2,
            markersize=8,
        )

ax1.set_xlabel("Sequence Length", fontsize=12)
ax1.set_ylabel("Throughput (tokens/second)", fontsize=12)
ax1.set_title("FP32 Throughput vs Sequence Length", fontsize=14, fontweight="bold")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best", fontsize=10)

# Plot 2: Memory Usage vs Sequence Length
for impl in implementations:
    if any(memory_by_impl[impl]):
        ax2.plot(
            seq_lengths,
            memory_by_impl[impl],
            "s-",
            label=impl,
            color=colors[impl],
            linewidth=2,
            markersize=8,
        )

ax2.set_xlabel("Sequence Length", fontsize=12)
ax2.set_ylabel("Peak Memory (MB)", fontsize=12)
ax2.set_title("FP32 Memory Usage vs Sequence Length", fontsize=14, fontweight="bold")
ax2.set_xscale("log", base=2)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best", fontsize=10)

# Plot 3: Speedup over Original
speedup_data = {}
for impl in implementations:
    if impl != "DilatedAttention":
        speedup_data[impl] = []
        for i in range(len(seq_lengths)):
            if (
                throughput_by_impl["DilatedAttention"][i] > 0
                and throughput_by_impl[impl][i] > 0
            ):
                speedup = (
                    throughput_by_impl[impl][i]
                    / throughput_by_impl["DilatedAttention"][i]
                )
                speedup_data[impl].append(speedup)
            else:
                speedup_data[impl].append(0)

x = np.arange(len(seq_lengths))
width = 0.25

for i, (impl, speedups) in enumerate(speedup_data.items()):
    ax3.bar(x + i * width, speedups, width, label=impl, color=colors[impl])

ax3.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
ax3.set_xlabel("Sequence Length", fontsize=12)
ax3.set_ylabel("Speedup vs Original", fontsize=12)
ax3.set_title(
    "Relative Performance (1.0 = Original DilatedAttention)",
    fontsize=14,
    fontweight="bold",
)
ax3.set_xticks(x + width)
ax3.set_xticklabels(seq_lengths)
ax3.grid(True, alpha=0.3, axis="y")
ax3.legend(loc="best", fontsize=10)

plt.suptitle(
    "FP32 Performance Analysis - GTX 1080 (Pascal)", fontsize=16, fontweight="bold"
)
plt.tight_layout()

# Save the plot
timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
filename = f"benchmarks/fp32_performance_analysis_{timestamp}.png"
plt.savefig(filename, dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualization saved to {filename}")

# Create a summary statistics table
print("\n=== FP32 PERFORMANCE STATISTICS ===\n")
print(
    f"{'Implementation':<35} {'Avg Throughput':>15} {'Peak Throughput':>15} {'Avg Memory':>12}"
)
print("-" * 80)

for impl in implementations:
    throughputs = [t for t in throughput_by_impl[impl] if t > 0]
    memories = [m for m in memory_by_impl[impl] if m > 0]

    if throughputs:
        avg_throughput = np.mean(throughputs)
        peak_throughput = np.max(throughputs)
        avg_memory = np.mean(memories)

        print(
            f"{impl:<35} {avg_throughput:>15,.0f} {peak_throughput:>15,.0f} {avg_memory:>12.1f}"
        )

# Best configuration for each implementation
print("\n=== OPTIMAL CONFIGURATIONS ===\n")
for impl in implementations:
    best_idx = np.argmax(throughput_by_impl[impl])
    if throughput_by_impl[impl][best_idx] > 0:
        print(f"{impl}:")
        print(
            f"  Best at seq_len={seq_lengths[best_idx]}: {throughput_by_impl[impl][best_idx]:,} tokens/sec"
        )
        print(f"  Memory: {memory_by_impl[impl][best_idx]:.1f} MB")
        print()
