#!/usr/bin/env python3
"""
Visualize extreme sequence benchmark results.
"""

from datetime import datetime

import matplotlib.pyplot as plt

# Results from our benchmarks
results = {
    "BlockSparseRingDilatedAttention": {
        "sequence_lengths": [32768, 65536, 131072, 262144, 524288, 786432],
        "memory_gb": [0.17, 0.32, 0.63, 1.26, 2.51, 3.01],
        "throughput": [0.21, 0.22, 0.21, 0.12, 0.06, 0.05],
        "success": [True, True, True, True, True, True],
    },
    "RingDilatedAttention": {
        "sequence_lengths": [32768, 65536, 131072, 262144, 524288],
        "memory_gb": [0.20, None, None, None, None],
        "throughput": [0.11, None, None, None, None],
        "success": [True, False, False, False, False],
    },
    "ImprovedDilatedAttention": {
        "sequence_lengths": [32768, 65536, 131072, 262144, 524288],
        "memory_gb": [0.20, None, None, None, None],
        "throughput": [0.22, None, None, None, None],
        "success": [True, False, False, False, False],
    },
}

# Create figure with multiple subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle(
    "Extreme Long Sequence Benchmark Results\nNVIDIA GTX 1080 (8GB)", fontsize=16, fontweight="bold"
)

# Colors for each implementation
colors = {
    "BlockSparseRingDilatedAttention": "#2ecc71",
    "RingDilatedAttention": "#3498db",
    "ImprovedDilatedAttention": "#e74c3c",
}

# 1. Maximum Sequence Length Bar Chart
ax1.set_title("Maximum Sequence Length Achieved", fontsize=14, pad=10)
implementations = list(results.keys())
max_lengths = []
for impl in implementations:
    data = results[impl]
    successful_lengths = [
        seq
        for seq, success in zip(data["sequence_lengths"], data["success"], strict=False)
        if success
    ]
    max_lengths.append(max(successful_lengths) if successful_lengths else 0)

bars = ax1.bar(
    range(len(implementations)), max_lengths, color=[colors[impl] for impl in implementations]
)
ax1.set_xticks(range(len(implementations)))
ax1.set_xticklabels(
    [impl.replace("DilatedAttention", "\nDilatedAttention") for impl in implementations], rotation=0
)
ax1.set_ylabel("Tokens")
ax1.set_yscale("log")
ax1.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar, length in zip(bars, max_lengths, strict=False):
    height = bar.get_height()
    if height > 0:
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.05,
            f"{int(length):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

# Add horizontal line at 1M tokens
ax1.axhline(y=1000000, color="red", linestyle="--", alpha=0.5, label="1M tokens")
ax1.axhline(y=100000, color="orange", linestyle="--", alpha=0.5, label="100K tokens")
ax1.legend()

# 2. Memory Efficiency (Memory per Token)
ax2.set_title("Memory Efficiency Comparison", fontsize=14, pad=10)
ax2.set_xlabel("Sequence Length (tokens)")
ax2.set_ylabel("Memory per Token (MB)")
ax2.set_xscale("log")

for impl, data in results.items():
    valid_lengths = []
    memory_per_token = []

    for seq_len, mem_gb, success in zip(
        data["sequence_lengths"], data["memory_gb"], data["success"], strict=False
    ):
        if success and mem_gb is not None:
            valid_lengths.append(seq_len)
            memory_per_token.append((mem_gb * 1024) / seq_len)  # Convert GB to MB

    if valid_lengths:
        ax2.plot(
            valid_lengths,
            memory_per_token,
            "o-",
            label=impl.replace("DilatedAttention", ""),
            color=colors[impl],
            linewidth=2,
            markersize=8,
        )

ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(bottom=0)

# 3. Throughput vs Sequence Length
ax3.set_title("Throughput vs Sequence Length", fontsize=14, pad=10)
ax3.set_xlabel("Sequence Length (tokens)")
ax3.set_ylabel("Throughput (M tokens/sec)")
ax3.set_xscale("log")

for impl, data in results.items():
    valid_lengths = []
    throughputs = []

    for seq_len, tput, success in zip(
        data["sequence_lengths"], data["throughput"], data["success"], strict=False
    ):
        if success and tput is not None:
            valid_lengths.append(seq_len)
            throughputs.append(tput)

    if valid_lengths:
        ax3.plot(
            valid_lengths,
            throughputs,
            "o-",
            label=impl.replace("DilatedAttention", ""),
            color=colors[impl],
            linewidth=2,
            markersize=8,
        )

ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(bottom=0)

# Add summary text box
summary_text = (
    "Key Findings:\n"
    "• BlockSparse handles 786,432 tokens (24x more than others)\n"
    "• Consistent 0.004 MB/token memory usage\n"
    "• Other implementations hit CUDA limits at >32K tokens\n"
    "• BlockSparse enables ~600 pages of text on consumer GPU"
)
fig.text(
    0.02,
    0.02,
    summary_text,
    transform=fig.transFigure,
    fontsize=11,
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# Save figure
timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
output_file = f"docs/benchmarks/extreme-sequence-visualization-{timestamp}.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Visualization saved to: {output_file}")

# Also create a simple comparison table
print("\n" + "=" * 60)
print("EXTREME SEQUENCE LENGTH COMPARISON")
print("=" * 60)
print(f"{'Implementation':<35} {'Max Tokens':>12} {'vs Dense':>10}")
print("-" * 60)
for impl in implementations:
    max_len = max_lengths[implementations.index(impl)]
    ratio = max_len / 32768 if max_len > 0 else 0
    print(f"{impl:<35} {max_len:>12,} {ratio:>9.1f}x")
print("=" * 60)
