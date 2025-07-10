#!/usr/bin/env python3
"""Analyze and compare ring attention scaling between 1 and 2 GPUs."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Load results
results_dir = Path("benchmarks/results/ring/sdpa_scaling")
single_gpu_file = sorted(results_dir.glob("sdpa_scaling_w1_*.json"))[-1]
two_gpu_file = sorted(results_dir.glob("sdpa_scaling_w2_*.json"))[-1]

single_gpu_results = json.load(open(single_gpu_file))
two_gpu_results = json.load(open(two_gpu_file))

# Extract data
seq_lens_1gpu = [r["sequence_length"] for r in single_gpu_results]
seq_lens_2gpu = [r["sequence_length"] for r in two_gpu_results]

memory_1gpu = [r["memory_used_mb"] for r in single_gpu_results]
memory_2gpu = [r["memory_used_mb"] for r in two_gpu_results]

throughput_1gpu = [r["throughput"] for r in single_gpu_results]
throughput_2gpu = [r["throughput"] for r in two_gpu_results]

kb_per_token_1gpu = [r["memory_per_token_kb"] for r in single_gpu_results]
kb_per_token_2gpu = [r["memory_per_token_kb"] for r in two_gpu_results]

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Memory Scaling Comparison
ax1.plot(
    seq_lens_1gpu,
    memory_1gpu,
    "o-",
    label="1 GPU",
    linewidth=2,
    markersize=8,
    color="#2E86AB",
)
ax1.plot(
    seq_lens_2gpu,
    memory_2gpu,
    "s-",
    label="2 GPUs",
    linewidth=2,
    markersize=8,
    color="#F24236",
)
ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("Memory Used (MB)")
ax1.set_title("Ring Attention Memory Scaling")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add annotations for key points
for i, seq_len in enumerate(seq_lens_2gpu):
    if seq_len in [16384, 32768]:
        ax1.annotate(
            f"{seq_len:,}",
            xy=(seq_len, memory_2gpu[i]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
        )

# 2. Memory Reduction Factor
common_seq_lens = sorted(set(seq_lens_1gpu) & set(seq_lens_2gpu))
memory_reduction = []
for seq_len in common_seq_lens:
    idx1 = seq_lens_1gpu.index(seq_len)
    idx2 = seq_lens_2gpu.index(seq_len)
    reduction = memory_1gpu[idx1] / memory_2gpu[idx2]
    memory_reduction.append(reduction)

ax2.plot(common_seq_lens, memory_reduction, "g^-", linewidth=2, markersize=10)
ax2.axhline(y=2, color="r", linestyle="--", alpha=0.5, label="Perfect scaling (2x)")
ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Memory Reduction Factor")
ax2.set_title("Memory Reduction: 1 GPU vs 2 GPUs")
ax2.set_xscale("log")
ax2.set_ylim(1.5, 2.1)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Throughput Comparison
ax3.plot(
    seq_lens_1gpu,
    throughput_1gpu,
    "o-",
    label="1 GPU",
    linewidth=2,
    markersize=8,
    color="#2E86AB",
)
ax3.plot(
    seq_lens_2gpu,
    throughput_2gpu,
    "s-",
    label="2 GPUs",
    linewidth=2,
    markersize=8,
    color="#F24236",
)
ax3.set_xlabel("Sequence Length")
ax3.set_ylabel("Throughput (tokens/sec)")
ax3.set_title("Throughput Comparison")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Memory per Token Comparison
x = np.arange(len(common_seq_lens))
width = 0.35

# Get memory per token for common sequences
kb_1gpu_common = [kb_per_token_1gpu[seq_lens_1gpu.index(s)] for s in common_seq_lens]
kb_2gpu_common = [kb_per_token_2gpu[seq_lens_2gpu.index(s)] for s in common_seq_lens]

ax4.bar(x - width / 2, kb_1gpu_common, width, label="1 GPU", alpha=0.8, color="#2E86AB")
ax4.bar(
    x + width / 2, kb_2gpu_common, width, label="2 GPUs", alpha=0.8, color="#F24236"
)
ax4.set_xlabel("Sequence Length")
ax4.set_ylabel("Memory per Token (KB)")
ax4.set_title("Memory Efficiency (O(n/k) Scaling)")
ax4.set_xticks(x)
ax4.set_xticklabels([f"{s:,}" for s in common_seq_lens])
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for i, (kb1, kb2) in enumerate(zip(kb_1gpu_common, kb_2gpu_common)):
    ax4.text(
        i - width / 2, kb1 + 0.5, f"{kb1:.1f}", ha="center", va="bottom", fontsize=8
    )
    ax4.text(
        i + width / 2, kb2 + 0.5, f"{kb2:.1f}", ha="center", va="bottom", fontsize=8
    )

plt.tight_layout()
plt.savefig(results_dir / "ring_scaling_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# Print detailed analysis
print("\n" + "=" * 80)
print("Ring Attention Multi-GPU Scaling Analysis")
print("=" * 80)

print("\nMemory Scaling (1 GPU → 2 GPUs):")
print(
    f"{'Seq Len':<12} {'1 GPU (MB)':<12} {'2 GPUs (MB)':<12} {'Reduction':<10} {'Efficiency':<10}"
)
print("-" * 65)
for seq_len in common_seq_lens:
    idx1 = seq_lens_1gpu.index(seq_len)
    idx2 = seq_lens_2gpu.index(seq_len)
    reduction = memory_1gpu[idx1] / memory_2gpu[idx2]
    efficiency = (reduction / 2.0) * 100  # How close to perfect 2x
    print(
        f"{seq_len:<12,} {memory_1gpu[idx1]:<12.2f} {memory_2gpu[idx2]:<12.2f} {reduction:<10.2f}x {efficiency:<10.1f}%"
    )

print("\nKey Findings:")
avg_reduction = sum(memory_reduction) / len(memory_reduction)
print(f"- Average memory reduction: {avg_reduction:.2f}x (expected 2x)")
print(f"- Efficiency: {(avg_reduction / 2) * 100:.1f}% of theoretical maximum")

# Throughput analysis
print("\nThroughput Analysis:")
for seq_len in common_seq_lens:
    idx1 = seq_lens_1gpu.index(seq_len)
    idx2 = seq_lens_2gpu.index(seq_len)
    speedup = throughput_2gpu[idx2] / throughput_1gpu[idx1]
    print(f"  {seq_len:,} tokens: {speedup:.1f}x speedup with 2 GPUs")

# Maximum sequence achieved
max_seq_2gpu = max(seq_lens_2gpu)
print("\nMaximum sequence length achieved:")
print(f"- 1 GPU: {max(seq_lens_1gpu):,} tokens")
print(f"- 2 GPUs: {max_seq_2gpu:,} tokens")
print(f"- Extension factor: {max_seq_2gpu / max(seq_lens_1gpu):.1f}x")

# Memory per token analysis
avg_kb_1gpu = np.mean(kb_per_token_1gpu)
avg_kb_2gpu = np.mean(kb_per_token_2gpu)
print("\nMemory per Token:")
print(f"- 1 GPU: {avg_kb_1gpu:.2f} KB/token")
print(f"- 2 GPUs: {avg_kb_2gpu:.2f} KB/token")
print(f"- Reduction: {(1 - avg_kb_2gpu / avg_kb_1gpu) * 100:.1f}%")

print(f"\nVisualization saved to: {results_dir / 'ring_scaling_comparison.png'}")
