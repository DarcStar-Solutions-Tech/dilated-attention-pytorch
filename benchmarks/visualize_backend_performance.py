#!/usr/bin/env python3
"""Visualize backend performance comparison."""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark results
seq_lengths = [1024, 2048, 4096, 8192]

# Time in milliseconds
pytorch_no_hilbert = [2.76, 6.73, 19.53, 304.30]
pytorch_hilbert = [2.74, 6.84, 20.28, 179.48]
triton_no_hilbert = [2.64, 6.49, 35.46, 1268.85]
triton_hilbert = [4.47, 14.71, 164.24, 195.44]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Absolute timing
ax1.loglog(
    seq_lengths, pytorch_no_hilbert, "b-o", label="PyTorch + No Hilbert", linewidth=2
)
ax1.loglog(seq_lengths, pytorch_hilbert, "b--s", label="PyTorch + Hilbert", linewidth=2)
ax1.loglog(
    seq_lengths, triton_no_hilbert, "r-^", label="Triton + No Hilbert", linewidth=2
)
ax1.loglog(seq_lengths, triton_hilbert, "r--d", label="Triton + Hilbert", linewidth=2)

ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("Time (ms)")
ax1.set_title("Backend Performance Comparison")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup relative to PyTorch baseline
baseline = np.array(pytorch_no_hilbert)
pytorch_hilbert_speedup = baseline / np.array(pytorch_hilbert)
triton_no_hilbert_speedup = baseline / np.array(triton_no_hilbert)
triton_hilbert_speedup = baseline / np.array(triton_hilbert)

ax2.semilogx(seq_lengths, np.ones_like(seq_lengths), "k--", alpha=0.5, label="Baseline")
ax2.semilogx(
    seq_lengths, pytorch_hilbert_speedup, "b--s", label="PyTorch + Hilbert", linewidth=2
)
ax2.semilogx(
    seq_lengths,
    triton_no_hilbert_speedup,
    "r-^",
    label="Triton + No Hilbert",
    linewidth=2,
)
ax2.semilogx(
    seq_lengths, triton_hilbert_speedup, "r--d", label="Triton + Hilbert", linewidth=2
)

# Add threshold line
ax2.axvline(x=1024, color="gray", linestyle=":", alpha=0.7, label="Hilbert Threshold")

ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Speedup vs PyTorch Baseline")
ax2.set_title("Relative Performance")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 2.0)

plt.tight_layout()
plt.savefig(
    "benchmarks/backend_performance_comparison.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Print analysis
print("\nKey Findings:")
print("=" * 60)
print("1. At seq_len <= 1024 (below threshold):")
print("   - All backends perform similarly (2.64-2.76ms)")
print("   - Hilbert has minimal impact (identity mapping)")
print("")
print("2. At seq_len = 2048:")
print("   - PyTorch backends are similar (6.73-6.84ms)")
print("   - Triton + Hilbert is 2.2x slower than Triton alone")
print("")
print("3. At seq_len = 4096:")
print("   - PyTorch maintains good performance (19.53-20.28ms)")
print("   - Triton + Hilbert is 4.6x slower than Triton alone")
print("   - Triton without Hilbert is already 1.8x slower than PyTorch")
print("")
print("4. At seq_len = 8192:")
print("   - PyTorch + Hilbert shows 1.7x speedup!")
print("   - Triton + Hilbert shows 1.56x speedup")
print("   - But Triton baseline is 4.2x slower than PyTorch baseline")
print("")
print("Conclusions:")
print("- Hilbert ordering becomes beneficial at 8192+ tokens")
print("- PyTorch backend is more efficient for sequences < 8K")
print("- Triton kernel needs optimization for medium sequences")
print("- Current Triton implementation has high overhead")
