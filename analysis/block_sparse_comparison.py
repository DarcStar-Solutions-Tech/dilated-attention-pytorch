"""
Compare BlockSparseAttention vs BlockSparseDilatedAttention

This script analyzes the differences in computation patterns, memory usage,
and performance between the two implementations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import time

from dilated_attention_pytorch import (
    BlockSparseAttention,
    BlockSparseDilatedAttention,
    SparsePatternConfig,
)


def visualize_attention_patterns(
    block_sparse_output: torch.Tensor,
    block_sparse_dilated_output: torch.Tensor,
    seq_len: int,
    save_path: str = "block_sparse_comparison.png",
):
    """Visualize the effective attention patterns of both implementations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # For visualization, we'll look at the first head of the first batch
    bs_pattern = block_sparse_output[0, :, 0, :].detach().cpu().numpy()
    bsd_pattern = block_sparse_dilated_output[0, :, 0, :].detach().cpu().numpy()

    # Compute attention norms to show which positions are attending strongly
    bs_norm = np.linalg.norm(bs_pattern, axis=1)
    bsd_norm = np.linalg.norm(bsd_pattern, axis=1)

    # Plot attention strength patterns
    axes[0].plot(bs_norm, label="Block Sparse", alpha=0.7)
    axes[0].plot(bsd_norm, label="Block Sparse Dilated", alpha=0.7)
    axes[0].set_title("Attention Output Norms")
    axes[0].set_xlabel("Sequence Position")
    axes[0].set_ylabel("L2 Norm")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot heatmaps of output patterns
    im1 = axes[1].imshow(bs_pattern.T, aspect="auto", cmap="viridis")
    axes[1].set_title("Block Sparse Output Pattern")
    axes[1].set_xlabel("Sequence Position")
    axes[1].set_ylabel("Head Dimension")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(bsd_pattern.T, aspect="auto", cmap="viridis")
    axes[2].set_title("Block Sparse Dilated Output Pattern")
    axes[2].set_xlabel("Sequence Position")
    axes[2].set_ylabel("Head Dimension")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def analyze_sparsity_patterns(
    model_type: str, seq_len: int, block_size: int, sparsity_ratio: float
) -> Dict[str, float]:
    """Analyze the actual sparsity patterns of each model."""
    results = {}

    # Create model
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    if model_type == "block_sparse":
        model = BlockSparseAttention(sparse_config=sparse_config)
        # Get block indices
        num_blocks = seq_len // block_size
        row_indices, col_indices = model._get_sparse_block_indices(
            num_blocks, num_heads=1, device=torch.device("cpu")
        )

        # Calculate actual sparsity
        total_blocks = num_blocks * num_blocks
        active_blocks = len(row_indices)
        actual_sparsity = 1.0 - (active_blocks / total_blocks)

        # Calculate receptive field statistics
        receptive_fields = []
        for i in range(num_blocks):
            attending_blocks = sum(
                1 for r, c in zip(row_indices, col_indices) if r == i
            )
            receptive_fields.append(attending_blocks * block_size)

        results["actual_sparsity"] = actual_sparsity
        results["active_blocks"] = active_blocks
        results["total_blocks"] = total_blocks
        results["avg_receptive_field"] = np.mean(receptive_fields)
        results["max_receptive_field"] = np.max(receptive_fields)
        results["min_receptive_field"] = np.min(receptive_fields)

    else:  # block_sparse_dilated
        model = BlockSparseDilatedAttention(
            segment_lengths=[block_size, block_size * 2],
            dilation_rates=[1, 2],
            sparse_config=sparse_config,
        )

        # Same block-level sparsity as BlockSparseAttention
        num_blocks = seq_len // block_size
        row_indices, col_indices = model.block_sparse._get_sparse_block_indices(
            num_blocks, num_heads=1, device=torch.device("cpu")
        )

        total_blocks = num_blocks * num_blocks
        active_blocks = len(row_indices)
        actual_sparsity = 1.0 - (active_blocks / total_blocks)

        # But within each block, dilated attention adds token-level sparsity
        # Estimate effective sparsity including dilation
        _ = block_size
        # With dilation rate 2, roughly half the tokens are sampled
        avg_dilation_sparsity = 0.5  # Approximate
        effective_sparsity = (
            actual_sparsity + (1 - actual_sparsity) * avg_dilation_sparsity
        )

        results["actual_block_sparsity"] = actual_sparsity
        results["effective_total_sparsity"] = effective_sparsity
        results["active_blocks"] = active_blocks
        results["total_blocks"] = total_blocks
        results["dilation_rates"] = [1, 2]
        results["multi_scale_coverage"] = True

    return results


def benchmark_performance(
    seq_lengths: List[int],
    batch_size: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    sparsity_ratio: float = 0.1,
    num_warmup: int = 10,
    num_trials: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Benchmark both implementations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"block_sparse": {}, "block_sparse_dilated": {}}

    for seq_len in seq_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")

        # Create models
        block_size = min(256, seq_len // 4)  # Adaptive block size
        sparse_config = SparsePatternConfig(
            pattern_type="local_window",
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
        )

        bs_model = BlockSparseAttention(sparse_config=sparse_config).to(device)

        # For dilated, use segments that fit within blocks
        # Ensure segments are valid
        segment_lengths = (
            [block_size // 4, block_size // 4] if block_size >= 64 else [block_size]
        )
        dilation_rates = [1, 2] if len(segment_lengths) > 1 else [1]
        bsd_model = BlockSparseDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
        ).to(device)

        # Create test tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Benchmark BlockSparseAttention
        for _ in range(num_warmup):
            _ = bs_model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_trials):
            _ = bs_model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        bs_time = (time.time() - start_time) / num_trials
        results["block_sparse"][seq_len] = bs_time

        # Benchmark BlockSparseDilatedAttention
        for _ in range(num_warmup):
            _ = bsd_model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_trials):
            _ = bsd_model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        bsd_time = (time.time() - start_time) / num_trials
        results["block_sparse_dilated"][seq_len] = bsd_time

        print(f"  Block Sparse: {bs_time * 1000:.2f} ms")
        print(f"  Block Sparse Dilated: {bsd_time * 1000:.2f} ms")
        print(f"  Overhead: {(bsd_time / bs_time - 1) * 100:.1f}%")

    return results


def main():
    """Run comprehensive comparison."""
    print("=" * 80)
    print("BlockSparseAttention vs BlockSparseDilatedAttention Comparison")
    print("=" * 80)

    # 1. Analyze sparsity patterns
    print("\n1. SPARSITY PATTERN ANALYSIS")
    print("-" * 40)

    seq_len = 2048
    block_size = 256
    sparsity_ratio = 0.1

    bs_patterns = analyze_sparsity_patterns(
        "block_sparse", seq_len, block_size, sparsity_ratio
    )
    bsd_patterns = analyze_sparsity_patterns(
        "block_sparse_dilated", seq_len, block_size, sparsity_ratio
    )

    print("\nBlockSparseAttention:")
    print(f"  Block-level sparsity: {bs_patterns['actual_sparsity']:.1%}")
    print(
        f"  Active blocks: {bs_patterns['active_blocks']} / {bs_patterns['total_blocks']}"
    )
    print(f"  Avg receptive field: {bs_patterns['avg_receptive_field']:.0f} tokens")

    print("\nBlockSparseDilatedAttention:")
    print(f"  Block-level sparsity: {bsd_patterns['actual_block_sparsity']:.1%}")
    print(f"  Effective total sparsity: {bsd_patterns['effective_total_sparsity']:.1%}")
    print(f"  Multi-scale coverage: {bsd_patterns['multi_scale_coverage']}")
    print(f"  Dilation rates: {bsd_patterns['dilation_rates']}")

    # 2. Key differences
    print("\n2. KEY DIFFERENCES")
    print("-" * 40)
    print("\nBlockSparseAttention:")
    print("  - Applies sparsity at block level only")
    print("  - Within each block: full dense attention")
    print("  - Single scale of attention")
    print("  - Simpler, faster computation")

    print("\nBlockSparseDilatedAttention:")
    print("  - Applies sparsity at block level AND token level")
    print("  - Within each block: dilated attention with segments")
    print("  - Multi-scale attention (different dilation rates)")
    print("  - More complex but captures multi-scale patterns")

    # 3. Computational comparison
    print("\n3. COMPUTATIONAL ANALYSIS")
    print("-" * 40)

    # BlockSparse: O(n * b * s^2) where n=num_active_blocks, b=batch*heads, s=block_size
    # BlockSparseDilated: O(n * b * s * (s/d)) where d=average_dilation

    print("\nTime Complexity (per active block):")
    print(f"  BlockSparse: O(block_size²) = O({block_size}²) = O({block_size**2})")
    print(
        f"  BlockSparseDilated: O(block_size × effective_size) ≈ O({block_size} × {block_size // 2})"
    )
    print("  Theoretical speedup within blocks: ~2x due to dilation")

    # 4. Memory usage
    print("\n4. MEMORY USAGE ANALYSIS")
    print("-" * 40)

    batch_size = 4
    num_heads = 8
    head_dim = 64

    # Both use same block-sparse pattern, so same memory for storing indices
    block_memory = bs_patterns["active_blocks"] * 2 * 4  # row/col indices as int32

    # Within blocks, BlockSparseDilated may use slightly more memory for
    # segment processing but it's temporary
    print(f"\nBlock index memory: {block_memory / 1024:.2f} KB")
    print("Peak memory usage: Similar (both process one block at a time)")
    print("BlockSparseDilated has slight overhead for segment bookkeeping")

    # 5. Performance benchmark
    print("\n5. PERFORMANCE BENCHMARK")
    print("-" * 40)

    if torch.cuda.is_available():
        seq_lengths = [512, 1024, 2048, 4096]
        results = benchmark_performance(seq_lengths, sparsity_ratio=0.1)

        print("\nRelative Performance (BlockSparseDilated / BlockSparse):")
        for seq_len in seq_lengths:
            ratio = (
                results["block_sparse_dilated"][seq_len]
                / results["block_sparse"][seq_len]
            )
            print(f"  Seq {seq_len}: {ratio:.2f}x")
    else:
        print("  (Skipping - CUDA not available)")

    # 6. When to use which
    print("\n6. USAGE RECOMMENDATIONS")
    print("-" * 40)

    print("\nUse BlockSparseAttention when:")
    print("  - Maximum speed is critical")
    print("  - Single-scale patterns are sufficient")
    print("  - Working with uniform data (e.g., images)")

    print("\nUse BlockSparseDilatedAttention when:")
    print("  - Multi-scale patterns are important")
    print("  - Working with hierarchical data (e.g., text, time series)")
    print("  - Need better long-range dependency modeling")
    print("  - Can afford ~20-50% computational overhead")

    # 7. Visual comparison
    print("\n7. VISUAL COMPARISON")
    print("-" * 40)

    # Create small example for visualization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 512
    batch_size = 1
    num_heads = 4
    head_dim = 32

    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.2,
        block_size=64,
    )

    bs_model = BlockSparseAttention(sparse_config=sparse_config).to(device)
    bsd_model = BlockSparseDilatedAttention(
        segment_lengths=[32, 32],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
    ).to(device)

    # Create structured input to see pattern differences
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    # Add some structure to the input
    for i in range(0, seq_len, 64):
        q[:, i : i + 32, :, :] *= 2.0  # Amplify first half of each block

    k = q.clone()
    v = q.clone()

    bs_output = bs_model(q, k, v)
    bsd_output = bsd_model(q, k, v)

    visualize_attention_patterns(bs_output, bsd_output, seq_len)

    print(
        "\nComparison complete! Check 'block_sparse_comparison.png' for visualization."
    )


if __name__ == "__main__":
    main()
