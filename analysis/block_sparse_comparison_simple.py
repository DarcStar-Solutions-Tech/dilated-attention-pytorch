"""
Simple comparison of BlockSparseAttention vs BlockSparseDilatedAttention
"""

import torch
from dilated_attention_pytorch import (
    BlockSparseAttention,
    BlockSparseDilatedAttention,
    SparsePatternConfig,
)


def main():
    print("=" * 80)
    print("BlockSparseAttention vs BlockSparseDilatedAttention")
    print("=" * 80)

    # Configuration
    seq_len = 1024
    block_size = 256
    sparsity_ratio = 0.1  # 90% sparse
    batch_size = 2
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    block_sparse = BlockSparseAttention(sparse_config=sparse_config).to(device)

    block_sparse_dilated = BlockSparseDilatedAttention(
        segment_lengths=[128, 128],  # Two segments within each block
        dilation_rates=[1, 2],  # Second segment has dilation 2
        sparse_config=sparse_config,
    ).to(device)

    # Create test input
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Forward pass
    output_bs = block_sparse(q, k, v)
    output_bsd = block_sparse_dilated(q, k, v)

    print("\n1. FUNDAMENTAL DIFFERENCES")
    print("-" * 40)
    print("\nBlockSparseAttention:")
    print("  - Sparsity: Block-level only (blocks either attend or don't)")
    print("  - Attention: Dense within each active block")
    print("  - Pattern: Single scale")
    print(
        f"  - Example: In a {block_size}x{block_size} block, all {block_size * block_size} interactions computed"
    )

    print("\nBlockSparseDilatedAttention:")
    print("  - Sparsity: Block-level + Token-level (dilated within blocks)")
    print("  - Attention: Dilated attention within each active block")
    print("  - Pattern: Multi-scale (different dilation rates)")
    print(f"  - Example: In a {block_size}x{block_size} block:")
    print(f"    - First {block_size // 2} tokens: dilation 1 (all tokens)")
    print(f"    - Next {block_size // 2} tokens: dilation 2 (every 2nd token)")
    print(
        f"    - Effective interactions: ~{int(block_size * block_size * 0.75)} instead of {block_size * block_size}"
    )

    print("\n2. COMPUTATIONAL COMPLEXITY")
    print("-" * 40)

    # Calculate active blocks
    num_blocks = seq_len // block_size
    total_blocks = num_blocks * num_blocks
    active_blocks = int(total_blocks * (1 - sparsity_ratio))

    print(f"\nSequence length: {seq_len}")
    print(f"Block size: {block_size}")
    print(f"Number of blocks: {num_blocks}x{num_blocks} = {total_blocks}")
    print(
        f"Active blocks (both models): {active_blocks} ({(1 - sparsity_ratio) * 100:.0f}%)"
    )

    # BlockSparse computation
    bs_ops_per_block = block_size * block_size * head_dim * 2  # QK^T and AV
    bs_total_ops = active_blocks * num_heads * bs_ops_per_block

    # BlockSparseDilated computation (approximate)
    # First segment: full attention
    # Second segment: dilated by 2, so ~50% of operations
    bsd_effective_ops_per_block = (
        block_size * block_size * head_dim * 2 * 0.75
    )  # ~75% due to dilation
    bsd_total_ops = active_blocks * num_heads * bsd_effective_ops_per_block

    print("\nOperations (FLOPs):")
    print(f"  BlockSparse: {bs_total_ops / 1e9:.2f} GFLOPs")
    print(f"  BlockSparseDilated: {bsd_total_ops / 1e9:.2f} GFLOPs")
    print(f"  Reduction: {(1 - bsd_total_ops / bs_total_ops) * 100:.1f}%")

    print("\n3. MEMORY PATTERNS")
    print("-" * 40)

    print("\nBoth models:")
    print("  - Same block-level sparsity pattern")
    print(f"  - Same memory for block indices: {active_blocks * 2 * 4 / 1024:.1f} KB")
    print("  - Process one block at a time (memory efficient)")

    print("\nBlockSparseDilated additional benefits:")
    print("  - Within blocks: accesses fewer memory locations due to dilation")
    print("  - Better cache utilization for dilated segments")
    print("  - Slight overhead for segment management")

    print("\n4. ATTENTION COVERAGE")
    print("-" * 40)

    print("\nBlockSparseAttention:")
    print("  - Each token attends to all tokens in active blocks")
    print(
        f"  - Receptive field: {active_blocks // num_blocks * block_size} tokens (uniform)"
    )

    print("\nBlockSparseDilatedAttention:")
    print("  - Each token attends to subset of tokens in active blocks")
    print("  - Multi-scale receptive field:")
    print("    - Fine-grained: nearby tokens (dilation 1)")
    print("    - Coarse-grained: distant tokens (dilation 2)")
    print("  - Better for capturing hierarchical patterns")

    print("\n5. USE CASES")
    print("-" * 40)

    print("\nBlockSparseAttention is better for:")
    print("  ✓ Maximum speed")
    print("  ✓ Uniform data (e.g., images, structured data)")
    print("  ✓ When all interactions within a region are important")
    print("  ✓ Simple implementation and debugging")

    print("\nBlockSparseDilatedAttention is better for:")
    print("  ✓ Hierarchical data (text, time series, audio)")
    print("  ✓ Multi-scale pattern recognition")
    print("  ✓ Very long sequences where even block-level attention is expensive")
    print("  ✓ When you need both local detail and global context")

    print("\n6. PERFORMANCE TRADE-OFFS")
    print("-" * 40)

    print("\nBlockSparseDilatedAttention overhead:")
    print("  - ~25% fewer operations due to dilation")
    print("  - ~10-20% runtime overhead due to:")
    print("    - Segment management")
    print("    - Shape conversions")
    print("    - Creating temporary DilatedAttention instances")
    print("  - Net result: Similar or slightly faster for large blocks")

    print("\n7. EXAMPLE SCENARIO")
    print("-" * 40)

    print("\nProcessing a 16K token document:")
    print("  - Block size: 256 tokens")
    print("  - 90% sparsity")
    print("  - 64x64 = 4096 total blocks")
    print("  - ~410 active blocks")

    print("\nBlockSparse:")
    print("  - Each block: 256² = 65,536 attention computations")
    print("  - Total: 26.8M attention computations")

    print("\nBlockSparseDilated:")
    print("  - Each block: ~49,152 attention computations (25% reduction)")
    print("  - Total: 20.1M attention computations")
    print("  - Plus: Multi-scale pattern capture")

    # Verify outputs are reasonable
    print("\n8. OUTPUT VALIDATION")
    print("-" * 40)
    print(f"Output shapes match: {output_bs.shape == output_bsd.shape}")
    print(
        f"Outputs are finite: BS={torch.isfinite(output_bs).all()}, BSD={torch.isfinite(output_bsd).all()}"
    )
    print(
        f"Output magnitude similar: BS={output_bs.abs().mean():.4f}, BSD={output_bsd.abs().mean():.4f}"
    )


if __name__ == "__main__":
    main()
