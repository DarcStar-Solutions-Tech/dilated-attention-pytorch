"""
Analyze the sparsity pattern to understand the bottleneck
"""
import torch
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)

def analyze_sparsity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    seq_len = 2048
    num_heads = 8
    head_dim = 64
    block_size = 32
    
    # Create sparse config
    sparse_config = SparsePatternConfig(
        pattern_type='dilated_sparse',
        sparsity_ratio=0.1,  # 90% sparse
        block_size=block_size,
    )
    
    # Create pattern generator
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import SparsePatternGenerator
    generator = SparsePatternGenerator(sparse_config)
    
    # Create pattern
    num_blocks = seq_len // block_size
    pattern = generator.create_pattern(seq_len, num_heads, device)
    
    print(f"Sequence length: {seq_len}")
    print(f"Block size: {block_size}")
    print(f"Number of blocks: {num_blocks}")
    print(f"Pattern shape: {pattern.shape}")
    
    # Analyze sparsity
    if pattern.dim() == 2:
        # Block pattern
        total_blocks = pattern.numel()
        active_blocks = pattern.sum().item()
    else:
        # Per-head pattern
        total_blocks = pattern[0, 0].numel()
        active_blocks = pattern[0, 0].sum().item()
    
    print(f"\nSparsity Analysis:")
    print(f"Total block pairs: {total_blocks}")
    print(f"Active block pairs: {active_blocks}")
    print(f"Sparsity: {(1 - active_blocks/total_blocks) * 100:.1f}%")
    
    # Estimate iterations in nonzero loop
    if pattern.dim() == 4:  # [batch, heads, blocks, blocks]
        ring_pattern = pattern  # Simplified for analysis
        nonzero_count = ring_pattern.sum().item()
        print(f"\nFor batch={batch_size}, heads={num_heads}:")
        print(f"Total nonzero elements to iterate: {nonzero_count}")
        print(f"That's {nonzero_count} Python loop iterations!")
        
        # Estimate time
        # Assuming 1ms per iteration (conservative)
        estimated_time = nonzero_count * 0.001  # seconds
        print(f"Estimated time at 1ms/iteration: {estimated_time:.2f}s")

if __name__ == "__main__":
    analyze_sparsity()