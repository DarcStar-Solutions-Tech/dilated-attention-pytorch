#!/usr/bin/env python3
"""
Debug why some sparse configurations are still slow in Enhanced.
"""

import torch
import sys

sys.path.append('..')

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def test_kernel_configs():
    """Check what kernel configurations are being used."""
    
    print("=== Kernel Configuration Analysis ===\n")
    
    # Test different sequence lengths
    test_seq_lens = [2048, 4096, 8192]
    
    for seq_len in test_seq_lens:
        print(f"\nSequence length: {seq_len}")
        
        # Unified
        unified = UnifiedHilbertAttention(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=2,
        )
        
        if hasattr(unified, '_get_kernel_config'):
            config = unified._get_kernel_config(seq_len)
            print(f"  Unified config: {config}")
        
        # Enhanced
        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=2,
            enable_8k_optimization=True,
            enable_multi_row=True,
        )
        
        if hasattr(enhanced, '_get_optimal_config'):
            config = enhanced._get_optimal_config(seq_len)
            print(f"  Enhanced config: {config}")
            
            # Check specific parameters that might affect sparse performance
            if 'rows_per_block' in config and config['rows_per_block'] > 1:
                print(f"    ⚠️  Multi-row processing enabled: {config['rows_per_block']} rows")
            if 'block_m' in config and config['block_m'] > 64:
                print(f"    ⚠️  Large block_m: {config['block_m']}")
            if 'block_n' in config and config['block_n'] > 64:
                print(f"    ⚠️  Large block_n: {config['block_n']}")


def test_correctness_detailed():
    """Test correctness with detailed comparison."""
    
    print("\n\n=== Correctness Analysis ===")
    
    # Simple test case
    batch_size = 1
    seq_len = 512  # Small for detailed analysis
    hidden_dim = 64
    num_heads = 4
    dilation_rate = 2
    
    # Create identical input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    
    # Create modules with same configuration
    config = {
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'segment_size': 128,
        'dilation_rate': dilation_rate,
        'hilbert_threshold': 256,  # Force Triton path
    }
    
    unified = UnifiedHilbertAttention(**config).cuda().eval()
    
    enhanced_config = config.copy()
    enhanced_config['enable_8k_optimization'] = False  # Disable for consistency
    enhanced_config['enable_multi_row'] = False  # Disable for consistency
    enhanced = UnifiedHilbertAttentionOptimizedEnhanced(**enhanced_config).cuda().eval()
    
    # Run forward pass
    with torch.no_grad():
        out_unified = unified(x, use_hilbert=False)  # Disable Hilbert for simpler comparison
        out_enhanced = enhanced(x, use_hilbert=False)
    
    # Compare outputs
    print(f"\nOutput shapes:")
    print(f"  Unified: {out_unified.shape}")
    print(f"  Enhanced: {out_enhanced.shape}")
    
    # Detailed comparison
    abs_diff = torch.abs(out_unified - out_enhanced)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    print(f"\nDifference statistics:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    # Check relative difference
    rel_diff = abs_diff / (torch.abs(out_unified) + 1e-8)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()
    
    print(f"  Max relative difference: {max_rel_diff:.6f}")
    print(f"  Mean relative difference: {mean_rel_diff:.6f}")
    
    # Find where differences are largest
    if max_diff > 1e-3:
        max_idx = torch.argmax(abs_diff)
        max_idx = torch.unravel_index(max_idx, abs_diff.shape)
        print(f"\nLargest difference at position: {max_idx}")
        print(f"  Unified value: {out_unified[max_idx].item():.6f}")
        print(f"  Enhanced value: {out_enhanced[max_idx].item():.6f}")


def check_sparse_iteration_logic():
    """Check if sparse iteration logic matches between implementations."""
    
    print("\n\n=== Sparse Iteration Logic ===")
    
    seq_len = 512
    segment_size = 128
    dilation_rate = 2
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Segment size: {segment_size}")
    print(f"  Dilation rate: {dilation_rate}")
    
    # Calculate how sparse positions should be computed
    num_segments = (seq_len + segment_size - 1) // segment_size
    print(f"  Number of segments: {num_segments}")
    
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)
        
        # Unified approach
        num_active = (seg_end - seg_start + dilation_rate - 1) // dilation_rate
        
        print(f"\nSegment {seg_idx}:")
        print(f"  Range: [{seg_start}, {seg_end})")
        print(f"  Active positions: {num_active}")
        
        # Show first few sparse positions
        positions = []
        for i in range(min(5, num_active)):
            pos = seg_start + i * dilation_rate
            if pos < seg_end:
                positions.append(pos)
        print(f"  First positions: {positions}")


def test_specific_config():
    """Test the specific configuration that's slow."""
    
    print("\n\n=== Testing Slow Configuration (4K, d=2) ===")
    
    seq_len = 4096
    dilation_rate = 2
    batch_size = 1
    hidden_dim = 512
    num_heads = 8
    
    # Get Enhanced configuration for this sequence length
    enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=128,
        dilation_rate=dilation_rate,
        enable_8k_optimization=True,
        enable_multi_row=True,
    )
    
    config = enhanced._get_optimal_config(seq_len)
    print(f"\nEnhanced config for 4K sequence: {config}")
    
    # Check if certain optimizations might hurt sparse performance
    if config.get('rows_per_block', 1) > 1:
        print("\n⚠️  Multi-row processing is enabled")
        print("  This might be inefficient for sparse patterns")
    
    if config.get('block_m', 64) > 64 or config.get('block_n', 64) > 64:
        print("\n⚠️  Large block sizes detected")
        print("  Large blocks are inefficient for sparse patterns")


if __name__ == "__main__":
    test_kernel_configs()
    test_correctness_detailed()
    check_sparse_iteration_logic()
    test_specific_config()