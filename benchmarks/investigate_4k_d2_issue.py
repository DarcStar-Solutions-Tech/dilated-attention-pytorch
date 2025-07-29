#!/usr/bin/env python3
"""
Investigate why 4K with dilation=2 is still slow in Enhanced.
"""

import torch
import time
import sys

sys.path.append('..')

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def profile_4k_d2():
    """Profile the specific slow configuration."""
    
    print("=== Investigating 4K Dilation=2 Performance ===\n")
    
    seq_len = 4096
    dilation_rate = 2
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    
    # Create modules
    unified = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        hilbert_threshold=1024,
    ).cuda().eval()
    
    enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        hilbert_threshold=1024,
        enable_8k_optimization=True,
        enable_multi_row=True,
    ).cuda().eval()
    
    # Check configurations
    print("Configurations:")
    if hasattr(unified, '_get_kernel_config'):
        unified_config = unified._get_kernel_config(seq_len)
        print(f"  Unified: {unified_config}")
    
    if hasattr(enhanced, '_get_optimal_config'):
        enhanced_config = enhanced._get_optimal_config(seq_len)
        print(f"  Enhanced: {enhanced_config}")
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    
    # Profile with different Hilbert settings
    print("\n\nTesting with Hilbert enabled/disabled:")
    
    for use_hilbert in [True, False]:
        print(f"\n  use_hilbert={use_hilbert}:")
        
        # Warmup
        with torch.no_grad():
            _ = unified(x, use_hilbert=use_hilbert)
            _ = enhanced(x, use_hilbert=use_hilbert)
        torch.cuda.synchronize()
        
        # Time Unified
        start = time.perf_counter()
        with torch.no_grad():
            out_unified = unified(x, use_hilbert=use_hilbert)
        torch.cuda.synchronize()
        unified_time = (time.perf_counter() - start) * 1000
        
        # Time Enhanced
        start = time.perf_counter()
        with torch.no_grad():
            out_enhanced = enhanced(x, use_hilbert=use_hilbert)
        torch.cuda.synchronize()
        enhanced_time = (time.perf_counter() - start) * 1000
        
        print(f"    Unified: {unified_time:.2f}ms")
        print(f"    Enhanced: {enhanced_time:.2f}ms")
        print(f"    Ratio: {enhanced_time/unified_time:.2f}x")
        
        # Check output difference
        max_diff = torch.max(torch.abs(out_unified - out_enhanced)).item()
        print(f"    Max diff: {max_diff:.6f}")


def test_different_segment_sizes():
    """Test if segment size affects the issue."""
    
    print("\n\n=== Testing Different Segment Sizes ===")
    
    seq_len = 4096
    dilation_rate = 2
    
    for segment_size in [64, 128, 256, 512]:
        print(f"\nSegment size: {segment_size}")
        
        # Skip if sequence not divisible by segment
        if seq_len % segment_size != 0:
            print("  Skipped (not divisible)")
            continue
        
        unified = UnifiedHilbertAttention(
            hidden_dim=512,
            num_heads=8,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).cuda().eval()
        
        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=512,
            num_heads=8,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).cuda().eval()
        
        x = torch.randn(1, seq_len, 512).cuda()
        
        # Time both
        with torch.no_grad():
            # Warmup
            _ = unified(x)
            _ = enhanced(x)
            torch.cuda.synchronize()
            
            # Unified
            start = time.perf_counter()
            _ = unified(x)
            torch.cuda.synchronize()
            unified_time = (time.perf_counter() - start) * 1000
            
            # Enhanced
            start = time.perf_counter()
            _ = enhanced(x)
            torch.cuda.synchronize()
            enhanced_time = (time.perf_counter() - start) * 1000
        
        print(f"  Unified: {unified_time:.2f}ms")
        print(f"  Enhanced: {enhanced_time:.2f}ms")
        print(f"  Ratio: {enhanced_time/unified_time:.2f}x")


def test_different_dilation_rates():
    """Test if the issue is specific to dilation=2."""
    
    print("\n\n=== Testing Different Dilation Rates ===")
    
    seq_len = 4096
    
    for dilation_rate in [1, 2, 3, 4, 8]:
        print(f"\nDilation rate: {dilation_rate}")
        
        unified = UnifiedHilbertAttention(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=dilation_rate,
        ).cuda().eval()
        
        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=dilation_rate,
        ).cuda().eval()
        
        x = torch.randn(1, seq_len, 512).cuda()
        
        # Time both
        with torch.no_grad():
            # Warmup
            _ = unified(x)
            _ = enhanced(x)
            torch.cuda.synchronize()
            
            # Unified
            start = time.perf_counter()
            _ = unified(x)
            torch.cuda.synchronize()
            unified_time = (time.perf_counter() - start) * 1000
            
            # Enhanced
            start = time.perf_counter()
            _ = enhanced(x)
            torch.cuda.synchronize()
            enhanced_time = (time.perf_counter() - start) * 1000
        
        print(f"  Unified: {unified_time:.2f}ms")
        print(f"  Enhanced: {enhanced_time:.2f}ms")
        print(f"  Ratio: {enhanced_time/unified_time:.2f}x")
        
        if dilation_rate > 1:
            # Check config for sparse
            config = enhanced._get_optimal_config(seq_len)
            print(f"  Enhanced config: block_m={config['block_m']}, block_n={config['block_n']}")


if __name__ == "__main__":
    profile_4k_d2()
    test_different_segment_sizes()
    test_different_dilation_rates()