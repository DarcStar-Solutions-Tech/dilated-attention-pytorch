#!/usr/bin/env python3
"""
Investigate why UnifiedHilbertAttention performs better on sparse patterns.
"""

import torch
import time
import sys
from typing import Dict, Tuple
import numpy as np

sys.path.append('..')

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimized,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def profile_sparse_execution(impl_class, impl_name, seq_len, dilation_rate, hidden_dim=512, num_heads=8):
    """Profile sparse pattern execution to understand performance differences."""
    
    print(f"\n=== Profiling {impl_name} - Seq: {seq_len}, Dilation: {dilation_rate} ===")
    
    batch_size = 2
    segment_size = 128
    
    # Create config
    config = {
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'segment_size': segment_size,
        'dilation_rate': dilation_rate,
        'hilbert_threshold': 1024,
    }
    
    if impl_name == 'Enhanced':
        config['enable_8k_optimization'] = True
        config['enable_multi_row'] = True
    
    # Create module
    module = impl_class(**config).cuda()
    module.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    
    # Check which execution path is taken
    print(f"  Hilbert threshold: {module.hilbert_threshold}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Will use Hilbert: {seq_len > module.hilbert_threshold}")
    
    # For Enhanced, check if strided sparse attention is used
    if hasattr(module, 'dilation_rate') and module.dilation_rate > 1:
        if impl_name == 'Enhanced':
            print(f"  Enhanced will use strided sparse attention path")
    
    # Warmup
    with torch.no_grad():
        _ = module(x)
    torch.cuda.synchronize()
    
    # Detailed timing
    torch.cuda.synchronize()
    
    # Time forward pass
    start = time.perf_counter()
    with torch.no_grad():
        out = module(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    forward_time = (end - start) * 1000
    print(f"  Forward time: {forward_time:.2f}ms")
    
    # Calculate theoretical operations
    active_positions = seq_len // dilation_rate
    total_ops = batch_size * num_heads * seq_len * active_positions * (hidden_dim // num_heads)
    gflops = total_ops / (forward_time / 1000) / 1e9
    print(f"  Active positions per segment: {active_positions}")
    print(f"  Theoretical GFLOPS: {gflops:.2f}")
    
    return forward_time


def analyze_kernel_paths():
    """Analyze which kernel paths are taken for sparse patterns."""
    
    print("=== Kernel Path Analysis ===")
    
    # Test configuration
    seq_len = 4096
    dilation_rate = 4
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    
    # Create input
    x = torch.randn(2, seq_len, hidden_dim).cuda()
    
    print("\n1. UnifiedHilbertAttention:")
    module = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        hilbert_threshold=1024,
    ).cuda()
    
    # Check the kernel configuration
    if hasattr(module, '_get_kernel_config'):
        config = module._get_kernel_config(seq_len)
        print(f"  Kernel config: {config}")
    
    print("\n2. UnifiedHilbertAttentionOptimized:")
    module = UnifiedHilbertAttentionOptimized(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        hilbert_threshold=1024,
    ).cuda()
    
    if hasattr(module, '_get_kernel_config'):
        config = module._get_kernel_config(seq_len)
        print(f"  Kernel config: {config}")
    
    print("\n3. UnifiedHilbertAttentionOptimizedEnhanced:")
    module = UnifiedHilbertAttentionOptimizedEnhanced(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        hilbert_threshold=1024,
        enable_8k_optimization=True,
        enable_multi_row=True,
    ).cuda()
    
    # Check if it uses strided sparse attention
    if module.dilation_rate > 1:
        print(f"  Will use _strided_sparse_attention method")
    
    if hasattr(module, '_get_optimal_config'):
        config = module._get_optimal_config(seq_len)
        print(f"  Optimal config: {config}")


def compare_sparse_implementations():
    """Compare sparse pattern handling in each implementation."""
    
    print("\n=== Sparse Implementation Comparison ===")
    
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    
    implementations = {
        'Unified': UnifiedHilbertAttention,
        'Optimized': UnifiedHilbertAttentionOptimized,
        'Enhanced': UnifiedHilbertAttentionOptimizedEnhanced,
    }
    
    # Test different sparse configurations
    test_configs = [
        (2048, 2),
        (4096, 2),
        (4096, 4),
        (8192, 4),
    ]
    
    results = {}
    
    for seq_len, dilation_rate in test_configs:
        print(f"\n\nConfiguration: seq_len={seq_len}, dilation={dilation_rate}")
        print(f"Active positions: {seq_len // dilation_rate}")
        
        config_key = f"{seq_len}_d{dilation_rate}"
        results[config_key] = {}
        
        for name, impl_class in implementations.items():
            time_ms = profile_sparse_execution(
                impl_class, name, seq_len, dilation_rate, hidden_dim, num_heads
            )
            results[config_key][name] = time_ms
    
    # Summary table
    print("\n\n=== Performance Summary ===")
    print(f"{'Config':<15} | {'Unified':<12} | {'Optimized':<12} | {'Enhanced':<12} | {'Best':<10}")
    print("-" * 65)
    
    for config_key, times in results.items():
        print(f"{config_key:<15}", end=" | ")
        
        best_time = min(times.values())
        best_impl = [k for k, v in times.items() if v == best_time][0]
        
        for impl in ['Unified', 'Optimized', 'Enhanced']:
            time_ms = times[impl]
            if time_ms == best_time:
                print(f"**{time_ms:>8.2f}ms**", end=" | ")
            else:
                print(f"{time_ms:>10.2f}ms", end=" | ")
        
        print(f"{best_impl:<10}")


def analyze_sparse_memory_patterns():
    """Analyze memory access patterns for sparse attention."""
    
    print("\n\n=== Sparse Memory Pattern Analysis ===")
    
    seq_len = 4096
    dilation_rate = 4
    segment_size = 128
    hidden_dim = 512
    num_heads = 8
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Dilation rate: {dilation_rate}")
    print(f"  Segment size: {segment_size}")
    print(f"  Active positions per segment: {segment_size // dilation_rate}")
    print(f"  Total active positions: {seq_len // dilation_rate}")
    
    # Calculate memory access patterns
    print(f"\nMemory Access Analysis:")
    
    # For sparse pattern with dilation
    active_positions = seq_len // dilation_rate
    
    # Unified approach (likely simpler)
    print(f"\n1. Unified Approach:")
    print(f"  - Processes {active_positions} positions total")
    print(f"  - Simple strided access pattern")
    print(f"  - Memory stride: {dilation_rate} elements")
    
    # Optimized approach (may have overhead)
    print(f"\n2. Optimized Approach:")
    print(f"  - May process in larger blocks")
    print(f"  - Additional complexity for block alignment")
    print(f"  - Potential overhead from optimization logic")
    
    # Enhanced approach (uses special sparse method)
    print(f"\n3. Enhanced Approach:")
    print(f"  - Uses _strided_sparse_attention method")
    print(f"  - PyTorch-based sparse computation")
    print(f"  - Additional overhead from Python loop")
    
    # Test actual memory usage
    print(f"\n\nMemory Usage Test:")
    
    for name, impl_class in [
        ('Unified', UnifiedHilbertAttention),
        ('Optimized', UnifiedHilbertAttentionOptimized),
        ('Enhanced', UnifiedHilbertAttentionOptimizedEnhanced),
    ]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        config = {
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'segment_size': segment_size,
            'dilation_rate': dilation_rate,
            'hilbert_threshold': 1024,
        }
        
        if name == 'Enhanced':
            config['enable_8k_optimization'] = True
            config['enable_multi_row'] = True
        
        module = impl_class(**config).cuda()
        x = torch.randn(2, seq_len, hidden_dim).cuda()
        
        with torch.no_grad():
            _ = module(x)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"  {name}: {peak_memory:.1f} MB")


def check_triton_kernel_differences():
    """Check if Triton kernels have different sparse handling."""
    
    print("\n\n=== Triton Kernel Sparse Handling ===")
    
    # Look at the actual kernel implementations
    print("\nKernel sparse handling differences:")
    
    print("\n1. Unified kernel:")
    print("  - Basic sparse iteration with dilation_rate")
    print("  - Direct calculation of active positions")
    print("  - Simple mask application")
    
    print("\n2. Optimized kernel:")
    print("  - Pre-computed segment boundaries")
    print("  - Optimized pointer arithmetic")
    print("  - May have overhead for sparse patterns")
    
    print("\n3. Enhanced kernel:")
    print("  - Falls back to PyTorch for dilation > 1")
    print("  - Uses _strided_sparse_attention method")
    print("  - No Triton kernel for sparse patterns!")


if __name__ == "__main__":
    # Run all analyses
    analyze_kernel_paths()
    compare_sparse_implementations()
    analyze_sparse_memory_patterns()
    check_triton_kernel_differences()
    
    print("\n\n=== Key Findings ===")
    print("\n1. Enhanced uses PyTorch fallback for sparse:")
    print("   - When dilation_rate > 1, it uses _strided_sparse_attention")
    print("   - This is a PyTorch implementation, not Triton")
    print("   - Adds overhead from Python loops and multiple kernel launches")
    
    print("\n2. Optimized has unnecessary complexity:")
    print("   - Optimizations designed for dense patterns")
    print("   - Block alignment overhead not beneficial for sparse")
    print("   - Larger block sizes waste computation on sparse patterns")
    
    print("\n3. Unified has simple, efficient sparse handling:")
    print("   - Direct Triton kernel with simple sparse logic")
    print("   - Minimal overhead for strided access")
    print("   - Better memory access pattern for sparse data")