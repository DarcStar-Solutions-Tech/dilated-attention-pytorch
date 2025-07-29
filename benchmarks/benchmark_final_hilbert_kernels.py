#!/usr/bin/env python3
"""
Benchmark the three final Hilbert attention implementations after consolidation.
"""

import torch
import time
import sys
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

sys.path.append('..')

# Import the three remaining implementations
from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimized,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_implementation(
    impl_class: type,
    impl_name: str,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    segment_size: int,
    dilation_rate: int,
    warmup: int = 3,
    runs: int = 10,
) -> Optional[Tuple[float, float, bool]]:
    """Benchmark a single implementation.
    
    Returns:
        (time_ms, memory_mb, valid) or None if failed
    """
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Create module
        kwargs = {
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'segment_size': segment_size,
            'dilation_rate': dilation_rate,
            'hilbert_threshold': 1024,
        }
        
        # Add extra params for enhanced version
        if 'Enhanced' in impl_name:
            kwargs['enable_8k_optimization'] = True
            kwargs['enable_multi_row'] = True
            
        module = impl_class(**kwargs).cuda()
        module.eval()
        
        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                try:
                    _ = module(x)
                except Exception as e:
                    print(f"  ✗ Runtime error during warmup: {str(e)}")
                    return None
            torch.cuda.synchronize()
        
        # Memory measurement
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(runs):
            with torch.no_grad():
                out = module(x)
            torch.cuda.synchronize()
            
        end = time.perf_counter()
        
        avg_time = (end - start) / runs * 1000  # ms
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # Check if output is valid
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("  ⚠ Warning: Output contains NaN or Inf")
            return avg_time, peak_memory, False
            
        return avg_time, peak_memory, True
        
    except Exception as e:
        print(f"  ✗ Failed to benchmark: {str(e)}")
        return None


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks on all three implementations."""
    
    # Test parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    
    # Implementations to test
    implementations = {
        'Unified': UnifiedHilbertAttention,
        'UnifiedOptimized': UnifiedHilbertAttentionOptimized,
        'UnifiedOptimizedEnhanced': UnifiedHilbertAttentionOptimizedEnhanced,
    }
    
    # Test configurations
    test_configs = [
        # (seq_len, dilation_rate, description)
        (512, 1, "Dense 512"),
        (1024, 1, "Dense 1K"),
        (2048, 1, "Dense 2K"),
        (4096, 1, "Dense 4K"),
        (8192, 1, "Dense 8K"),
        (16384, 1, "Dense 16K"),
        (2048, 2, "Sparse 2K (d=2)"),
        (4096, 2, "Sparse 4K (d=2)"),
        (8192, 2, "Sparse 8K (d=2)"),
        (4096, 4, "Sparse 4K (d=4)"),
        (8192, 4, "Sparse 8K (d=4)"),
    ]
    
    print("=== Final Hilbert Kernel Benchmarks ===")
    print(f"Batch: {batch_size}, Hidden: {hidden_dim}, Heads: {num_heads}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print()
    
    # Results storage
    results: Dict[str, Dict[str, Optional[Tuple[float, float]]]] = {}
    
    # Benchmark each implementation
    for impl_name, impl_class in implementations.items():
        print(f"\n--- {impl_name} Implementation ---")
        results[impl_name] = {}
        
        for seq_len, dilation_rate, desc in test_configs:
            print(f"\n{desc}:")
            
            result = benchmark_implementation(
                impl_class=impl_class,
                impl_name=impl_name,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            
            if result is not None:
                time_ms, memory_mb, valid = result
                results[impl_name][desc] = (time_ms, memory_mb)
                status = "✓" if valid else "⚠"
                print(f"  {status} Time: {time_ms:.2f}ms, Memory: {memory_mb:.1f}MB")
            else:
                results[impl_name][desc] = None
                print("  ✗ Failed")
    
    # Print comparison table
    print("\n\n=== Performance Comparison Table ===")
    print(f"{'Config':<20} | {'Unified':<20} | {'Optimized':<20} | {'Enhanced':<20}")
    print("-" * 85)
    
    for seq_len, dilation_rate, desc in test_configs:
        print(f"{desc:<20}", end=" | ")
        
        # Get baseline (Unified) time
        baseline_time = None
        if desc in results['Unified'] and results['Unified'][desc] is not None:
            baseline_time = results['Unified'][desc][0]
        
        for impl_name in ['Unified', 'UnifiedOptimized', 'UnifiedOptimizedEnhanced']:
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                time_ms, memory_mb = results[impl_name][desc]
                if baseline_time and baseline_time > 0:
                    speedup = baseline_time / time_ms
                    print(f"{time_ms:>6.2f}ms ({speedup:>4.2f}x)", end=" | ")
                else:
                    print(f"{time_ms:>6.2f}ms", end=" | ")
            else:
                print(f"{'Failed':<20}", end=" | ")
        print()
    
    # Memory comparison table
    print("\n\n=== Memory Usage Comparison ===")
    print(f"{'Config':<20} | {'Unified':<15} | {'Optimized':<15} | {'Enhanced':<15}")
    print("-" * 70)
    
    for seq_len, dilation_rate, desc in test_configs:
        print(f"{desc:<20}", end=" | ")
        
        for impl_name in ['Unified', 'UnifiedOptimized', 'UnifiedOptimizedEnhanced']:
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                _, memory_mb = results[impl_name][desc]
                print(f"{memory_mb:>10.1f} MB", end=" | ")
            else:
                print(f"{'N/A':<15}", end=" | ")
        print()
    
    # Calculate average speedups
    print("\n\n=== Average Performance Summary ===")
    
    for pattern in ['Dense', 'Sparse']:
        print(f"\n{pattern} Patterns:")
        
        for impl_name in ['UnifiedOptimized', 'UnifiedOptimizedEnhanced']:
            speedups = []
            
            for desc, data in results[impl_name].items():
                if pattern in desc and data is not None:
                    baseline_key = desc
                    if baseline_key in results['Unified'] and results['Unified'][baseline_key] is not None:
                        baseline_time = results['Unified'][baseline_key][0]
                        impl_time = data[0]
                        if baseline_time > 0:
                            speedups.append(baseline_time / impl_time)
            
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"  {impl_name}: {avg_speedup:.2f}x average speedup")
    
    # Special tests
    print("\n\n=== Special Optimizations ===")
    
    # Test 8K optimization
    if 'Dense 8K' in results['UnifiedOptimizedEnhanced']:
        enhanced_time = results['UnifiedOptimizedEnhanced']['Dense 8K'][0]
        optimized_time = results['UnifiedOptimized']['Dense 8K'][0]
        
        if enhanced_time and optimized_time:
            print(f"\n8K Sequence Optimization:")
            print(f"  Optimized: {optimized_time:.2f}ms")
            print(f"  Enhanced (with 8K opt): {enhanced_time:.2f}ms")
            print(f"  Improvement: {optimized_time / enhanced_time:.2f}x")
    
    # Plot results
    plot_benchmark_results(results, test_configs)
    
    return results


def plot_benchmark_results(results: Dict, test_configs: list):
    """Create visualization of benchmark results."""
    
    # Prepare data for plotting
    dense_configs = [(s, d, desc) for s, d, desc in test_configs if d == 1]
    sparse_configs = [(s, d, desc) for s, d, desc in test_configs if d > 1]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Dense performance comparison
    ax1.set_title('Dense Attention Performance')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_yscale('log')
    
    for impl_name, marker in [('Unified', 'o'), ('UnifiedOptimized', 's'), ('UnifiedOptimizedEnhanced', '^')]:
        seq_lens = []
        times = []
        
        for seq_len, _, desc in dense_configs:
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                seq_lens.append(seq_len)
                times.append(results[impl_name][desc][0])
        
        if seq_lens:
            ax1.plot(seq_lens, times, marker=marker, label=impl_name, linewidth=2, markersize=8)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sparse performance comparison
    ax2.set_title('Sparse Attention Performance')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Time (ms)')
    
    sparse_labels = [desc for _, _, desc in sparse_configs]
    x_pos = range(len(sparse_labels))
    
    width = 0.25
    for i, impl_name in enumerate(['Unified', 'UnifiedOptimized', 'UnifiedOptimizedEnhanced']):
        times = []
        for _, _, desc in sparse_configs:
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                times.append(results[impl_name][desc][0])
            else:
                times.append(0)
        
        ax2.bar([p + width * (i - 1) for p in x_pos], times, width, label=impl_name)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sparse_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory usage comparison
    ax3.set_title('Memory Usage vs Sequence Length')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Memory (MB)')
    
    for impl_name, marker in [('Unified', 'o'), ('UnifiedOptimized', 's'), ('UnifiedOptimizedEnhanced', '^')]:
        seq_lens = []
        memories = []
        
        for seq_len, _, desc in dense_configs:
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                seq_lens.append(seq_len)
                memories.append(results[impl_name][desc][1])
        
        if seq_lens:
            ax3.plot(seq_lens, memories, marker=marker, label=impl_name, linewidth=2, markersize=8)
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speedup comparison
    ax4.set_title('Speedup vs Baseline (Unified)')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Speedup Factor')
    
    all_labels = [desc for _, _, desc in test_configs[:8]]  # First 8 configs
    x_pos = range(len(all_labels))
    
    for impl_name in ['UnifiedOptimized', 'UnifiedOptimizedEnhanced']:
        speedups = []
        for _, _, desc in test_configs[:8]:
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                if desc in results['Unified'] and results['Unified'][desc] is not None:
                    baseline_time = results['Unified'][desc][0]
                    impl_time = results[impl_name][desc][0]
                    speedups.append(baseline_time / impl_time)
                else:
                    speedups.append(1.0)
            else:
                speedups.append(1.0)
        
        ax4.plot(x_pos, speedups, marker='o', label=impl_name, linewidth=2, markersize=8)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(all_labels, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hilbert_kernel_final_benchmarks.png', dpi=150)
    print("\n✓ Benchmark plots saved to hilbert_kernel_final_benchmarks.png")


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    # Save detailed results
    with open('hilbert_kernel_final_benchmarks.txt', 'w') as f:
        f.write("=== Detailed Benchmark Results ===\n\n")
        
        for impl_name, impl_results in results.items():
            f.write(f"\n{impl_name}:\n")
            for config, data in impl_results.items():
                if data is not None:
                    time_ms, memory_mb = data
                    f.write(f"  {config}: {time_ms:.2f}ms, {memory_mb:.1f}MB\n")
                else:
                    f.write(f"  {config}: Failed\n")
    
    print("\n✓ Detailed results saved to hilbert_kernel_final_benchmarks.txt")