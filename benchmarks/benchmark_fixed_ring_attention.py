#!/usr/bin/env python3
"""
Comprehensive benchmark of the FIXED Ring Attention implementations.

This script tests:
1. Memory scaling with ring size
2. Maximum sequence lengths achievable
3. Performance characteristics
4. Comparison with standard attention and broken implementation
"""

import gc
import time
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention_fixed import RingDilatedAttentionFixed
from dilated_attention_pytorch.true_ring_dilated_attention import TrueRingDilatedAttention
from dilated_attention_pytorch.dilated_attention import DilatedAttention

# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager


def get_gpu_memory_info():
    """Get current GPU memory stats."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - allocated
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }


def benchmark_implementation(
    impl_name: str,
    module: torch.nn.Module,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    num_runs: int = 3,
    warmup: int = 1,
) -> Dict:
    """Benchmark a single implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    result = {
        "implementation": impl_name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "success": False,
        "error": None,
        "time_ms": None,
        "memory_gb": None,
        "throughput_tokens_per_sec": None,
    }
    
    try:
        # Move module to device
        module = module.to(device, dtype)
        
        # Clear memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = module(q, k, v)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                output = module(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        # Get memory stats
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_memory_gb = 0
        
        # Calculate metrics
        avg_time_ms = sum(times) / len(times)
        throughput = (seq_len * batch_size) / (avg_time_ms / 1000)
        
        result.update({
            "success": True,
            "time_ms": avg_time_ms,
            "memory_gb": peak_memory_gb,
            "throughput_tokens_per_sec": throughput,
            "times": times,
        })
        
        print(f"  ✓ {impl_name}: {avg_time_ms:.1f}ms, {peak_memory_gb:.3f}GB, {throughput/1e6:.2f}M tok/s")
        
    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            print(f"  ✗ {impl_name}: OOM")
            result["error"] = "OOM"
        else:
            print(f"  ✗ {impl_name}: {error_msg[:50]}...")
            result["error"] = error_msg
    
    finally:
        # Cleanup
        if 'q' in locals():
            del q, k, v
        if 'output' in locals():
            del output
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    return result


def test_memory_scaling():
    """Test memory scaling with different ring sizes."""
    print("\n" + "="*80)
    print("MEMORY SCALING TEST")
    print("="*80)
    
    results = []
    
    # Test configurations
    test_configs = [
        # (seq_len, ring_sizes_to_test)
        (4096, [1, 2, 4, 8]),
        (8192, [1, 2, 4, 8, 16]),
        (16384, [1, 4, 8, 16, 32]),
        (32768, [1, 8, 16, 32, 64]),
        (65536, [1, 16, 32, 64, 128]),
    ]
    
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]
    
    for seq_len, ring_sizes in test_configs:
        print(f"\nSequence length: {seq_len:,}")
        
        for ring_size in ring_sizes:
            print(f"\n  Ring size: {ring_size}")
            
            # Test each implementation
            implementations = []
            
            # Current (broken) implementation
            if seq_len >= 4096:  # Skip small sequences that don't work
                implementations.append((
                    "RingDilatedAttention (current)",
                    RingDilatedAttention(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        ring_size=ring_size,
                    )
                ))
            
            # Fixed implementation
            implementations.append((
                "RingDilatedAttentionFixed",
                RingDilatedAttentionFixed(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=ring_size,
                )
            ))
            
            # True ring attention (reference)
            implementations.append((
                "TrueRingDilatedAttention",
                TrueRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=ring_size,
                )
            ))
            
            for impl_name, module in implementations:
                result = benchmark_implementation(impl_name, module, seq_len)
                result["ring_size"] = ring_size
                results.append(result)
    
    return results


def test_extreme_sequences():
    """Test extreme sequence lengths with optimal ring sizes."""
    print("\n" + "="*80)
    print("EXTREME SEQUENCE LENGTH TEST")
    print("="*80)
    
    results = []
    
    # Get available GPU memory
    gpu_info = get_gpu_memory_info()
    available_gb = gpu_info["total"] * 0.8  # Use 80% of total
    
    print(f"Available GPU memory: {available_gb:.1f}GB")
    
    # Test configurations with auto-computed ring sizes
    test_sequences = [
        8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
    ]
    
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]
    
    for seq_len in test_sequences:
        print(f"\nTesting sequence length: {seq_len:,}")
        
        # Auto-compute optimal ring size based on available memory
        # Rough estimate: need ~4 bytes per token for Q, K, V, output
        bytes_per_token = 4 * 8 * 64 * 2  # batch * heads * dim * float16
        total_bytes = seq_len * bytes_per_token
        total_gb = total_bytes / (1024**3)
        
        # Estimate ring size needed to fit in memory
        if total_gb > available_gb:
            ring_size = int(np.ceil(total_gb / available_gb))
            # Round up to power of 2
            ring_size = 2 ** int(np.ceil(np.log2(ring_size)))
        else:
            ring_size = 1
        
        print(f"  Estimated memory: {total_gb:.1f}GB")
        print(f"  Using ring_size: {ring_size}")
        
        # Test only the fixed implementations
        implementations = [
            ("RingDilatedAttentionFixed", RingDilatedAttentionFixed(
                segment_lengths=[min(s, seq_len) for s in segment_lengths],
                dilation_rates=dilation_rates,
                ring_size=ring_size,
            )),
            ("TrueRingDilatedAttention", TrueRingDilatedAttention(
                segment_lengths=[min(s, seq_len) for s in segment_lengths],
                dilation_rates=dilation_rates,
                ring_size=ring_size,
            )),
        ]
        
        for impl_name, module in implementations:
            result = benchmark_implementation(impl_name, module, seq_len, num_runs=2)
            result["ring_size"] = ring_size
            result["estimated_memory_gb"] = total_gb
            results.append(result)
            
            # Stop testing this sequence if we hit OOM
            if result["error"] == "OOM":
                print(f"  Stopping at {seq_len:,} due to OOM")
                break
    
    return results


def test_performance_comparison():
    """Compare performance across implementations."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TEST")
    print("="*80)
    
    results = []
    
    # Test different sequence lengths with fixed ring size
    test_configs = [
        (1024, 1),
        (2048, 1),
        (4096, 1),
        (8192, 1),
        (4096, 4),
        (8192, 4),
        (16384, 4),
        (8192, 8),
        (16384, 8),
        (32768, 8),
    ]
    
    segment_lengths = [512, 1024, 2048]
    dilation_rates = [1, 2, 4]
    
    for seq_len, ring_size in test_configs:
        print(f"\nSeq {seq_len:,}, Ring {ring_size}:")
        
        # Test implementations
        implementations = []
        
        # Standard dilated attention (baseline)
        if ring_size == 1:
            implementations.append((
                "DilatedAttention (baseline)",
                DilatedAttention(
                    segment_lengths=[min(s, seq_len) for s in segment_lengths],
                    dilation_rates=dilation_rates,
                )
            ))
        
        # Current implementation (if it works)
        if seq_len >= 4096:
            implementations.append((
                "RingDilatedAttention (current)",
                RingDilatedAttention(
                    segment_lengths=[min(s, seq_len) for s in segment_lengths],
                    dilation_rates=dilation_rates,
                    ring_size=ring_size,
                )
            ))
        
        # Fixed implementations
        implementations.extend([
            ("RingDilatedAttentionFixed", RingDilatedAttentionFixed(
                segment_lengths=[min(s, seq_len) for s in segment_lengths],
                dilation_rates=dilation_rates,
                ring_size=ring_size,
            )),
            ("TrueRingDilatedAttention", TrueRingDilatedAttention(
                segment_lengths=[min(s, seq_len) for s in segment_lengths],
                dilation_rates=dilation_rates,
                ring_size=ring_size,
            )),
        ])
        
        for impl_name, module in implementations:
            result = benchmark_implementation(impl_name, module, seq_len)
            result["ring_size"] = ring_size
            results.append(result)
    
    return results


def create_plots(memory_results, extreme_results, perf_results, output_dir):
    """Create visualization plots."""
    
    # 1. Memory scaling plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    impl_types = ["RingDilatedAttention (current)", "RingDilatedAttentionFixed", "TrueRingDilatedAttention"]
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']
    
    ax1.set_title("Memory Usage vs Sequence Length")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Memory (GB)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    
    for impl, color, marker in zip(impl_types, colors, markers):
        impl_data = [r for r in memory_results if r["implementation"] == impl and r["success"]]
        if impl_data:
            # Group by sequence length, take ring_size=1
            seq_lens = sorted(set(r["seq_len"] for r in impl_data if r["ring_size"] == 1))
            memories = [next((r["memory_gb"] for r in impl_data 
                            if r["seq_len"] == s and r["ring_size"] == 1), None) 
                       for s in seq_lens]
            memories = [m for m in memories if m is not None]
            seq_lens = seq_lens[:len(memories)]
            
            if seq_lens and memories:
                ax1.plot(seq_lens, memories, f'{marker}-', label=impl, color=color, markersize=8)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory reduction with ring size
    ax2.set_title("Memory Reduction with Ring Size")
    ax2.set_xlabel("Ring Size")
    ax2.set_ylabel("Memory (GB)")
    ax2.set_xscale("log")
    
    # Plot for seq_len=16384
    target_seq = 16384
    for impl, color, marker in zip(impl_types[1:], colors[1:], markers[1:]):  # Skip broken impl
        impl_data = [r for r in memory_results 
                    if r["implementation"] == impl and r["seq_len"] == target_seq and r["success"]]
        if impl_data:
            ring_sizes = sorted(set(r["ring_size"] for r in impl_data))
            memories = [next((r["memory_gb"] for r in impl_data if r["ring_size"] == rs), None) 
                       for rs in ring_sizes]
            memories = [m for m in memories if m is not None]
            ring_sizes = ring_sizes[:len(memories)]
            
            if ring_sizes and memories:
                ax2.plot(ring_sizes, memories, f'{marker}-', label=f"{impl} (16K seq)", 
                        color=color, markersize=8)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = time.strftime("%Y-%m-%d-%H%M-UTC", time.gmtime())
    plot_path = f"{output_dir}/fixed-ring-attention-benchmark-{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Maximum sequence length achieved
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title("Maximum Sequence Length Achieved")
    ax.set_xlabel("Implementation")
    ax.set_ylabel("Maximum Sequence Length")
    ax.set_yscale("log")
    
    # Find max sequence for each implementation
    max_seqs = {}
    for impl in ["RingDilatedAttentionFixed", "TrueRingDilatedAttention"]:
        successful = [r["seq_len"] for r in extreme_results 
                     if r["implementation"] == impl and r["success"]]
        if successful:
            max_seqs[impl] = max(successful)
    
    if max_seqs:
        impls = list(max_seqs.keys())
        max_lens = list(max_seqs.values())
        bars = ax.bar(impls, max_lens)
        
        # Add value labels on bars
        for bar, val in zip(bars, max_lens):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}', ha='center', va='bottom')
    
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot2_path = f"{output_dir}/max-sequence-lengths-{timestamp}.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [plot_path, plot2_path]


def main():
    """Run comprehensive benchmarks."""
    
    # Setup output manager
    output_manager = BenchmarkOutputManager(
        benchmark_type="fixed-ring-attention-comprehensive",
        parameters={
            "tests": ["memory_scaling", "extreme_sequences", "performance"],
            "implementations": ["current", "fixed", "true"],
        }
    )
    
    # Run tests
    print("Running Fixed Ring Attention Comprehensive Benchmarks")
    print("="*80)
    
    # 1. Memory scaling test
    memory_results = test_memory_scaling()
    output_manager.add_result("memory_scaling_results", memory_results)
    
    # 2. Extreme sequence test
    extreme_results = test_extreme_sequences()
    output_manager.add_result("extreme_sequence_results", extreme_results)
    
    # 3. Performance comparison
    perf_results = test_performance_comparison()
    output_manager.add_result("performance_comparison_results", perf_results)
    
    # Create plots
    output_dir = "docs/benchmarks"
    plot_paths = create_plots(memory_results, extreme_results, perf_results, output_dir)
    output_manager.add_result("plot_paths", plot_paths)
    
    # Analysis summary
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*80)
    
    # Memory scaling analysis
    print("\n1. MEMORY SCALING:")
    for seq_len in [4096, 8192, 16384]:
        print(f"\n  Sequence {seq_len:,}:")
        for impl in ["RingDilatedAttention (current)", "RingDilatedAttentionFixed"]:
            impl_data = [r for r in memory_results 
                        if r["implementation"] == impl and r["seq_len"] == seq_len and r["success"]]
            if impl_data:
                ring1 = next((r for r in impl_data if r["ring_size"] == 1), None)
                ring8 = next((r for r in impl_data if r["ring_size"] == 8), None)
                if ring1 and ring8:
                    reduction = (1 - ring8["memory_gb"] / ring1["memory_gb"]) * 100
                    print(f"    {impl}: {reduction:.1f}% memory reduction with ring_size=8")
    
    # Maximum sequence achieved
    print("\n2. MAXIMUM SEQUENCES ACHIEVED:")
    for impl in ["RingDilatedAttentionFixed", "TrueRingDilatedAttention"]:
        successful = [r for r in extreme_results if r["implementation"] == impl and r["success"]]
        if successful:
            max_seq = max(r["seq_len"] for r in successful)
            max_result = next(r for r in successful if r["seq_len"] == max_seq)
            print(f"  {impl}: {max_seq:,} tokens")
            print(f"    Ring size: {max_result['ring_size']}")
            print(f"    Memory: {max_result['memory_gb']:.2f}GB")
            print(f"    Throughput: {max_result['throughput_tokens_per_sec']/1e6:.2f}M tok/s")
    
    # Performance analysis
    print("\n3. PERFORMANCE CHARACTERISTICS:")
    # Compare at seq_len=8192
    perf_8k = [r for r in perf_results if r["seq_len"] == 8192 and r["success"]]
    if perf_8k:
        baseline = next((r for r in perf_8k if "baseline" in r["implementation"]), None)
        if baseline:
            print(f"\n  At 8192 tokens (vs baseline {baseline['time_ms']:.1f}ms):")
            for r in perf_8k:
                if r != baseline:
                    overhead = (r["time_ms"] / baseline["time_ms"] - 1) * 100
                    print(f"    {r['implementation']}: {r['time_ms']:.1f}ms ({overhead:+.1f}% overhead)")
    
    # Save results
    json_path, md_path = output_manager.save_results()
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Plots: {', '.join(plot_paths)}")
    
    return output_manager


if __name__ == "__main__":
    main()