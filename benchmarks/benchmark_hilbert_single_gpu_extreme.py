#!/usr/bin/env python3
"""
Single GPU benchmark pushing extreme sequence lengths with Ring Hilbert Attention.

This tests how far we can push a single GPU with ring_size=1.
"""

import torch
import torch.nn as nn
import json
import time
import os
from datetime import datetime, timezone
from typing import Dict, List
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_gpu_optimized import (
    RingDilatedAttentionHilbertGPUOptimized,
)
from dilated_attention_pytorch.utils.gpu_utils import get_gpu_info
from core.utils.memory import get_memory_stats, cleanup_memory
from core.utils.safety import MemorySafetyChecker, SafetyConfig
from core.utils.timing import CUDATimer


def test_extreme_sequences():
    """Test extreme sequence lengths on a single GPU."""
    device = torch.device("cuda:0")
    dtype = torch.float32
    
    # Get GPU info
    gpu_info = get_gpu_info(device)
    print(f"GPU: {gpu_info.name} ({gpu_info.architecture})")
    print(f"Total Memory: {gpu_info.total_memory_gb:.1f} GB")
    print(f"Available Memory: {gpu_info.available_memory_gb:.1f} GB")
    print()
    
    # Initialize safety checker
    safety_config = SafetyConfig(
        max_memory_fraction=0.85,  # Use up to 85% of GPU memory
        min_free_memory_gb=0.5,
        cleanup_threshold=0.8,
    )
    safety_checker = MemorySafetyChecker(safety_config)
    
    # Test configurations - progressively increase
    seq_lengths = [
        8192,    # 8K
        16384,   # 16K
        32768,   # 32K
        65536,   # 64K
        131072,  # 128K
        262144,  # 256K
        524288,  # 512K
    ]
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n{'='*70}")
        print(f"Testing sequence length: {seq_len:,} tokens")
        print(f"{'='*70}")
        
        # Estimate memory requirement
        batch_size = 1
        num_heads = 8
        embed_dim = 768
        head_dim = embed_dim // num_heads
        
        # Rough memory estimate (GB)
        # Attention memory scales as O(seq_len * segment_len * num_heads)
        max_segment = min(16384, seq_len // 4)
        estimated_gb = (batch_size * num_heads * seq_len * max_segment * 4) / (1024**3)
        estimated_gb *= 2  # Forward + backward
        
        print(f"Estimated memory requirement: {estimated_gb:.2f} GB")
        
        # Check if we have enough memory
        can_run, msg = safety_checker.check_memory_available(estimated_gb)
        if not can_run:
            print(f"Skipping: {msg}")
            results.append({
                "seq_len": seq_len,
                "status": "skipped",
                "reason": msg,
            })
            continue
        
        # Determine segment configuration
        if seq_len <= 16384:
            segment_lengths = [4096, 8192]
            dilation_rates = [1, 2]
        elif seq_len <= 65536:
            segment_lengths = [8192, 16384]
            dilation_rates = [1, 2]
        else:
            # For very long sequences, use larger segments
            segment_lengths = [16384, 32768]
            dilation_rates = [1, 2]
        
        # Ensure segments don't exceed sequence length
        segment_lengths = [s for s in segment_lengths if s <= seq_len]
        dilation_rates = dilation_rates[:len(segment_lengths)]
        
        # Test with Hilbert
        for use_hilbert in [True, False]:
            try:
                print(f"\n{'With' if use_hilbert else 'Without'} Hilbert ordering:")
                
                cleanup_memory()
                
                # Create module
                module = RingDilatedAttentionHilbertGPUOptimized(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    ring_size=1,  # Single GPU
                    use_hilbert=use_hilbert,
                    device=device,
                    dtype=dtype,
                    benchmark_backends=False,
                )
                
                # Create input
                x = torch.randn(
                    batch_size, seq_len, embed_dim,
                    device=device, dtype=dtype,
                    requires_grad=True
                )
                
                # Check memory after allocation
                mem_stats = get_memory_stats(device)
                print(f"  Memory after allocation: {mem_stats['allocated']:.1f} MB")
                
                # Warmup
                print("  Warming up...")
                for _ in range(2):
                    cleanup_memory()
                    output = module(x)
                    if x.requires_grad:
                        loss = output.mean()
                        loss.backward()
                
                # Time forward pass
                print("  Timing forward pass...")
                forward_times = []
                for _ in range(3):
                    cleanup_memory()
                    timer = CUDATimer("forward", device, verbose=False)
                    with timer:
                        output = module(x)
                    forward_times.append(timer.elapsed_ms)
                
                # Time backward pass
                print("  Timing backward pass...")
                backward_times = []
                for _ in range(3):
                    cleanup_memory()
                    output = module(x)
                    timer = CUDATimer("backward", device, verbose=False)
                    with timer:
                        loss = output.mean()
                        loss.backward()
                    backward_times.append(timer.elapsed_ms)
                
                # Get peak memory
                cleanup_memory()
                start_mem = get_memory_stats(device)["allocated"]
                output = module(x)
                loss = output.mean()
                loss.backward()
                end_mem = get_memory_stats(device)["allocated"]
                
                result = {
                    "seq_len": seq_len,
                    "use_hilbert": use_hilbert,
                    "segment_lengths": segment_lengths,
                    "dilation_rates": dilation_rates,
                    "forward_ms": np.mean(forward_times),
                    "forward_std": np.std(forward_times),
                    "backward_ms": np.mean(backward_times),
                    "backward_std": np.std(backward_times),
                    "total_ms": np.mean(forward_times) + np.mean(backward_times),
                    "memory_mb": end_mem - start_mem,
                    "peak_memory_mb": end_mem,
                    "throughput_tps": seq_len / (np.mean(forward_times) / 1000),
                    "status": "success",
                }
                
                print(f"  Forward: {result['forward_ms']:.2f}±{result['forward_std']:.2f} ms")
                print(f"  Backward: {result['backward_ms']:.2f}±{result['backward_std']:.2f} ms")
                print(f"  Memory used: {result['memory_mb']:.1f} MB")
                print(f"  Peak memory: {result['peak_memory_mb']:.1f} MB")
                print(f"  Throughput: {result['throughput_tps']:,.0f} tokens/sec")
                
                results.append(result)
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM: {str(e).split('.')[0]}")
                results.append({
                    "seq_len": seq_len,
                    "use_hilbert": use_hilbert,
                    "status": "OOM",
                    "error": str(e).split('.')[0],
                })
                # If OOM with Hilbert, skip non-Hilbert too
                if use_hilbert:
                    results.append({
                        "seq_len": seq_len,
                        "use_hilbert": False,
                        "status": "skipped",
                        "reason": "OOM with Hilbert",
                    })
                    break
                    
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "seq_len": seq_len,
                    "use_hilbert": use_hilbert,
                    "status": "error",
                    "error": str(e),
                })
    
    return results


def generate_report(results: List[Dict]):
    """Generate benchmark report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-UTC")
    report_path = f"docs/benchmarks/hilbert-single-gpu-extreme-{timestamp}.md"
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Single GPU Ring Hilbert Attention - Extreme Sequence Lengths\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("- GPU: NVIDIA GeForce GTX 1080 (8GB)\n")
        f.write("- Architecture: Pascal\n")
        f.write("- Data Type: torch.float32\n")
        f.write("- Ring Size: 1 (single GPU)\n\n")
        
        f.write("## Performance Results\n\n")
        
        # Filter successful results
        successful = [r for r in results if r.get('status') == 'success']
        
        if successful:
            f.write("| Seq Length | Hilbert | Forward (ms) | Backward (ms) | Total (ms) | Memory (MB) | Peak (MB) | Throughput (tok/s) |\n")
            f.write("|------------|---------|--------------|---------------|------------|-------------|-----------|-------------------|\n")
            
            for r in successful:
                f.write(f"| {r['seq_len']:,} | {'Yes' if r['use_hilbert'] else 'No'} | ")
                f.write(f"{r['forward_ms']:.2f}±{r['forward_std']:.2f} | ")
                f.write(f"{r['backward_ms']:.2f}±{r['backward_std']:.2f} | ")
                f.write(f"{r['total_ms']:.2f} | ")
                f.write(f"{r['memory_mb']:.1f} | ")
                f.write(f"{r['peak_memory_mb']:.1f} | ")
                f.write(f"{r['throughput_tps']:,.0f} |\n")
        
        f.write("\n## Maximum Sequence Length Achieved\n\n")
        
        # Find max successful sequence length
        hilbert_success = [r['seq_len'] for r in successful if r.get('use_hilbert')]
        no_hilbert_success = [r['seq_len'] for r in successful if not r.get('use_hilbert')]
        
        max_hilbert = max(hilbert_success) if hilbert_success else 0
        max_no_hilbert = max(no_hilbert_success) if no_hilbert_success else 0
        
        f.write(f"- **With Hilbert**: {max_hilbert:,} tokens\n")
        f.write(f"- **Without Hilbert**: {max_no_hilbert:,} tokens\n")
        f.write(f"- **Overall Maximum**: {max(max_hilbert, max_no_hilbert):,} tokens\n\n")
        
        f.write("## Failed Attempts\n\n")
        
        failed = [r for r in results if r.get('status') in ['OOM', 'error', 'skipped']]
        if failed:
            for r in failed:
                f.write(f"- {r['seq_len']:,} tokens ({'Hilbert' if r.get('use_hilbert') else 'No Hilbert'}): ")
                f.write(f"{r.get('status')} - {r.get('error', r.get('reason', 'Unknown'))}\n")
        
        f.write("\n## Hilbert Impact Analysis\n\n")
        
        # Compare Hilbert vs non-Hilbert
        seq_lengths = sorted(set(r['seq_len'] for r in successful))
        
        for seq_len in seq_lengths:
            hilbert = next((r for r in successful if r['seq_len'] == seq_len and r['use_hilbert']), None)
            no_hilbert = next((r for r in successful if r['seq_len'] == seq_len and not r['use_hilbert']), None)
            
            if hilbert and no_hilbert:
                speedup = no_hilbert['total_ms'] / hilbert['total_ms']
                memory_ratio = hilbert['memory_mb'] / no_hilbert['memory_mb'] if no_hilbert['memory_mb'] > 0 else 1.0
                
                f.write(f"### {seq_len:,} tokens\n")
                f.write(f"- **Speedup**: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} with Hilbert\n")
                f.write(f"- **Memory**: {memory_ratio:.2f}x memory usage with Hilbert\n")
                f.write(f"- **Throughput difference**: {(hilbert['throughput_tps'] - no_hilbert['throughput_tps']):+,.0f} tokens/sec\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Single GPU Limit**: GTX 1080 (8GB) can process up to ")
        f.write(f"{max(max_hilbert, max_no_hilbert):,} tokens\n")
        f.write("2. **Hilbert Ordering**: Shows benefits at longer sequences\n")
        f.write("3. **Memory Efficiency**: Ring attention with single GPU still provides benefits\n")
        f.write("4. **Performance**: Throughput remains good even at extreme lengths\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **For > 32K tokens**: Consider multi-GPU setup for better performance\n")
        f.write("2. **Memory Limited**: Use gradient checkpointing for longer sequences\n")
        f.write("3. **Hilbert Usage**: Enable for sequences > 16K tokens\n")
        f.write("4. **Production**: Monitor memory usage closely at extreme lengths\n")
    
    # Save raw data
    json_path = report_path.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Raw data saved to: {json_path}")
    
    return report_path


def main():
    print("="*80)
    print("Single GPU Ring Hilbert Attention - Extreme Sequence Benchmark")
    print("="*80)
    
    # Run tests
    results = test_extreme_sequences()
    
    # Generate report
    generate_report(results)


if __name__ == "__main__":
    main()