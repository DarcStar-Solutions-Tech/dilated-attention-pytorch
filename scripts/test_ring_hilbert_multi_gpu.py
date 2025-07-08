#!/usr/bin/env python3
"""
Simple test of Ring Hilbert Attention with multiple GPUs.

Run with:
torchrun --nproc_per_node=2 scripts/test_ring_hilbert_multi_gpu.py
"""

import os
import torch
import torch.distributed as dist
from datetime import datetime, timezone

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_gpu_optimized import (
    RingDilatedAttentionHilbertGPUOptimized,
)

# Add benchmarks to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmarks'))
from core.utils.memory import get_memory_stats, cleanup_memory


def test_ring_attention():
    """Test ring attention with distributed setup."""
    # Initialize distributed if not already
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    
    # Set device
    torch.cuda.set_device(device)
    
    print(f"[Rank {rank}] Initialized on {torch.cuda.get_device_name(device)}")
    
    # Test configurations
    test_configs = [
        # (seq_len, batch_size, num_heads, embed_dim)
        (16384, 1, 8, 768),   # 16K tokens
        (32768, 1, 8, 768),   # 32K tokens
        (65536, 1, 8, 768),   # 64K tokens
        (131072, 1, 8, 768),  # 128K tokens
        (262144, 1, 8, 768),  # 256K tokens
    ]
    
    results = []
    
    for seq_len, batch_size, num_heads, embed_dim in test_configs:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Testing {seq_len:,} tokens across {world_size} GPUs")
            print(f"Tokens per GPU: {seq_len // world_size:,}")
            print(f"{'='*70}")
        
        # Determine segments
        if seq_len <= 32768:
            segment_lengths = [4096, 8192, 16384]
            dilation_rates = [1, 2, 4]
        else:
            segment_lengths = [8192, 16384, 32768]
            dilation_rates = [1, 2, 4]
        
        # Filter segments that don't exceed sequence length
        segment_lengths = [s for s in segment_lengths if s <= seq_len]
        dilation_rates = dilation_rates[:len(segment_lengths)]
        
        for use_hilbert in [True, False]:
            try:
                cleanup_memory()
                dist.barrier()
                
                if rank == 0:
                    print(f"\n{'With' if use_hilbert else 'Without'} Hilbert ordering:")
                
                # Create module
                module = RingDilatedAttentionHilbertGPUOptimized(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    ring_size=world_size,
                    use_hilbert=use_hilbert,
                    device=device,
                    dtype=torch.float32,
                )
                
                # Create input
                x = torch.randn(
                    batch_size, seq_len, embed_dim,
                    device=device, dtype=torch.float32,
                    requires_grad=True
                )
                
                # Get initial memory
                start_mem = get_memory_stats(device)["allocated"]
                
                # Forward pass
                dist.barrier()
                start_time = datetime.now(timezone.utc)
                
                output = module(x)
                
                dist.barrier()
                forward_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                # Backward pass
                start_time = datetime.now(timezone.utc)
                
                loss = output.mean()
                loss.backward()
                
                dist.barrier()
                backward_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                # Get memory usage
                end_mem = get_memory_stats(device)["allocated"]
                memory_mb = end_mem - start_mem
                
                if rank == 0:
                    throughput = (batch_size * seq_len) / (forward_time / 1000)
                    
                    result = {
                        "seq_len": seq_len,
                        "world_size": world_size,
                        "use_hilbert": use_hilbert,
                        "forward_ms": forward_time,
                        "backward_ms": backward_time,
                        "total_ms": forward_time + backward_time,
                        "memory_mb_per_gpu": memory_mb,
                        "throughput_tps": throughput,
                    }
                    
                    print(f"  Forward: {forward_time:.2f} ms")
                    print(f"  Backward: {backward_time:.2f} ms")
                    print(f"  Memory per GPU: {memory_mb:.1f} MB")
                    print(f"  Throughput: {throughput:,.0f} tokens/sec")
                    
                    results.append(result)
                
            except torch.cuda.OutOfMemoryError as e:
                if rank == 0:
                    print(f"  OOM: {str(e).split('.')[0]}")
                dist.barrier()
                break
                
            except Exception as e:
                if rank == 0:
                    print(f"  Error: {e}")
                dist.barrier()
    
    # Save results from rank 0
    if rank == 0 and results:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-UTC")
        report_path = f"ring-hilbert-multi-gpu-test-{timestamp}.txt"
        
        with open(report_path, "w") as f:
            f.write("Ring Hilbert Multi-GPU Test Results\n")
            f.write("="*70 + "\n\n")
            
            for r in results:
                f.write(f"Sequence Length: {r['seq_len']:,} tokens\n")
                f.write(f"Hilbert: {'Yes' if r['use_hilbert'] else 'No'}\n")
                f.write(f"Forward: {r['forward_ms']:.2f} ms\n")
                f.write(f"Backward: {r['backward_ms']:.2f} ms\n")
                f.write(f"Memory per GPU: {r['memory_mb_per_gpu']:.1f} MB\n")
                f.write(f"Throughput: {r['throughput_tps']:,.0f} tokens/sec\n")
                f.write("-"*70 + "\n")
        
        print(f"\nResults saved to: {report_path}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    test_ring_attention()