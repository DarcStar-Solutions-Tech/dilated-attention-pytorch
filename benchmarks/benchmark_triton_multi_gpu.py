#!/usr/bin/env python3
"""
Benchmark Triton integrated Ring Dilated Attention on multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import argparse
import os
from typing import List
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_triton_integrated import (
    RingDilatedAttentionTritonIntegrated,
)
from dilated_attention_pytorch.ring_dilated_attention_simple_triton import (
    RingDilatedAttentionSimpleTriton,
)


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def benchmark_distributed_attention(
    rank: int,
    world_size: int,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    implementation: str,
    warmup_iters: int = 5,
    test_iters: int = 20,
):
    """Benchmark attention on a single GPU in distributed setting."""

    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create model based on implementation
    if implementation == "simple_no_hilbert":
        model = RingDilatedAttentionSimpleTriton(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            use_hilbert=False,
        )
        impl_name = "Simple (No Hilbert)"
    elif implementation == "simple_hilbert":
        model = RingDilatedAttentionSimpleTriton(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            use_hilbert=True,
        )
        impl_name = "Simple (Python Hilbert)"
    elif implementation == "triton_pytorch":
        model = RingDilatedAttentionTritonIntegrated(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            use_triton=False,
        )
        impl_name = "Triton Integrated (PyTorch)"
    elif implementation == "triton_kernel":
        model = RingDilatedAttentionTritonIntegrated(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            use_triton=True,
        )
        impl_name = "Triton Integrated (Triton)"
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    # Create input tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Synchronize before starting
    dist.barrier()

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    # Synchronize after warmup
    dist.barrier()
    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1024**3  # GB

    # Benchmark
    dist.barrier()
    start_event.record()

    with torch.no_grad():
        for _ in range(test_iters):
            output = model(q, k, v, is_causal=False)

    end_event.record()
    torch.cuda.synchronize()

    # Get timing
    forward_time = start_event.elapsed_time(end_event) / test_iters  # ms

    # Get memory usage
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    mem_used = peak_mem - initial_mem

    # Calculate throughput
    total_tokens = batch_size * seq_len * world_size  # Total across all GPUs
    throughput = (total_tokens / forward_time) * 1000  # tokens/sec

    # Gather results from all ranks
    results = {
        "rank": rank,
        "forward_time_ms": forward_time,
        "throughput_tokens_sec": throughput,
        "memory_gb": mem_used,
        "output_shape": list(output.shape),
    }

    # Only rank 0 prints results
    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Multi-GPU Benchmark Results: {impl_name}")
        print(f"{'=' * 80}")
        print(f"GPUs: {world_size}")
        print(f"Total Sequence Length: {seq_len * world_size}")
        print(f"Sequence per GPU: {seq_len}")
        print(f"Batch Size: {batch_size}")
        print(f"Forward Time: {forward_time:.2f} ms")
        print(f"Total Throughput: {throughput:,.0f} tokens/sec")
        print(f"Throughput per GPU: {throughput / world_size:,.0f} tokens/sec")
        print(f"Memory per GPU: {mem_used:.2f} GB")
        print(f"Output shape: {output.shape}")

    # Store results for comparison
    results_tensor = torch.tensor([forward_time], device=device)
    dist.all_reduce(results_tensor, op=dist.ReduceOp.AVG)

    cleanup()

    return results


def run_single_gpu_baseline(args):
    """Run single GPU baseline for comparison."""
    print(f"\n{'=' * 80}")
    print("Single GPU Baseline")
    print(f"{'=' * 80}")

    device = torch.device("cuda:0")

    # Test with full sequence on single GPU
    total_seq = args.seq_len * args.world_size

    print(f"\nTesting sequence length: {total_seq} on single GPU")

    try:
        model = RingDilatedAttentionSimpleTriton(
            segment_lengths=args.segment_lengths,
            dilation_rates=args.dilation_rates,
            dropout=0.0,
            ring_size=1,
            device=device,
            dtype=torch.float32,
            use_hilbert=False,
        )

        q = torch.randn(
            args.batch_size,
            total_seq,
            args.num_heads,
            args.head_dim,
            device=device,
            dtype=torch.float32,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()

        # Time
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        forward_time = (time.time() - start) * 100  # ms per iteration
        throughput = (args.batch_size * total_seq / forward_time) * 1000
        memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"Forward Time: {forward_time:.2f} ms")
        print(f"Throughput: {throughput:,.0f} tokens/sec")
        print(f"Memory: {memory:.2f} GB")

        return forward_time, throughput

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM: Cannot run {total_seq} sequence on single GPU")
            return None, None
        else:
            raise e


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Triton Benchmark")
    parser.add_argument(
        "--world_size", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--seq_len", type=int, default=8192, help="Sequence length per GPU"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--segment_lengths",
        type=int,
        nargs="+",
        default=[2048, 4096],
        help="Segment lengths for dilated attention",
    )
    parser.add_argument(
        "--dilation_rates",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Dilation rates for dilated attention",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--test_iters", type=int, default=20, help="Number of test iterations"
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="all",
        choices=[
            "all",
            "simple_no_hilbert",
            "simple_hilbert",
            "triton_pytorch",
            "triton_kernel",
        ],
        help="Which implementation to test",
    )

    args = parser.parse_args()

    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus < args.world_size:
        print(f"Requested {args.world_size} GPUs but only {available_gpus} available")
        args.world_size = available_gpus

    print(f"Testing with {args.world_size} GPUs")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    if args.world_size > 1:
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")

    # Run single GPU baseline first
    single_time, single_throughput = run_single_gpu_baseline(args)

    # Test implementations
    implementations = []
    if args.implementation == "all":
        # Test all implementations now that multi-GPU is fixed
        implementations = [
            "simple_no_hilbert",
            "simple_hilbert",
            "triton_pytorch",
            "triton_kernel",
        ]
    else:
        implementations = [args.implementation]

    _ = {}

    for impl in implementations:
        try:
            print(f"\nTesting {impl} on {args.world_size} GPUs...")

            # All implementations should now support multi-GPU
            # if impl.startswith('simple') and args.world_size > 1:
            #     print(f"Skipping {impl} - multi-GPU not implemented")
            #     continue

            # Spawn processes for distributed training
            mp.spawn(
                benchmark_distributed_attention,
                args=(
                    args.world_size,
                    args.seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    args.segment_lengths,
                    args.dilation_rates,
                    impl,
                    args.warmup_iters,
                    args.test_iters,
                ),
                nprocs=args.world_size,
                join=True,
            )

        except Exception as e:
            print(f"Failed to test {impl}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print("Multi-GPU Scaling Summary")
    print(f"{'=' * 80}")

    if single_time is not None:
        print(f"\nSingle GPU ({args.seq_len * args.world_size} tokens):")
        print(f"  Time: {single_time:.2f} ms")
        print(f"  Throughput: {single_throughput:,.0f} tokens/sec")
        print(
            "\nNote: Direct comparison shows memory scaling benefit of ring attention"
        )
    else:
        print(f"\nSingle GPU: OOM for {args.seq_len * args.world_size} tokens")
        print("This demonstrates the memory scaling advantage of ring attention!")


if __name__ == "__main__":
    main()
