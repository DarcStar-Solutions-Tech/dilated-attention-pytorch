#!/usr/bin/env python3
"""
Test RingDistributedDilatedAttention with DeepSpeed optimizations.

This script benchmarks the enterprise-grade distributed implementation
that uses DeepSpeed for memory optimization and communication.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dilated_attention_pytorch.ring_distributed_dilated_attention import (
    RingDistributedDilatedAttention,
)

# Try to import DeepSpeed
try:
    import deepspeed

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("Warning: DeepSpeed not available. Install with: pip install deepspeed")


def setup(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def create_deepspeed_config(args):
    """Create DeepSpeed configuration."""
    config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8},
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_optimizer": {
                "device": "cpu" if args.cpu_offload else "none",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu" if args.cpu_offload else "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 50000000,
            "stage3_prefetch_bucket_size": 50000000,
            "stage3_param_persistence_threshold": 10000,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }

    if args.gradient_compression:
        config["compression_training"] = {
            "weight_quantization": {
                "shared_parameters": {
                    "enabled": True,
                    "quantizer_kernel": True,
                    "schedule_offset": 0,
                    "quantize_groups": 1,
                    "quantize_verbose": False,
                    "quantization_type": "symmetric",
                    "rounding": "nearest",
                }
            }
        }

    return config


def run_deepspeed_test(rank, world_size, args, results_queue):
    """Run test with DeepSpeed optimizations."""
    print(f"[Rank {rank}] Starting on GPU {rank}")

    # Setup distributed
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    # Create model
    embed_dim = args.num_heads * args.head_dim
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Filter segment lengths
    segment_lengths = [s for s in segment_lengths if s <= args.seq_len]
    dilation_rates = dilation_rates[: len(segment_lengths)]

    try:
        model = RingDistributedDilatedAttention(
            embed_dim=embed_dim,
            num_heads=args.num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            # DeepSpeed settings
            use_deepspeed=HAS_DEEPSPEED and args.use_deepspeed,
            zero_stage=args.zero_stage,
            cpu_offload=args.cpu_offload,
            use_gradient_compression=args.gradient_compression,
            # Communication settings
            overlap_communication=True,
            bucket_size=50,  # MB
            # Device settings
            device=device,
            dtype=dtype,
        )

        # Initialize with DeepSpeed if available
        if HAS_DEEPSPEED and args.use_deepspeed:
            ds_config = create_deepspeed_config(args)

            # Create optimizer (required by DeepSpeed)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Initialize DeepSpeed
            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=ds_config,
                dist_init_required=False,  # Already initialized
            )

            if rank == 0:
                print(f"DeepSpeed initialized with ZeRO stage {args.zero_stage}")

        # Create inputs
        q = torch.randn(
            args.batch_size, args.seq_len, embed_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            args.batch_size, args.seq_len, embed_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            args.batch_size, args.seq_len, embed_dim, device=device, dtype=dtype
        )

        # Warmup
        for _ in range(3):
            if HAS_DEEPSPEED and args.use_deepspeed:
                output = model(q, k, v)
                # Fake backward for DeepSpeed
                loss = output.mean()
                model.backward(loss)
                model.step()
            else:
                with torch.no_grad():
                    output = model(q, k, v)

        # Synchronize
        dist.barrier()
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()

        num_iters = 10
        for _ in range(num_iters):
            if HAS_DEEPSPEED and args.use_deepspeed:
                output = model(q, k, v)
                loss = output.mean()
                model.backward(loss)
                model.step()
            else:
                with torch.no_grad():
                    output = model(q, k, v)

        torch.cuda.synchronize()
        dist.barrier()

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / num_iters * 1000  # ms

        # Get memory stats
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Get DeepSpeed stats if available
        ds_stats = {}
        if HAS_DEEPSPEED and args.use_deepspeed and hasattr(model, "get_memory_info"):
            ds_stats = model.get_memory_info()

        # Send results from rank 0
        if rank == 0:
            results_queue.put(
                {
                    "success": True,
                    "avg_time": avg_time,
                    "memory_mb": memory_mb,
                    "deepspeed_stats": ds_stats,
                }
            )

    except Exception as e:
        if rank == 0:
            results_queue.put({"success": False, "error": str(e)})
        import traceback

        traceback.print_exc()

    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Test DeepSpeed Ring Attention")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument(
        "--use-deepspeed", action="store_true", help="Use DeepSpeed optimizations"
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="DeepSpeed ZeRO optimization stage",
    )
    parser.add_argument("--cpu-offload", action="store_true", help="Offload to CPU")
    parser.add_argument(
        "--gradient-compression",
        action="store_true",
        help="Enable gradient compression",
    )

    args = parser.parse_args()

    # Check GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus < args.world_size:
        print(f"Error: Requested {args.world_size} GPUs but only {n_gpus} available")
        return

    print(f"\n{'=' * 60}")
    print("DeepSpeed Ring Attention Test")
    print(f"{'=' * 60}")
    print(f"World size: {args.world_size} GPUs")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batch size: {args.batch_size * args.world_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Use DeepSpeed: {args.use_deepspeed}")
    if args.use_deepspeed:
        print(f"  ZeRO stage: {args.zero_stage}")
        print(f"  CPU offload: {args.cpu_offload}")
        print(f"  Gradient compression: {args.gradient_compression}")
    print(f"{'=' * 60}\n")

    # Create results queue
    results_queue = mp.Queue()

    # Spawn processes
    mp.spawn(
        run_deepspeed_test,
        args=(args.world_size, args, results_queue),
        nprocs=args.world_size,
        join=True,
    )

    # Get results
    results = results_queue.get()

    if results["success"]:
        print(f"\n{'=' * 60}")
        print("Results:")
        print(f"{'=' * 60}")
        print(f"Average time: {results['avg_time']:.2f} ms")
        print(f"Memory per GPU: {results['memory_mb']:.2f} MB")
        print(f"Total memory: {results['memory_mb'] * args.world_size:.2f} MB")
        print(
            f"Throughput: {args.batch_size * args.world_size * args.seq_len / (results['avg_time'] / 1000):.0f} tokens/s"
        )

        if results["deepspeed_stats"]:
            print("\nDeepSpeed Stats:")
            print(json.dumps(results["deepspeed_stats"], indent=2))
    else:
        print(f"\nError: {results['error']}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
