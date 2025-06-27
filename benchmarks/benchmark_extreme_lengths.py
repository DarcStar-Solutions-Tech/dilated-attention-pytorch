#!/usr/bin/env python3
"""
Benchmark all attention implementations with extreme sequence lengths.

This script tests each implementation with progressively longer sequences
to find their maximum supported length before failure.
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import torch

# Import all attention implementations
from dilated_attention_pytorch.dilated_attention import DilatedAttention
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


def format_memory(bytes):
    """Format bytes into human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def test_sequence_length(
    model, seq_len, batch_size, num_heads, head_dim, device, dtype
):
    """Test if model can handle given sequence length."""
    torch.cuda.empty_cache() if device.type == "cuda" else None
    gc.collect()

    try:
        # Create inputs
        query = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

        # Time forward pass
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        forward_time = (time.perf_counter() - start_time) * 1000  # ms

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated()
            mem_used = peak_mem - start_mem
        else:
            mem_used = 0

        # Cleanup
        del query, key, value, output
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()

        return {
            "success": True,
            "forward_time_ms": forward_time,
            "memory_used": mem_used,
            "error": None,
        }

    except Exception as e:
        # Cleanup on error
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()

        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            error_type = "OOM"
        elif "divisible" in error_msg.lower():
            error_type = "DIVISIBILITY"
        else:
            error_type = "OTHER"

        return {
            "success": False,
            "forward_time_ms": None,
            "memory_used": None,
            "error": error_type,
            "error_msg": error_msg,
        }


def create_models(device, dtype, max_seq_len):
    """Create all model configurations to test."""
    models = []

    # Calculate segment lengths based on max sequence length
    segment_lengths = []
    seg = 2048
    while seg <= max_seq_len:
        segment_lengths.append(seg)
        seg *= 2

    if not segment_lengths:
        segment_lengths = [2048]

    # Limit to 3 segments for efficiency
    segment_lengths = segment_lengths[-3:]
    dilation_rates = [1, 2, 4][: len(segment_lengths)]

    print(f"Using segment_lengths: {segment_lengths}")
    print(f"Using dilation_rates: {dilation_rates}")

    # 1. Basic DilatedAttention
    models.append(
        {
            "name": "DilatedAttention",
            "model": DilatedAttention(
                segment_lengths=segment_lengths, dilation_rates=dilation_rates
            ),
        }
    )

    # 2. ImprovedDilatedAttention
    models.append(
        {
            "name": "ImprovedDilatedAttention",
            "model": ImprovedDilatedAttention(
                segment_lengths=segment_lengths, dilation_rates=dilation_rates
            ),
        }
    )

    # 3. RingDilatedAttentionV2 with different ring sizes
    for ring_size in [1, 4, 8, 16]:
        models.append(
            {
                "name": f"RingDilatedAttentionV2_r{ring_size}",
                "model": RingDilatedAttentionV2(
                    ring_size=ring_size,
                    device=device,
                    dtype=dtype,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                ),
            }
        )

    # 4. RingDilatedAttentionProduction
    for ring_size in [1, 8, 16]:
        config = RingAttentionConfig(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=ring_size,
            use_memory_pool=True,
            use_gradient_checkpointing=True,
            mixed_precision=(dtype == torch.float16),
        )
        models.append(
            {
                "name": f"RingDilatedAttentionProduction_r{ring_size}",
                "model": RingDilatedAttentionProduction(config),
            }
        )

    # 5. BlockSparseRingDilatedAttention with different sparsity
    for sparsity in [0.9, 0.95]:
        sparse_config = SparsePatternConfig(
            pattern_type="local_window",
            sparsity_ratio=sparsity,
            block_size=64,
            local_window_size=256,
        )

        models.append(
            {
                "name": f"BlockSparse_LocalWindow_{sparsity}",
                "model": BlockSparseRingDilatedAttention(
                    ring_size=1,
                    device=device,
                    dtype=dtype,
                    sparse_config=sparse_config,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                ),
            }
        )

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Test extreme sequence lengths for all attention implementations"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--min_seq_len", type=int, default=4096, help="Starting sequence length"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1048576,
        help="Maximum sequence length to test",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Extreme Sequence Length Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU Memory: {format_memory(total_memory)}")

    # Test sequence lengths (powers of 2)
    seq_lengths = []
    seq_len = args.min_seq_len
    while seq_len <= args.max_seq_len:
        seq_lengths.append(seq_len)
        seq_len *= 2

    print(f"\nTesting sequence lengths: {seq_lengths}")

    # Create models
    models = create_models(device, dtype, args.max_seq_len)

    # Results storage
    results = {}

    # Test each model
    for model_info in models:
        model_name = model_info["name"]
        model = model_info["model"]

        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 80}")

        results[model_name] = []
        max_successful_len = 0

        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len:,}")

            result = test_sequence_length(
                model,
                seq_len,
                args.batch_size,
                args.num_heads,
                args.head_dim,
                device,
                dtype,
            )

            result["seq_len"] = seq_len
            results[model_name].append(result)

            if result["success"]:
                max_successful_len = seq_len
                print(
                    f"  ✓ Success - Time: {result['forward_time_ms']:.2f} ms, "
                    f"Memory: {format_memory(result['memory_used'])}"
                )
            else:
                print(f"  ✗ Failed - Error: {result['error']}")
                if result["error"] == "OOM":
                    print(f"    Maximum successful length: {max_successful_len:,}")
                    break
                elif result["error"] == "DIVISIBILITY":
                    print("    Skipping due to divisibility constraint")
                    continue

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"extreme-lengths-benchmark-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Extreme Sequence Length Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Data type: {dtype}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        if device.type == "cuda":
            f.write(f"- GPU: {torch.cuda.get_device_name()}\n")
            f.write(f"- Total GPU Memory: {format_memory(total_memory)}\n")
        f.write("\n")

        f.write("## Results Summary\n\n")

        f.write(
            "| Implementation | Max Successful Length | Max Time (ms) | Max Memory | Failure Reason |\n"
        )
        f.write(
            "|----------------|----------------------|---------------|------------|----------------|\n"
        )

        for model_name, model_results in results.items():
            # Find max successful length
            max_len = 0
            max_time = 0
            max_mem = 0
            failure_reason = "N/A"

            for r in model_results:
                if r["success"]:
                    max_len = r["seq_len"]
                    max_time = max(max_time, r["forward_time_ms"])
                    max_mem = max(max_mem, r["memory_used"])
                else:
                    if failure_reason == "N/A":
                        failure_reason = r["error"]

            f.write(
                f"| {model_name} | {max_len:,} | "
                f"{max_time:.2f} | {format_memory(max_mem)} | {failure_reason} |\n"
            )

        f.write("\n## Detailed Results\n\n")

        for model_name, model_results in results.items():
            f.write(f"### {model_name}\n\n")

            f.write("| Seq Length | Status | Time (ms) | Memory | Error |\n")
            f.write("|------------|--------|-----------|--------|-------|\n")

            for r in model_results:
                status = "✓" if r["success"] else "✗"
                time_str = f"{r['forward_time_ms']:.2f}" if r["success"] else "-"
                mem_str = format_memory(r["memory_used"]) if r["success"] else "-"
                error_str = r["error"] if not r["success"] else "-"

                f.write(
                    f"| {r['seq_len']:,} | {status} | {time_str} | {mem_str} | {error_str} |\n"
                )

            f.write("\n")

        f.write("## Analysis\n\n")

        # Find winner for longest sequence
        max_overall = 0
        winner = None
        for model_name, model_results in results.items():
            for r in model_results:
                if r["success"] and r["seq_len"] > max_overall:
                    max_overall = r["seq_len"]
                    winner = model_name

        if winner:
            f.write(
                f"- **Longest sequence achieved**: {max_overall:,} tokens by {winner}\n"
            )

        # Compare Ring Attention effectiveness
        ring_models = [m for m in results.keys() if "Ring" in m]
        if ring_models:
            f.write("\n### Ring Attention Analysis\n\n")
            for model in ring_models:
                max_len = max(
                    [r["seq_len"] for r in results[model] if r["success"]], default=0
                )
                if max_len > 0:
                    f.write(f"- {model}: up to {max_len:,} tokens\n")

    print(f"\n\nResults saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_name, model_results in results.items():
        max_len = max([r["seq_len"] for r in model_results if r["success"]], default=0)
        print(f"{model_name:40} max length: {max_len:,}")


if __name__ == "__main__":
    main()
