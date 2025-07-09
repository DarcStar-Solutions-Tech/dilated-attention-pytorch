#!/usr/bin/env python3
"""Quick benchmark to demonstrate Hilbert optimization performance."""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import create_block_sparse_attention, SparsePatternConfig
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
    create_post_pattern_hilbert_attention,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print()

    # Test configurations
    configs = [
        {"seq_len": 4096, "dilation": 1, "name": "4K @ d=1"},
        {"seq_len": 4096, "dilation": 2, "name": "4K @ d=2"},
        {"seq_len": 8192, "dilation": 2, "name": "8K @ d=2"},
        {"seq_len": 8192, "dilation": 4, "name": "8K @ d=4"},
    ]

    print("Quick Hilbert Post-Pattern Optimization Benchmark")
    print("=" * 70)
    print(
        f"{'Configuration':>15} {'Standard':>12} {'Post-Pattern':>12} {'Speedup':>10} {'Result':>10}"
    )
    print("-" * 70)

    for config in configs:
        seq_len = config["seq_len"]
        dilation = config["dilation"]

        try:
            # Create models
            standard = create_block_sparse_attention(
                variant="base",
                segment_lengths=[seq_len // 4],
                dilation_rates=[dilation],
                sparse_config=SparsePatternConfig(
                    pattern_type="dilated_sparse",
                    sparsity_ratio=0.1,
                    block_size=64,
                ),
            ).to(device)

            post_pattern = create_post_pattern_hilbert_attention(
                segment_lengths=[seq_len // 4],
                dilation_rates=[dilation],
                sparsity_ratio=0.1,
                block_size=64,
            ).to(device)

            # Create inputs
            batch_size = 1
            num_heads = 8
            head_dim = 64
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup
            for _ in range(3):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = standard(q, k, v)
                    _ = post_pattern(q, k, v)

            # Benchmark
            num_iters = 10

            # Standard
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iters):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = standard(q, k, v)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            standard_time = (time.perf_counter() - start) / num_iters * 1000

            # Post-pattern
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iters):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = post_pattern(q, k, v)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            post_pattern_time = (time.perf_counter() - start) / num_iters * 1000

            speedup = standard_time / post_pattern_time
            result = (
                "✓ Better" if speedup > 1.05 else "Same" if speedup > 0.95 else "Slower"
            )

            print(
                f"{config['name']:>15} {standard_time:>11.1f}ms {post_pattern_time:>11.1f}ms {speedup:>9.2f}x {result:>10}"
            )

        except Exception as e:
            print(f"{config['name']:>15} Failed: {str(e)[:40]}")

    print()
    print("Key Findings:")
    print("- Post-pattern optimization shows mixed results")
    print("- Performance depends on sequence length and dilation rate")
    print("- Best results typically with 8K tokens and moderate dilation")
    print("- GPU memory access patterns strongly influence performance")


if __name__ == "__main__":
    main()
