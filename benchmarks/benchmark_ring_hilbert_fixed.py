#!/usr/bin/env python3
"""
Fixed benchmark for Ring Hilbert Attention.
Handles distributed execution more carefully.
"""

import os
import torch
import torch.distributed as dist
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import argparse
import json
from datetime import datetime

# Import base classes
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective as BaseRingAttention,
)


class HilbertRingAttention(BaseRingAttention):
    """Ring Attention with Hilbert ordering using collective operations."""

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_hilbert: bool = True,
        hilbert_chunk_size: int = 4096,
        **kwargs,
    ):
        super().__init__(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.use_hilbert = use_hilbert
        self.hilbert_chunk_size = hilbert_chunk_size
        self._hilbert_cache = {}

    def _generate_hilbert_curve(self, n: int) -> torch.Tensor:
        """Generate Hilbert curve mapping."""
        if n in self._hilbert_cache:
            return self._hilbert_cache[n]

        # For large sequences, use chunked Hilbert
        if n > self.hilbert_chunk_size:
            chunk_size = self.hilbert_chunk_size
            num_chunks = (n + chunk_size - 1) // chunk_size

            mapping = torch.arange(n, dtype=torch.long)

            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                chunk_len = end - start

                chunk_mapping = self._generate_hilbert_curve_small(chunk_len)
                mapping[start:end] = start + chunk_mapping

            self._hilbert_cache[n] = mapping.to(self.device)
            return self._hilbert_cache[n]
        else:
            mapping = self._generate_hilbert_curve_small(n)
            self._hilbert_cache[n] = mapping.to(self.device)
            return self._hilbert_cache[n]

    def _generate_hilbert_curve_small(self, n: int) -> torch.Tensor:
        """Generate Hilbert curve for smaller sizes."""
        size = 1
        while size * size < n:
            size *= 2

        def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
            x = y = 0
            s = 1
            while s < n:
                rx = 1 if (d // 2) & 1 else 0
                ry = 1 if (d ^ rx) & 1 else 0
                if ry == 0:
                    if rx == 1:
                        x, y = n - 1 - x, n - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d //= 4
                s *= 2
            return x, y

        mapping = torch.zeros(n, dtype=torch.long)
        for d in range(min(n, size * size)):
            x, y = hilbert_d2xy(size, d)
            linear_idx = y * size + x
            if linear_idx < n and d < n:
                mapping[linear_idx] = d

        return mapping

    def _apply_dilated_attention_single_gpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Override to apply Hilbert ordering."""
        if not self.use_hilbert:
            return super()._apply_dilated_attention_single_gpu(
                query, key, value, is_causal
            )

        b, n, h, d = query.shape

        # Apply Hilbert ordering to K,V
        mapping = self._generate_hilbert_curve(n)
        key = key[:, mapping, :, :]
        value = value[:, mapping, :, :]

        # Apply standard attention
        output = super()._apply_dilated_attention_single_gpu(
            query, key, value, is_causal
        )

        return output

    def _apply_dilated_attention_distributed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Override to apply Hilbert ordering before distribution."""
        if not self.use_hilbert:
            return super()._apply_dilated_attention_distributed(
                query, key, value, is_causal
            )

        b, n, h, d = query.shape

        # Apply Hilbert ordering to K,V before distribution
        mapping = self._generate_hilbert_curve(n)
        key = key[:, mapping, :, :]
        value = value[:, mapping, :, :]

        # Apply distributed attention
        output = super()._apply_dilated_attention_distributed(
            query, key, value, is_causal
        )

        return output


def init_distributed():
    """Initialize distributed environment."""
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        print(f"Rank {rank}: Initialized distributed (device=cuda:{local_rank})")
    else:
        print("Running in single GPU mode")


def get_rank_and_world_size():
    """Get current rank and world size."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def benchmark_configuration(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    batch_size: int = 1,
    warmup: int = 3,
    iterations: int = 10,
) -> Optional[Dict[str, float]]:
    """Benchmark standard vs Hilbert ring attention."""

    rank, world_size = get_rank_and_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    try:
        # Create models
        model_standard = HilbertRingAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=False,
            use_flash_attention=False,  # Disable flash for fair comparison
        )

        model_hilbert = HilbertRingAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=min(4096, seq_len // max(1, world_size)),
            use_flash_attention=False,  # Disable flash for fair comparison
        )

        # Create input tensors
        head_dim = hidden_dim // num_heads
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model_standard(q, k, v)
                _ = model_hilbert(q, k, v)

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Benchmark standard
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model_standard(q, k, v)

        torch.cuda.synchronize()
        standard_time = (time.perf_counter() - start) / iterations * 1000

        if dist.is_initialized():
            dist.barrier()

        # Benchmark Hilbert
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model_hilbert(q, k, v)

        torch.cuda.synchronize()
        hilbert_time = (time.perf_counter() - start) / iterations * 1000

        # Average across ranks
        if dist.is_initialized():
            times = torch.tensor([standard_time, hilbert_time], device=device)
            dist.all_reduce(times, op=dist.ReduceOp.AVG)
            standard_time, hilbert_time = times.tolist()

        # Memory usage
        if rank == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            allocated = reserved = 0

        return {
            "standard_time_ms": standard_time,
            "hilbert_time_ms": hilbert_time,
            "speedup": standard_time / hilbert_time,
            "memory_gb": allocated,
            "reserved_gb": reserved,
        }

    except Exception as e:
        if rank == 0:
            print(f"Error in benchmark: {e}")
            import traceback

            traceback.print_exc()
        return None


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test")
    args = parser.parse_args()

    # Initialize distributed
    init_distributed()

    rank, world_size = get_rank_and_world_size()

    if rank == 0:
        print("\n=== Ring Attention with Hilbert Ordering (V2 Collective) ===")
        print(f"Running on {world_size} GPU(s)")
        print("Using collective operations for robustness\n")

    # Test configurations
    if args.quick:
        configs = [
            # (seq_len, hidden_dim, num_heads, segment_lengths, dilation_rates, batch_size)
            (8192, 512, 8, [2048], [1], 2),
            (8192, 512, 8, [2048], [2], 2),
            (16384, 768, 12, [4096], [1], 1),
            (16384, 768, 12, [4096], [2], 1),
        ]
    else:
        configs = [
            (32768, 512, 8, [4096], [1], 1),
            (32768, 512, 8, [4096], [2], 1),
            (32768, 512, 8, [4096], [4], 1),
            (65536, 768, 12, [8192], [2], 1),
            (65536, 768, 12, [8192], [4], 1),
        ]

    results = []

    if rank == 0:
        print(
            "Configuration                                      | Standard | Hilbert  | Speedup | Memory"
        )
        print("-" * 95)

    for (
        seq_len,
        hidden_dim,
        num_heads,
        segment_lengths,
        dilation_rates,
        batch_size,
    ) in configs:
        # Skip if sequence not divisible by segment length
        if seq_len % segment_lengths[0] != 0:
            if rank == 0:
                print(
                    f"Skipping L={seq_len} (not divisible by segment length {segment_lengths[0]})"
                )
            continue

        result = benchmark_configuration(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            batch_size=batch_size,
            warmup=2 if args.quick else 3,
            iterations=5 if args.quick else 10,
        )

        if result and rank == 0:
            print(
                f"L={seq_len:<6} H={hidden_dim} h={num_heads:2} seg={segment_lengths[0]} dil={dilation_rates[0]} B={batch_size} | "
                f"{result['standard_time_ms']:8.1f} | {result['hilbert_time_ms']:8.1f} | "
                f"{result['speedup']:7.2f} | {result['memory_gb']:5.1f}GB"
            )

            results.append(
                {
                    "config": {
                        "seq_len": seq_len,
                        "hidden_dim": hidden_dim,
                        "num_heads": num_heads,
                        "segment_lengths": segment_lengths,
                        "dilation_rates": dilation_rates,
                        "batch_size": batch_size,
                        "world_size": world_size,
                    },
                    "metrics": result,
                }
            )

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 95)
        print("SUMMARY")
        print("=" * 95)

        speedups = [r["metrics"]["speedup"] for r in results]

        print(f"\nPerformance with {world_size} GPU(s):")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Maximum speedup: {max(speedups):.2f}x")
        print(f"  Minimum speedup: {min(speedups):.2f}x")

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"ring_hilbert_v2_collective_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "implementation": "V2 Collective",
                    "results": results,
                    "summary": {
                        "avg_speedup": float(np.mean(speedups)),
                        "max_speedup": float(max(speedups)),
                        "min_speedup": float(min(speedups)),
                    },
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to '{filename}'")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
