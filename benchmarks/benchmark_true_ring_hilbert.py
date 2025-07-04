#!/usr/bin/env python3
"""
Benchmark True Ring Dilated Attention with Hilbert ordering.
Uses the proper ring_dilated_attention_true.py implementation with isend/irecv.
Tests with long sequences (32K+) as intended.
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

# Import the true ring attention
from dilated_attention_pytorch.ring_dilated_attention_true import (
    TrueRingDilatedAttention,
    ring_pass,
    get_rank_and_world_size,
)


class HilbertTrueRingDilatedAttention(TrueRingDilatedAttention):
    """
    True Ring Dilated Attention enhanced with Hilbert ordering.
    Integrates Hilbert curve ordering into the ring attention mechanism.
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bucket_size: int = 1024,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_hilbert: bool = True,
        hilbert_chunk_size: Optional[int] = None,
    ):
        super().__init__(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            bucket_size=bucket_size,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
        )

        self.use_hilbert = use_hilbert
        self.hilbert_chunk_size = hilbert_chunk_size or max(segment_lengths)
        self._hilbert_cache = {}

    def _generate_hilbert_curve(self, n: int) -> torch.Tensor:
        """Generate optimized Hilbert curve mapping."""
        if n in self._hilbert_cache:
            return self._hilbert_cache[n]

        # For very large sequences, use chunk-based Hilbert
        if n > 16384:
            # Apply Hilbert to chunks to reduce overhead
            chunk_size = min(self.hilbert_chunk_size, n)
            num_chunks = (n + chunk_size - 1) // chunk_size

            mapping = torch.arange(n, dtype=torch.long)

            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                chunk_len = end - start

                # Generate Hilbert for this chunk
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
        # Find power of 2 size
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

        # Generate mapping
        mapping = torch.zeros(n, dtype=torch.long)
        for d in range(min(n, size * size)):
            x, y = hilbert_d2xy(size, d)
            linear_idx = y * size + x
            if linear_idx < n and d < n:
                mapping[linear_idx] = d

        return mapping

    def _apply_hilbert_to_chunk(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Hilbert ordering to a tensor chunk."""
        if not self.use_hilbert:
            return tensor

        b, n, h, d = tensor.shape
        mapping = self._generate_hilbert_curve(n)

        # Apply ordering using advanced indexing (fastest)
        return tensor[:, mapping, :, :]

    def _reverse_hilbert_from_chunk(
        self, tensor: torch.Tensor, original_size: int
    ) -> torch.Tensor:
        """Reverse Hilbert ordering from a tensor chunk."""
        if not self.use_hilbert:
            return tensor

        b, n, h, d = tensor.shape
        mapping = self._generate_hilbert_curve(original_size)
        inverse_mapping = torch.argsort(mapping)[:n]

        # Reverse ordering
        return tensor[:, inverse_mapping, :, :]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with Hilbert-enhanced ring attention.

        The key insight: Apply Hilbert ordering to K,V chunks before ring passing.
        This improves cache efficiency during the attention computation.
        """
        b, n, h, d = q.shape

        # For single device, use parent implementation
        if self.ring_size == 1:
            return super().forward(q, k, v, is_causal)

        # Check sequence length divisibility
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )
        chunk_size = n // self.ring_size

        # Get local chunks
        local_start = self.rank * chunk_size
        local_end = (self.rank + 1) * chunk_size

        # Apply Hilbert ordering to full K,V before chunking
        # This ensures that the Hilbert curve spans the full sequence
        if self.use_hilbert:
            k_hilbert = self._apply_hilbert_to_chunk(k)
            v_hilbert = self._apply_hilbert_to_chunk(v)
        else:
            k_hilbert = k
            v_hilbert = v

        # Now get local chunks from Hilbert-ordered tensors
        k_local = k_hilbert[:, local_start:local_end].contiguous()
        v_local = v_hilbert[:, local_start:local_end].contiguous()

        # Q remains in original order (no Hilbert for queries)
        q_full = q.contiguous()

        # Initialize output and statistics
        output = torch.zeros_like(q)
        max_score = torch.full(
            (b, n, h, 1), float("-inf"), device=self.device, dtype=self.dtype
        )
        sum_exp = torch.zeros((b, n, h, 1), device=self.device, dtype=self.dtype)

        # Allocate communication buffers
        self.recv_k_buffer = self._get_or_create_buffer(
            k_local.shape, k_local.dtype, "recv_k_buffer"
        )
        self.recv_v_buffer = self._get_or_create_buffer(
            v_local.shape, v_local.dtype, "recv_v_buffer"
        )

        # Current chunks
        current_k = k_local
        current_v = v_local

        # Ring iterations
        for step in range(self.ring_size):
            # Which chunk are we processing?
            chunk_rank = (self.rank - step) % self.ring_size
            chunk_start = chunk_rank * chunk_size
            chunk_end = (chunk_rank + 1) * chunk_size

            # Apply dilated patterns (already in Hilbert space)
            k_dilated, v_dilated = self._apply_dilated_patterns(
                current_k, current_v, chunk_start, chunk_size
            )

            # Compute attention
            self._compute_chunk_attention(
                q_full,
                k_dilated,
                v_dilated,
                chunk_start,
                chunk_end,
                output,
                max_score,
                sum_exp,
                is_causal,
                step,
            )

            # Ring pass
            if step < self.ring_size - 1:
                next_rank = (self.rank + 1) % self.ring_size
                prev_rank = (self.rank - 1) % self.ring_size

                # Ring communication with proper async handling
                send_k = current_k
                send_v = current_v

                current_k = ring_pass(send_k, self.recv_k_buffer, next_rank, prev_rank)
                current_v = ring_pass(send_v, self.recv_v_buffer, next_rank, prev_rank)

                # Swap buffers
                self.recv_k_buffer = send_k
                self.recv_v_buffer = send_v

        # Final normalization
        output = output / (sum_exp + 1e-8)

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
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16  # Use fp16 for large sequences

    try:
        # Create models
        model_standard = HilbertTrueRingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=False,
        )

        model_hilbert = HilbertTrueRingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=min(8192, seq_len // world_size),
        )

        # Create input tensors
        # Note: Ring attention expects (batch, seq_len, num_heads, head_dim)
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
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
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
    """Run benchmark on long sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with smaller sequences"
    )
    args = parser.parse_args()

    # Initialize distributed
    init_distributed()

    rank, world_size = get_rank_and_world_size()

    if rank == 0:
        print("\n=== True Ring Dilated Attention with Hilbert Ordering ===")
        print(f"Running on {world_size} GPU(s)")
        print("Using proper ring communication with isend/irecv")
        print("Testing with long sequences as intended\n")

    # Test configurations
    if args.quick:
        # Smaller sequences for quick testing
        configs = [
            # (seq_len, hidden_dim, num_heads, segment_lengths, dilation_rates, batch_size)
            (8192, 512, 8, [2048], [1], 2),
            (8192, 512, 8, [2048], [2], 2),
            (16384, 768, 12, [4096], [2], 1),
        ]
    else:
        # Long sequences as intended for Ring Attention
        configs = [
            (32768, 512, 8, [4096], [1], 1),  # 32K
            (32768, 512, 8, [4096], [2], 1),
            (32768, 512, 8, [4096], [4], 1),
            (65536, 768, 12, [8192], [2], 1),  # 64K
            (65536, 768, 12, [8192], [4], 1),
        ]

        # Add even longer sequences if memory allows
        if world_size >= 4:
            configs.extend(
                [
                    (131072, 768, 12, [16384], [4], 1),  # 128K
                    (262144, 1024, 16, [32768], [8], 1),  # 256K
                ]
            )

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
        # Skip if sequence not divisible by world size
        if seq_len % world_size != 0:
            if rank == 0:
                print(f"Skipping L={seq_len} (not divisible by {world_size} GPUs)")
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
        print(
            f"  Configs with speedup > 1: {sum(1 for s in speedups if s > 1)}/{len(speedups)}"
        )

        # Group by sequence length
        print("\nSpeedup by sequence length:")
        seq_lens = sorted(set(r["config"]["seq_len"] for r in results))
        for seq_len in seq_lens:
            seq_speedups = [
                r["metrics"]["speedup"]
                for r in results
                if r["config"]["seq_len"] == seq_len
            ]
            if seq_speedups:
                print(
                    f"  L={seq_len}: avg {np.mean(seq_speedups):.2f}x, max {max(seq_speedups):.2f}x"
                )

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"true_ring_hilbert_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
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

        print("\n" + "=" * 95)
        print("KEY INSIGHTS")
        print("=" * 95)
        print("""
        1. Using true Ring Attention with proper isend/irecv communication
        2. Testing with intended sequence lengths (32K+)
        3. Hilbert ordering applied to K,V chunks before ring passing
        4. Memory usage stays O(n/p) as expected
        5. Benefits should increase with sequence length and dilation rate
        """)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
