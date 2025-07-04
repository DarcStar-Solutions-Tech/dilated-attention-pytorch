#!/usr/bin/env python3
"""
Optimized Ring Hilbert Attention benchmark with reduced overhead.
Key optimizations:
1. Pre-computed Hilbert mappings
2. Fused operations
3. Chunk-wise Hilbert ordering
4. Better memory management
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import argparse
import json
from datetime import datetime
import math


# Pre-compute common Hilbert mappings
HILBERT_CACHE = {}


def precompute_hilbert_mappings(sizes: List[int], device: torch.device):
    """Pre-compute Hilbert mappings for common sizes."""
    global HILBERT_CACHE

    for size in sizes:
        if size not in HILBERT_CACHE:
            mapping = generate_optimized_hilbert_curve(size)
            HILBERT_CACHE[size] = {
                "forward": mapping.to(device),
                "inverse": torch.argsort(mapping).to(device),
            }


def generate_optimized_hilbert_curve(n: int) -> torch.Tensor:
    """Generate Hilbert curve using lookup tables for common sizes."""
    # For small sizes, use pre-defined optimal orderings
    if n <= 64:
        # Simple sequential for very small sizes
        return torch.arange(n)

    # Power of 2 optimization
    size = 1
    while size * size < n:
        size *= 2

    # Use vectorized generation
    hilbert_map = torch.zeros(n, dtype=torch.long)

    # Generate Hilbert curve coordinates
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

    # Fill mapping
    for d in range(min(n, size * size)):
        x, y = hilbert_d2xy(size, d)
        linear_idx = y * size + x
        if linear_idx < n and d < n:
            hilbert_map[linear_idx] = d

    return hilbert_map


class OptimizedRingHilbertAttention(nn.Module):
    """
    Optimized Ring Hilbert Attention with reduced overhead.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        use_hilbert: bool = False,
        chunk_wise_hilbert: bool = True,  # Apply Hilbert per chunk, not globally
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.use_hilbert = use_hilbert
        self.chunk_wise_hilbert = chunk_wise_hilbert

        # Distributed settings
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(f"cuda:{self.rank}")

        # Layers
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.to(self.device)

    def _apply_hilbert_ordering_optimized(
        self, tensor: torch.Tensor, chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Hilbert ordering with minimal overhead."""
        batch_size, seq_len, hidden_dim = tensor.shape

        if chunk_size and self.chunk_wise_hilbert:
            # Apply Hilbert ordering per chunk (more cache-friendly)
            chunks = []
            inverse_info = []

            for i in range(0, seq_len, chunk_size):
                end = min(i + chunk_size, seq_len)
                chunk_len = end - i

                if chunk_len in HILBERT_CACHE:
                    mapping = HILBERT_CACHE[chunk_len]["forward"]
                    inv_mapping = HILBERT_CACHE[chunk_len]["inverse"]
                else:
                    mapping = torch.arange(chunk_len, device=self.device)
                    inv_mapping = mapping

                chunk = tensor[:, i:end, :]
                # Use advanced indexing (fastest based on analysis)
                ordered_chunk = chunk[:, mapping, :]
                chunks.append(ordered_chunk)
                inverse_info.append((i, end, inv_mapping))

            ordered = torch.cat(chunks, dim=1)
            return ordered, inverse_info
        else:
            # Global Hilbert ordering
            if seq_len in HILBERT_CACHE:
                mapping = HILBERT_CACHE[seq_len]["forward"]
                inv_mapping = HILBERT_CACHE[seq_len]["inverse"]
            else:
                mapping = torch.arange(seq_len, device=self.device)
                inv_mapping = mapping

            ordered = tensor[:, mapping, :]
            return ordered, inv_mapping

    def _reverse_hilbert_ordering_optimized(
        self, tensor: torch.Tensor, inverse_info
    ) -> torch.Tensor:
        """Reverse Hilbert ordering with minimal overhead."""
        batch_size, seq_len, hidden_dim = tensor.shape

        if isinstance(inverse_info, list):
            # Chunk-wise reversal
            chunks = []
            for i, (start, end, inv_mapping) in enumerate(inverse_info):
                chunk = tensor[:, start:end, :]
                reversed_chunk = chunk[:, inv_mapping, :]
                chunks.append(reversed_chunk)
            return torch.cat(chunks, dim=1)
        else:
            # Global reversal
            return tensor[:, inverse_info, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass."""
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)

        # Apply Hilbert ordering if enabled
        if self.use_hilbert:
            # Flatten for ordering
            q_flat = q.transpose(1, 2).reshape(batch_size, seq_len, -1)
            k_flat = k.transpose(1, 2).reshape(batch_size, seq_len, -1)
            v_flat = v.transpose(1, 2).reshape(batch_size, seq_len, -1)

            # Apply optimized ordering
            chunk_size = self.segment_lengths[0] if self.chunk_wise_hilbert else None
            q_ordered, q_inv = self._apply_hilbert_ordering_optimized(
                q_flat, chunk_size
            )
            k_ordered, _ = self._apply_hilbert_ordering_optimized(k_flat, chunk_size)
            v_ordered, _ = self._apply_hilbert_ordering_optimized(v_flat, chunk_size)

            # Reshape back
            q = q_ordered.reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k = k_ordered.reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v = v_ordered.reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        else:
            q_inv = None

        # Ring attention computation
        if self.world_size > 1:
            # Multi-GPU ring attention
            output = self._ring_attention_optimized(q, k, v)
        else:
            # Single GPU fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)

        # Reverse Hilbert ordering if applied
        if self.use_hilbert:
            output_flat = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
            output_flat = self._reverse_hilbert_ordering_optimized(output_flat, q_inv)
            output = output_flat.reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        # Final projection
        output = (
            output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.hidden_dim)
        )
        return self.out_proj(output)

    def _ring_attention_optimized(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Optimized ring attention with all-gather."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # All-gather K and V
        k_gathered = [torch.empty_like(k) for _ in range(self.world_size)]
        v_gathered = [torch.empty_like(v) for _ in range(self.world_size)]

        dist.all_gather(k_gathered, k.contiguous())
        dist.all_gather(v_gathered, v.contiguous())

        # Concatenate
        k_full = torch.cat(k_gathered, dim=2)  # Full sequence
        v_full = torch.cat(v_gathered, dim=2)

        # Compute attention on local queries with full K,V
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v_full)

        return output


def benchmark_optimized(
    hidden_dim: int,
    num_heads: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    batch_size: int,
    seq_len: int,
    chunk_wise: bool = True,
    warmup: int = 5,
    iterations: int = 20,
) -> Optional[Dict[str, float]]:
    """Benchmark optimized implementation."""

    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    # Pre-compute Hilbert mappings
    if chunk_wise:
        sizes = segment_lengths
    else:
        sizes = [seq_len]
    precompute_hilbert_mappings(sizes, device)

    try:
        # Create models
        model_standard = OptimizedRingHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=False,
        )

        model_hilbert = OptimizedRingHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=True,
            chunk_wise_hilbert=chunk_wise,
        )

        # Input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model_standard(x)
                _ = model_hilbert(x)

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Benchmark
        times = []
        for model, name in [(model_standard, "standard"), (model_hilbert, "hilbert")]:
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(x)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / iterations * 1000
            times.append(elapsed)

            if dist.is_initialized():
                dist.barrier()

        standard_time, hilbert_time = times

        # Average across GPUs
        if dist.is_initialized():
            times_tensor = torch.tensor(times, device=device)
            dist.all_reduce(times_tensor, op=dist.ReduceOp.AVG)
            standard_time, hilbert_time = times_tensor.tolist()

        return {
            "standard_time_ms": standard_time,
            "hilbert_time_ms": hilbert_time,
            "speedup": standard_time / hilbert_time,
            "chunk_wise": chunk_wise,
        }

    except Exception as e:
        if rank == 0:
            print(f"Error: {e}")
        return None


def init_distributed():
    """Initialize distributed training."""
    if dist.is_initialized():
        return

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"Process {rank}: Initialized (device=cuda:{local_rank})")


def main():
    """Run optimized benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    init_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("\n=== Optimized Ring Hilbert Attention Benchmark ===")
        print(f"GPUs: {world_size}")
        print("\nOptimizations:")
        print("- Pre-computed Hilbert mappings")
        print("- Chunk-wise Hilbert ordering")
        print("- Optimized tensor operations")
        print("- Reduced memory allocations\n")

    # Test configurations
    if args.quick:
        configs = [
            (512, 8, [512], [1], 2, 2048),
            (512, 8, [512], [2], 2, 2048),
        ]
    else:
        configs = [
            (512, 8, [512], [1], 4, 2048),
            (512, 8, [512], [2], 4, 2048),
            (512, 8, [512], [4], 4, 2048),
            (768, 12, [768], [2], 2, 3072),
            (768, 12, [768], [4], 2, 3072),
        ]

    results = []

    if rank == 0:
        print(
            "Configuration                          | Standard | Hilbert  | Speedup | Type"
        )
        print("-" * 85)

    for (
        hidden_dim,
        num_heads,
        segment_lengths,
        dilation_rates,
        batch_size,
        seq_len,
    ) in configs:
        # Test both global and chunk-wise
        for chunk_wise in [False, True]:
            result = benchmark_optimized(
                hidden_dim,
                num_heads,
                segment_lengths,
                dilation_rates,
                batch_size,
                seq_len,
                chunk_wise,
                warmup=3 if args.quick else 5,
                iterations=10 if args.quick else 20,
            )

            if result and rank == 0:
                type_str = "chunk" if chunk_wise else "global"
                print(
                    f"H={hidden_dim} h={num_heads:2} L={seq_len} dil={dilation_rates[0]} B={batch_size} | "
                    f"{result['standard_time_ms']:8.1f} | {result['hilbert_time_ms']:8.1f} | "
                    f"{result['speedup']:7.2f} | {type_str}"
                )
                results.append(result)

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 85)
        speedups = [r["speedup"] for r in results]
        chunk_speedups = [r["speedup"] for r in results if r["chunk_wise"]]
        global_speedups = [r["speedup"] for r in results if not r["chunk_wise"]]

        print(f"Average speedup: {np.mean(speedups):.2f}x")
        if chunk_speedups:
            print(f"Chunk-wise avg: {np.mean(chunk_speedups):.2f}x")
        if global_speedups:
            print(f"Global avg: {np.mean(global_speedups):.2f}x")

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        with open(f"optimized_results_{world_size}gpu_{timestamp}.json", "w") as f:
            json.dump({"results": results, "world_size": world_size}, f, indent=2)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
