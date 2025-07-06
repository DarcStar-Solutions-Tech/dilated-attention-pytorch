#!/usr/bin/env python3
"""
Simple wrapper approach: Apply Hilbert ordering outside the model.
This avoids the model recreation issue entirely.
"""

import gc
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import math
from typing import Dict

from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)


class HilbertWrapper:
    """Simple wrapper to apply Hilbert ordering to tensors."""

    def __init__(self, device="cuda"):
        self.device = device
        self._cache = {}

    def _generate_hilbert_mapping(self, seq_len: int) -> torch.Tensor:
        """Generate a simple permutation for cache-friendly access."""
        if seq_len in self._cache:
            return self._cache[seq_len]

        # Use bit-reversal for power-of-2 sequences
        indices = torch.arange(seq_len, device=self.device, dtype=torch.long)

        n_bits = int(math.log2(seq_len))
        if 2**n_bits == seq_len:
            # Bit reversal permutation
            reversed_indices = torch.zeros_like(indices)
            for i in range(seq_len):
                rev = 0
                num = i
                for _ in range(n_bits):
                    rev = (rev << 1) | (num & 1)
                    num >>= 1
                reversed_indices[i] = rev
            indices = reversed_indices

        self._cache[seq_len] = indices
        return indices

    def apply_hilbert(
        self, tensor: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Apply Hilbert ordering to tensor."""
        batch_size, seq_len = tensor.shape[:2]
        rest_dims = list(tensor.shape[2:])

        mapping = self._generate_hilbert_mapping(seq_len)

        if inverse:
            # Create inverse mapping
            inverse_mapping = torch.empty_like(mapping)
            inverse_mapping[mapping] = torch.arange(seq_len, device=self.device)
            mapping = inverse_mapping

        # Reshape and apply mapping
        tensor_flat = tensor.reshape(batch_size, seq_len, -1)
        mapping_expanded = (
            mapping.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, tensor_flat.shape[-1])
        )
        reordered = tensor_flat.gather(1, mapping_expanded)

        # Reshape back
        final_shape = [batch_size, seq_len] + rest_dims
        return reordered.reshape(final_shape)


def benchmark_with_hilbert_wrapper(
    rank: int,
    world_size: int,
    seq_len: int,
    results_dict: Dict,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    iterations: int = 3,
):
    """Benchmark using Hilbert wrapper approach."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12361"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    base_segment = min(4096, seq_len // 4)
    segment_lengths = [base_segment, base_segment * 2]
    dilation_rates = [8, 16]

    # Ensure divisibility
    max_segment = max(segment_lengths)
    if seq_len % max_segment != 0:
        seq_len = ((seq_len // max_segment) + 1) * max_segment

    if rank == 0:
        print(f"\nTesting {seq_len:,} tokens with Hilbert wrapper on {world_size} GPUs")

    try:
        # Create base model
        model = RingDilatedAttentionHybridOptimizedV2(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
        )

        # Create Hilbert wrapper
        hilbert = HilbertWrapper(device)

        # Create tensors
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        dist.barrier()

        # Warmup
        if rank == 0:
            print("  Warming up...")

        # Apply Hilbert ordering
        q_hilbert = hilbert.apply_hilbert(q)
        k_hilbert = hilbert.apply_hilbert(k)
        v_hilbert = hilbert.apply_hilbert(v)

        with torch.no_grad():
            output_hilbert = model(q_hilbert, k_hilbert, v_hilbert)
            _ = hilbert.apply_hilbert(output_hilbert, inverse=True)

        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark with Hilbert
        if rank == 0:
            print("  Benchmarking with Hilbert...")

        times_hilbert = []
        for i in range(iterations):
            torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                # Apply Hilbert
                q_h = hilbert.apply_hilbert(q)
                k_h = hilbert.apply_hilbert(k)
                v_h = hilbert.apply_hilbert(v)

                # Forward pass
                output_h = model(q_h, k_h, v_h)

                # Reverse Hilbert
                output = hilbert.apply_hilbert(output_h, inverse=True)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times_hilbert.append(end - start)
            if rank == 0:
                print(f"    Iteration {i + 1}: {end - start:.3f}s")
            del output

        # Benchmark without Hilbert
        if rank == 0:
            print("  Benchmarking without Hilbert...")

        times_no_hilbert = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_no_hilbert.append(end - start)
            del output

        # Calculate metrics
        avg_hilbert = np.mean(times_hilbert)
        avg_no_hilbert = np.mean(times_no_hilbert)
        speedup = avg_no_hilbert / avg_hilbert
        improvement = (speedup - 1) * 100

        if rank == 0:
            peak_mem = torch.cuda.max_memory_allocated(rank) / 1024**3

            results_dict[seq_len] = {
                "success": True,
                "avg_time_hilbert": avg_hilbert,
                "avg_time_no_hilbert": avg_no_hilbert,
                "speedup": speedup,
                "improvement_pct": improvement,
                "peak_memory_gb": peak_mem,
            }

            print("\n  Results:")
            print(f"    With Hilbert: {avg_hilbert:.3f}s")
            print(f"    Without Hilbert: {avg_no_hilbert:.3f}s")
            print(f"    Speedup: {speedup:.3f}x ({improvement:+.1f}%)")
            print(f"    Memory: {peak_mem:.2f} GB")

    except Exception as e:
        if rank == 0:
            print(f"  Error: {str(e)}")
            results_dict[seq_len] = {"success": False, "error": str(e)}
    finally:
        dist.destroy_process_group()


def main():
    """Run benchmark with Hilbert wrapper approach."""

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("Error: This benchmark requires at least 2 GPUs")
        return

    world_size = 2

    print("=" * 80)
    print("HILBERT WRAPPER BENCHMARK (Avoiding Model Recreation)")
    print("=" * 80)

    # Test sequence lengths
    seq_lengths = [16384, 32768, 65536]

    manager = mp.Manager()
    results_dict = manager.dict()

    for seq_len in seq_lengths:
        mp.spawn(
            benchmark_with_hilbert_wrapper,
            args=(world_size, seq_len, results_dict),
            nprocs=world_size,
            join=True,
        )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Seq Len':>10} | {'Hilbert':>10} | {'No Hilbert':>10} | {'Speedup':>8} | {'Improvement':>12}"
    )
    print("-" * 65)

    for seq_len, result in results_dict.items():
        if result["success"]:
            print(
                f"{seq_len:>10,} | {result['avg_time_hilbert']:>9.3f}s | "
                f"{result['avg_time_no_hilbert']:>10.3f}s | {result['speedup']:>7.3f}x | "
                f"{result['improvement_pct']:>10.1f}%"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
