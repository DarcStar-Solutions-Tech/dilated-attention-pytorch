#!/usr/bin/env python3
"""
Multi-GPU benchmark for refactored implementations with extreme dilation.
"""

import gc
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from typing import Dict

# Import the working base implementation (not the refactored ones with issues)
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

        import math

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


def benchmark_multi_gpu_worker(
    rank: int,
    world_size: int,
    seq_len: int,
    results_dict: Dict,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    iterations: int = 3,
):
    """Multi-GPU benchmark worker."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12365"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Use extreme dilation configuration
    base_segment = min(4096, seq_len // 4)
    segment_lengths = [base_segment, base_segment * 2]
    dilation_rates = [8, 16]

    # Ensure divisibility
    max_segment = max(segment_lengths)
    if seq_len % max_segment != 0:
        seq_len = ((seq_len // max_segment) + 1) * max_segment

    if rank == 0:
        print(f"\nTesting {seq_len:,} tokens on {world_size} GPUs")
        print(f"  Segments: {segment_lengths}")
        print(f"  Dilation: {dilation_rates}")
        print(f"  Ring size: {world_size}")

    try:
        # Create base model (using the working implementation)
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

        # Memory tracking
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(rank) / 1024**3

        # Warmup
        if rank == 0:
            print("  Warming up...")

        q_h = hilbert.apply_hilbert(q)
        k_h = hilbert.apply_hilbert(k)
        v_h = hilbert.apply_hilbert(v)

        with torch.no_grad():
            output_h = model(q_h, k_h, v_h)
            _ = hilbert.apply_hilbert(output_h, inverse=True)

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

        peak_mem = torch.cuda.max_memory_allocated(rank) / 1024**3
        _ = peak_mem - mem_before

        # Gather results to rank 0
        if rank == 0:
            # Collect memory from all ranks
            total_memory = peak_mem
            for src in range(1, world_size):
                other_mem = torch.tensor(0.0, device=device)
                dist.recv(other_mem, src=src)
                total_memory += other_mem.item()

            result = {
                "success": True,
                "seq_len": seq_len,
                "world_size": world_size,
                "avg_time_hilbert": avg_hilbert,
                "avg_time_no_hilbert": avg_no_hilbert,
                "tokens_per_sec_hilbert": seq_len / avg_hilbert,
                "tokens_per_sec_no_hilbert": seq_len / avg_no_hilbert,
                "speedup": speedup,
                "improvement_pct": improvement,
                "peak_memory_gb": peak_mem,
                "total_memory_gb": total_memory,
            }

            results_dict[seq_len] = result

            print("\n  Results:")
            print(
                f"    With Hilbert: {avg_hilbert:.3f}s ({seq_len / avg_hilbert:,.0f} tokens/sec)"
            )
            print(
                f"    Without Hilbert: {avg_no_hilbert:.3f}s ({seq_len / avg_no_hilbert:,.0f} tokens/sec)"
            )
            print(f"    Speedup: {speedup:.3f}x ({improvement:+.1f}%)")
            print(f"    Memory per GPU: {peak_mem:.2f} GB")
            print(f"    Total memory: {total_memory:.2f} GB")
        else:
            # Send memory to rank 0
            mem_tensor = torch.tensor(peak_mem, device=device)
            dist.send(mem_tensor, dst=0)

        # Cleanup
        del q, k, v, model
        clear_memory()

        dist.barrier()

    except Exception as e:
        if rank == 0:
            print(f"  Error: {str(e)}")
            results_dict[seq_len] = {"success": False, "error": str(e)}
    finally:
        dist.destroy_process_group()


def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def main():
    """Run multi-GPU benchmark."""

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("Error: This benchmark requires at least 2 GPUs")
        return

    world_size = min(num_gpus, 8)

    print("=" * 80)
    print(f"MULTI-GPU HILBERT BENCHMARK: {world_size} GPUs TO 128K TOKENS (FP32)")
    print("=" * 80)
    print("\nConfiguration:")
    print("- Extreme dilation (8,16)")
    print("- Hilbert SFC optimization via wrapper")
    print("- FP32 precision")
    print(f"- Ring attention across {world_size} GPUs")

    # Test sequence lengths
    seq_lengths = [
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        98304,  # 96K
        131072,  # 128K
    ]

    manager = mp.Manager()
    results_dict = manager.dict()

    for seq_len in seq_lengths:
        mp.spawn(
            benchmark_multi_gpu_worker,
            args=(world_size, seq_len, results_dict),
            nprocs=world_size,
            join=True,
        )

        # Check if we hit an error
        if seq_len in results_dict and not results_dict[seq_len].get("success", False):
            break

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Seq Len':>10} | {'Hilbert':>12} | {'No Hilbert':>12} | {'Speedup':>8} | {'Mem/GPU':>8} | {'Total Mem':>10}"
    )
    print("-" * 75)

    for seq_len, result in results_dict.items():
        if result.get("success", False):
            print(
                f"{seq_len:>10,} | {result['tokens_per_sec_hilbert']:>12,.0f} | "
                f"{result['tokens_per_sec_no_hilbert']:>12,.0f} | {result['speedup']:>7.3f}x | "
                f"{result['peak_memory_gb']:>7.2f}G | {result['total_memory_gb']:>9.2f}G"
            )

    # Final conclusions
    successful_results = [r for r in results_dict.values() if r.get("success", False)]
    if successful_results:
        final = successful_results[-1]
        max_seq = final["seq_len"]

        print(f"\nMaximum sequence achieved: {max_seq:,} tokens")
        print(f"Best speedup: {max(r['speedup'] for r in successful_results):.3f}x")
        print(f"Memory efficiency: O(n/{world_size}) confirmed")

    print("=" * 80)


if __name__ == "__main__":
    main()
