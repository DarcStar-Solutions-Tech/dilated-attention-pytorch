#!/usr/bin/env python3
"""
Fixed distributed benchmark for Ring Hilbert Attention.
Addresses tensor contiguity and memory access issues.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import time
from typing import Dict, List
import argparse
import json
from datetime import datetime
import math


class DistributedRingAttentionFixed(nn.Module):
    """
    Fixed Distributed Ring Attention implementation.
    Ensures tensor contiguity and proper memory management.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        use_hilbert: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.use_hilbert = use_hilbert

        # Distributed settings
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(f"cuda:{self.rank}")

        # QKV projection
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Move to device
        self.to(self.device)

        # Cache for Hilbert mappings
        self._hilbert_cache = {}

    def _generate_proper_hilbert_curve(self, n: int) -> torch.Tensor:
        """Generate proper 2D Hilbert curve mapping."""
        # Find appropriate power of 2 size
        size = 1
        while size * size < n:
            size *= 2

        def hilbert_index_to_xy(index: int, n: int) -> tuple:
            """Convert Hilbert index to (x, y) coordinates."""
            x = y = 0
            s = 1
            while s < n:
                rx = 1 if (index // 2) % 2 else 0
                ry = 1 if (index ^ rx) % 2 else 0
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                index //= 4
                s *= 2
            return x, y

        # Create forward mapping (linear -> hilbert)
        forward_map = torch.zeros(n, dtype=torch.long)
        hilbert_to_linear = {}

        # Generate all points in Hilbert order
        for hilbert_idx in range(min(n, size * size)):
            x, y = hilbert_index_to_xy(hilbert_idx, size)
            linear_idx = y * size + x
            if linear_idx < n:
                hilbert_to_linear[hilbert_idx] = linear_idx

        # Create the mapping
        hilbert_idx = 0
        for i in sorted(hilbert_to_linear.keys()):
            if hilbert_idx < n:
                linear_idx = hilbert_to_linear[i]
                forward_map[linear_idx] = hilbert_idx
                hilbert_idx += 1

        return forward_map.to(self.device)

    def _apply_hilbert_ordering(
        self, tensor: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Apply Hilbert ordering ensuring contiguous output."""
        batch_size, seq_len, hidden_dim = tensor.shape

        if seq_len not in self._hilbert_cache:
            self._hilbert_cache[seq_len] = self._generate_proper_hilbert_curve(seq_len)

        mapping = self._hilbert_cache[seq_len]

        if inverse:
            # Create inverse mapping
            inverse_mapping = torch.argsort(mapping)
            result = tensor.gather(
                1,
                inverse_mapping.unsqueeze(0)
                .unsqueeze(-1)
                .expand(batch_size, -1, hidden_dim),
            )
        else:
            result = tensor.gather(
                1, mapping.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim)
            )

        # Ensure contiguous
        return result.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fixed distributed ring attention forward pass.
        Ensures all tensors are contiguous for P2P operations.
        """
        batch_size, seq_len, _ = x.shape

        # Ensure input is contiguous
        x = x.contiguous()

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]

        # Apply Hilbert ordering if enabled
        if self.use_hilbert:
            # Reshape for ordering
            q_flat = q_local.transpose(1, 2).reshape(batch_size, seq_len, -1)
            k_flat = k_local.transpose(1, 2).reshape(batch_size, seq_len, -1)
            v_flat = v_local.transpose(1, 2).reshape(batch_size, seq_len, -1)

            # Apply ordering (returns contiguous tensors)
            q_flat = self._apply_hilbert_ordering(q_flat)
            k_flat = self._apply_hilbert_ordering(k_flat)
            v_flat = self._apply_hilbert_ordering(v_flat)

            # Reshape back
            q_local = (
                q_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            k_local = (
                k_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            v_local = (
                v_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        # Initialize output
        output = torch.zeros_like(q_local)

        # Online softmax state
        running_max = torch.full(
            (batch_size, self.num_heads, seq_len, 1),
            float("-inf"),
            device=self.device,
            dtype=q_local.dtype,
        )
        running_sum = torch.zeros_like(running_max)

        # Ring attention with fixed communication
        if self.world_size > 1:
            # Use all_gather instead of P2P for more robust communication
            # Gather all K and V chunks
            k_chunks = [torch.empty_like(k_local) for _ in range(self.world_size)]
            v_chunks = [torch.empty_like(v_local) for _ in range(self.world_size)]

            # Ensure tensors are contiguous before all_gather
            k_local_contig = k_local.contiguous()
            v_local_contig = v_local.contiguous()

            dist.all_gather(k_chunks, k_local_contig)
            dist.all_gather(v_chunks, v_local_contig)

            # Process all chunks
            for i in range(self.world_size):
                k_chunk = k_chunks[i]
                v_chunk = v_chunks[i]

                # Compute attention scores
                scores = torch.matmul(q_local, k_chunk.transpose(-2, -1)) / math.sqrt(
                    self.head_dim
                )

                # Apply causal mask if needed (optional)
                # if self.causal:
                #     mask = torch.triu(torch.ones_like(scores), diagonal=1)
                #     scores = scores.masked_fill(mask > 0, float('-inf'))

                # Online softmax update
                scores_max = scores.max(dim=-1, keepdim=True)[0]
                new_max = torch.maximum(running_max, scores_max)

                # Recompute old weights
                old_weight = torch.exp(running_max - new_max)
                output = output * old_weight
                running_sum = running_sum * old_weight

                # Add new contribution
                scores_exp = torch.exp(scores - new_max)
                output = output + torch.matmul(scores_exp, v_chunk)
                running_sum = running_sum + scores_exp.sum(dim=-1, keepdim=True)

                # Update running max
                running_max = new_max
        else:
            # Single GPU case
            scores = torch.matmul(q_local, k_local.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v_local)
            running_sum = torch.ones_like(running_max)

        # Normalize output
        output = output / (running_sum + 1e-8)

        # Reverse Hilbert ordering if applied
        if self.use_hilbert:
            output_flat = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
            output_flat = self._apply_hilbert_ordering(output_flat, inverse=True)
            output = (
                output_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        # Final reshape and projection
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


def init_distributed():
    """Initialize distributed process group."""
    if dist.is_initialized():
        print(f"Process {dist.get_rank()}: Already initialized")
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])

        # Set device before init
        torch.cuda.set_device(local_rank)

        # Initialize
        dist.init_process_group(backend="nccl", init_method="env://")

        print(
            f"Process {rank}: Initialized distributed (world_size={world_size}, device=cuda:{local_rank})"
        )
    else:
        print(
            "Warning: Distributed environment not detected, running in single GPU mode"
        )


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def benchmark_configuration(
    hidden_dim: int,
    num_heads: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    batch_size: int,
    seq_len: int,
    warmup: int = 5,
    iterations: int = 20,
) -> Dict[str, float]:
    """Benchmark with proper error handling."""

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{rank}")

    try:
        # Create models
        model_standard = DistributedRingAttentionFixed(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=False,
        )

        model_hilbert = DistributedRingAttentionFixed(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_hilbert=True,
        )

        # Create input
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32
        )

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model_standard(x)
                _ = model_hilbert(x)

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Benchmark standard
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model_standard(x)

        torch.cuda.synchronize()
        standard_time = (time.perf_counter() - start) / iterations * 1000

        if dist.is_initialized():
            dist.barrier()

        # Benchmark Hilbert
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model_hilbert(x)

        torch.cuda.synchronize()
        hilbert_time = (time.perf_counter() - start) / iterations * 1000

        # Gather times from all ranks
        if dist.is_initialized():
            times_tensor = torch.tensor([standard_time, hilbert_time], device=device)
            dist.all_reduce(times_tensor, op=dist.ReduceOp.AVG)
            standard_time, hilbert_time = times_tensor.tolist()

        return {
            "standard_time_ms": standard_time,
            "hilbert_time_ms": hilbert_time,
            "speedup": standard_time / hilbert_time,
            "world_size": world_size,
        }

    except Exception as e:
        if rank == 0:
            print(f"Error in benchmark: {e}")
        return None


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    # Initialize distributed
    init_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("\n=== Fixed Distributed Ring Hilbert Attention Benchmark ===")
        print(f"Running on {world_size} GPU(s)")
        print(
            f"Backend: {dist.get_backend() if dist.is_initialized() else 'single-gpu'}"
        )
        print()

    # Test configurations (conservative for GTX 1080)
    if args.quick:
        configs = [
            # (hidden_dim, num_heads, segment_lengths, dilation_rates, batch_size, seq_len)
            (256, 4, [256], [1], 2, 1024),
            (256, 4, [256], [2], 2, 1024),
            (512, 8, [512], [1], 1, 2048),
        ]
    else:
        configs = [
            (256, 4, [256], [1], 4, 1024),
            (256, 4, [256], [2], 4, 1024),
            (256, 4, [256], [4], 4, 1024),
            (512, 8, [512], [1], 2, 2048),
            (512, 8, [512], [2], 2, 2048),
            (512, 8, [512], [4], 2, 2048),
        ]

    results = []

    if rank == 0:
        print(
            "Configuration                               | Standard (ms) | Hilbert (ms) | Speedup"
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
        result = benchmark_configuration(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            batch_size=batch_size,
            seq_len=seq_len,
            warmup=3 if args.quick else 5,
            iterations=10 if args.quick else 20,
        )

        if result and rank == 0:
            print(
                f"H={hidden_dim:3} h={num_heads} L={seq_len:4} seg={segment_lengths[0]:3} dil={dilation_rates[0]} B={batch_size} | "
                f"{result['standard_time_ms']:13.2f} | {result['hilbert_time_ms']:12.2f} | "
                f"{result['speedup']:7.2f}"
            )

            results.append(
                {
                    "config": {
                        "hidden_dim": hidden_dim,
                        "num_heads": num_heads,
                        "segment_lengths": segment_lengths,
                        "dilation_rates": dilation_rates,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                    },
                    "metrics": result,
                }
            )

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 85)
        print("SUMMARY")
        print("=" * 85)

        speedups = [r["metrics"]["speedup"] for r in results if r["metrics"]]
        if speedups:
            print(f"\nResults across {world_size} GPU(s):")
            print(f"  Average speedup: {np.mean(speedups):.2f}x")
            print(f"  Maximum speedup: {max(speedups):.2f}x")
            print(f"  Minimum speedup: {min(speedups):.2f}x")
            print(
                f"  Configs with speedup > 1: {sum(1 for s in speedups if s > 1)}/{len(speedups)}"
            )

            # Save results
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
            filename = f"ring_hilbert_fixed_{world_size}gpu_{timestamp}.json"

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

            print("\n" + "=" * 85)
            print("KEY IMPROVEMENTS IN THIS VERSION:")
            print("=" * 85)
            print("""
            1. Fixed tensor contiguity issues for P2P operations
            2. Replaced isend/irecv with all_gather for stability
            3. Proper Hilbert curve generation (not just snake pattern)
            4. Better memory management and error handling
            5. Conservative configurations for GTX 1080 memory limits
            """)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
