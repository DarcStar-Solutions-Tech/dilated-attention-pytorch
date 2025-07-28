#!/usr/bin/env python3
"""
Benchmark BlockSparseRingAttention to verify true ring attention implementation.

This benchmark tests:
1. Memory scaling with O(n/k) pattern
2. Ring communication via isend/irecv
3. Performance with multiple GPUs
4. Comparison with theoretical expectations
"""

import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.ring.block_sparse_ring_attention import (
    BlockSparseRingAttention,
)
from dilated_attention_pytorch.ring.base.ring_config import RingAttentionConfig

# Import benchmark utilities directly to avoid circular imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
from base_benchmark import BaseBenchmark
from utils.timing import time_with_events
from utils.memory import get_memory_stats
from utils.distributed import get_rank, get_world_size, barrier, is_main_process


class BlockSparseRingAttentionBenchmark(BaseBenchmark):
    """Benchmark for BlockSparseRingAttention."""

    def __init__(self, config: Dict):
        """Initialize benchmark with configuration."""
        rank = get_rank()
        super().__init__(
            device=torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"),
            dtype=config.get("dtype", torch.float32),
            warmup_iterations=config.get("warmup", 3),
            benchmark_iterations=config.get("iterations", 10),
        )
        self.config = config
        self.rank = rank
        self.world_size = get_world_size()

    def setup_model(
        self, block_size: int, sparsity_ratio: float
    ) -> BlockSparseRingAttention:
        """Create BlockSparseRingAttention model."""
        ring_config = RingAttentionConfig(
            segment_lengths=[block_size],
            dilation_rates=[1],
            dropout=0.0,
            log_communication_stats=True,
        )

        model = BlockSparseRingAttention(
            config=ring_config,
            block_size=block_size,
            sparsity_ratio=sparsity_ratio,
            pattern_type=self.config.get("pattern_type", "local"),
            device=self.device,
            dtype=self.dtype,
        )

        return model

    def benchmark_sequence_length(
        self,
        model: BlockSparseRingAttention,
        seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> Dict:
        """Benchmark a specific sequence length."""
        # Ensure divisibility
        block_size = model.block_size
        seq_len = (seq_len // (self.world_size * block_size)) * (
            self.world_size * block_size
        )

        if seq_len == 0:
            return None

        local_seq_len = seq_len // self.world_size

        # Create local tensors
        q = torch.randn(
            batch_size,
            local_seq_len,
            num_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Clear memory stats
        self.cleanup_memory()
        model.reset_communication_stats()

        # Measure memory before
        mem_before = get_memory_stats(self.device)

        # Time forward pass with CUDA events
        timing_stats = time_with_events(
            lambda: model(q, k, v, already_split=True),
            warmup=self.warmup_iterations,
            iterations=self.benchmark_iterations,
            device=self.device,
        )

        # Measure memory after
        mem_after = get_memory_stats(self.device)

        # Get communication stats
        comm_stats = model.get_communication_stats()

        # Calculate metrics
        memory_used_gb = (mem_after["allocated"] - mem_before["allocated"]) / 1024**3
        total_tokens = batch_size * seq_len * num_heads
        mem_per_token_mb = memory_used_gb * 1024 / total_tokens

        result = {
            "seq_len": seq_len,
            "local_seq_len": local_seq_len,
            "time_ms": timing_stats["mean"],
            "time_std_ms": timing_stats["std"],
            "memory_gb": memory_used_gb,
            "peak_memory_gb": mem_after["peak"] / 1024**3,
            "mem_per_token_mb": mem_per_token_mb,
            "comm_sends": comm_stats.get("sends", 0),
            "comm_bytes_mb": comm_stats.get("total_bytes", 0) / 1024**2,
            "comm_bandwidth_gbps": comm_stats.get("bandwidth_gbps", 0),
            "tokens_per_sec": total_tokens / (timing_stats["mean"] / 1000),
        }

        return result

    def run_benchmark(self) -> List[Dict]:
        """Run full benchmark suite."""
        results = []

        # Model configuration
        block_size = self.config.get("block_size", 256)
        sparsity_ratio = self.config.get("sparsity_ratio", 0.9)

        # Create model
        model = self.setup_model(block_size, sparsity_ratio)

        if is_main_process():
            print(f"\n{'=' * 80}")
            print(f"BlockSparseRingAttention Benchmark - {datetime.now()}")
            print(f"{'=' * 80}")
            print("Configuration:")
            print(f"  World size: {self.world_size} GPUs")
            print(f"  Block size: {block_size}")
            print(f"  Sparsity: {sparsity_ratio * 100:.0f}%")
            print(f"  Pattern: {self.config.get('pattern_type', 'local')}")
            print(f"  Dtype: {self.dtype}")
            print(f"{'=' * 80}\n")

        # Test different sequence lengths
        seq_lengths = self.config.get("seq_lengths", [4096, 8192, 16384, 32768])
        batch_size = self.config.get("batch_size", 2)
        num_heads = self.config.get("num_heads", 8)
        head_dim = self.config.get("head_dim", 64)

        for seq_len in seq_lengths:
            result = self.benchmark_sequence_length(
                model, seq_len, batch_size, num_heads, head_dim
            )

            if result is None:
                continue

            results.append(result)

            # Synchronize and report
            barrier()

            if is_main_process():
                print(f"Sequence length: {result['seq_len']:,} tokens")
                print(f"  Local seq/GPU: {result['local_seq_len']:,}")
                print(
                    f"  Time: {result['time_ms']:.2f} ± {result['time_std_ms']:.2f} ms"
                )
                print(f"  Memory: {result['memory_gb']:.3f} GB/GPU")
                print(f"  Peak memory: {result['peak_memory_gb']:.3f} GB/GPU")
                print(f"  Memory/token: {result['mem_per_token_mb']:.3f} MB")
                print(f"  Throughput: {result['tokens_per_sec'] / 1e6:.2f}M tokens/sec")
                print(f"  Communications: {result['comm_sends']}")
                print(f"  Comm bandwidth: {result['comm_bandwidth_gbps']:.1f} GB/s")
                print()

        return results

    def run(self) -> Dict:
        """Run the benchmark (required by BaseBenchmark)."""
        results = self.run_benchmark()
        self.verify_ring_attention(results)
        return {"results": results}

    def verify_ring_attention(self, results: List[Dict]) -> None:
        """Verify that ring attention is working correctly."""
        if not is_main_process() or not results:
            return

        print(f"\n{'=' * 80}")
        print("RING ATTENTION VERIFICATION")
        print(f"{'=' * 80}\n")

        # 1. Verify O(n/k) memory scaling
        print("1. Memory Scaling Analysis:")
        print("   (Memory per GPU should remain ~constant with O(n/k) scaling)")

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            seq_ratio = curr["seq_len"] / prev["seq_len"]
            mem_ratio = curr["memory_gb"] / prev["memory_gb"]

            print(
                f"\n   Seq {prev['seq_len']:,} → {curr['seq_len']:,} (×{seq_ratio:.1f})"
            )
            print(
                f"   Memory {prev['memory_gb']:.3f} → {curr['memory_gb']:.3f} GB (×{mem_ratio:.2f})"
            )

            # With O(n/k) scaling, memory should stay roughly constant
            if abs(mem_ratio - 1.0) < 0.3:  # Allow 30% variance
                print("   ✓ O(n/k) scaling confirmed")
            else:
                print("   ✗ Memory scaling doesn't match O(n/k) pattern")

        # 2. Verify ring communication
        print("\n2. Ring Communication Verification:")

        expected_comms_per_iter = (self.world_size - 1) * 2  # K and V passes
        expected_total = expected_comms_per_iter * self.benchmark_iterations

        for result in results:
            actual_comms = result["comm_sends"]
            print(f"\n   Seq {result['seq_len']:,}:")
            print(f"   Expected comms: ~{expected_total} ({self.world_size} GPUs)")
            print(f"   Actual comms: {actual_comms}")

            if actual_comms > 0:
                print("   ✓ Ring communication ACTIVE (isend/irecv)")
            else:
                print("   ✗ No ring communication detected")

        # 3. Memory efficiency analysis
        print("\n3. Memory Efficiency:")

        model = self.setup_model(
            self.config.get("block_size", 256), self.config.get("sparsity_ratio", 0.9)
        )
        savings = model.get_memory_savings()

        print(f"   Ring memory factor: {savings['ring_memory_factor']:.2f}x")
        print(f"   Sparse memory factor: {savings['sparse_memory_factor']:.2f}x")
        print(f"   Total memory reduction: {savings['total_memory_factor']:.2f}x")
        print(f"   Effective speedup: {savings['effective_speedup']:.2f}x")


def main():
    """Run BlockSparseRingAttention benchmark."""
    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Benchmark configuration
    config = {
        "seq_lengths": [4096, 8192, 16384, 32768],
        "batch_size": 2,
        "num_heads": 8,
        "head_dim": 64,
        "block_size": 256,
        "sparsity_ratio": 0.9,
        "pattern_type": "local",
        "dtype": torch.float32,  # Use float32 for stability
        "warmup": 3,
        "iterations": 10,
    }

    # Run benchmark
    benchmark = BlockSparseRingAttentionBenchmark(config)
    results = benchmark.run_benchmark()
    benchmark.verify_ring_attention(results)

    # Cleanup
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
