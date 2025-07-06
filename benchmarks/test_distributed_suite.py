"""Comprehensive distributed testing suite."""

import argparse
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dilated_attention_pytorch import (
    RingDilatedAttentionProduction,
)
from core.base_benchmark import BaseDistributedBenchmark
from core.utils import (
    generate_qkv_data,
    MemoryMonitor,
    time_cuda_operation,
)


class DistributedTestSuite(BaseDistributedBenchmark):
    """Test suite for distributed implementations."""

    def test_distributed_setup(self) -> Dict[str, Any]:
        """Test basic distributed setup."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "device": str(self.device),
            "backend": dist.get_backend() if dist.is_initialized() else "none",
        }

    def test_ring_attention(self) -> Dict[str, Any]:
        """Test ring attention in distributed mode."""
        results = {}

        # Test configurations
        configs = [
            {
                "seq_len": 4096,
                "segment_lengths": [1024, 2048],
                "dilation_rates": [1, 2],
            },
            {
                "seq_len": 8192,
                "segment_lengths": [2048, 4096],
                "dilation_rates": [1, 2],
            },
        ]

        for i, config in enumerate(configs):
            try:
                model = RingDilatedAttentionProduction(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    dropout=0.0,
                    ring_size=self.world_size,
                ).to(self.device)

                # Generate test data
                q, k, v = generate_qkv_data(
                    batch_size=1,
                    seq_len=config["seq_len"],
                    num_heads=8,
                    head_dim=64,
                    dtype=self.dtype,
                    device=self.device,
                )

                # Time forward pass
                timing = time_cuda_operation(
                    model,
                    q,
                    k,
                    v,
                    warmup=2,
                    iterations=5,
                    device=self.device,
                )

                # Measure memory
                _, peak_memory = self.measure_memory(model, q, k, v)

                results[f"config_{i}"] = {
                    "success": True,
                    "seq_len": config["seq_len"],
                    "time_ms": timing["mean"] * 1000,
                    "memory_mb": peak_memory,
                }

            except Exception as e:
                results[f"config_{i}"] = {
                    "success": False,
                    "seq_len": config["seq_len"],
                    "error": str(e),
                }

        return results

    def test_communication_patterns(self) -> Dict[str, Any]:
        """Test ring communication patterns."""
        if not dist.is_initialized():
            return {"error": "Distributed not initialized"}

        results = {}

        # Test tensor sizes
        sizes = [1024, 4096, 16384]

        for size in sizes:
            # Create test tensor
            tensor = torch.randn(size, device=self.device)

            # Test all-reduce
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            dist.all_reduce(tensor)
            end.record()

            torch.cuda.synchronize()
            all_reduce_time = start.elapsed_time(end)

            # Test send/recv (ring pattern)
            next_rank = (self.rank + 1) % self.world_size
            prev_rank = (self.rank - 1) % self.world_size

            send_tensor = torch.randn(size, device=self.device)
            recv_tensor = torch.empty_like(send_tensor)

            start.record()
            if self.rank == 0:
                dist.send(send_tensor, next_rank)
                dist.recv(recv_tensor, prev_rank)
            else:
                dist.recv(recv_tensor, prev_rank)
                dist.send(send_tensor, next_rank)
            end.record()

            torch.cuda.synchronize()
            ring_time = start.elapsed_time(end)

            results[f"size_{size}"] = {
                "all_reduce_ms": all_reduce_time,
                "ring_ms": ring_time,
                "ratio": ring_time / all_reduce_time if all_reduce_time > 0 else 0,
            }

        return results

    def test_memory_scaling(self) -> Dict[str, Any]:
        """Test memory scaling across GPUs."""
        results = {}

        # Test different sequence lengths
        seq_lengths = [4096, 8192, 16384, 32768]

        model = RingDilatedAttentionProduction(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            dropout=0.0,
            ring_size=self.world_size,
        ).to(self.device)

        for seq_len in seq_lengths:
            try:
                # Each GPU handles seq_len // world_size tokens
                local_seq_len = seq_len // self.world_size

                q, k, v = generate_qkv_data(
                    batch_size=1,
                    seq_len=local_seq_len,
                    num_heads=8,
                    head_dim=64,
                    dtype=self.dtype,
                    device=self.device,
                )

                with MemoryMonitor(self.device) as monitor:
                    _ = model(q, k, v)

                results[f"seq_{seq_len}"] = {
                    "success": True,
                    "total_seq_len": seq_len,
                    "local_seq_len": local_seq_len,
                    "memory_mb": monitor.memory_used,
                }

            except torch.cuda.OutOfMemoryError:
                results[f"seq_{seq_len}"] = {
                    "success": False,
                    "total_seq_len": seq_len,
                    "error": "OOM",
                }
                self.cleanup_memory()

        return results

    def run(self) -> Dict[str, Any]:
        """Run all distributed tests."""
        all_results = {
            "rank": self.rank,
            "world_size": self.world_size,
        }

        # Setup tests
        all_results["setup"] = self.test_distributed_setup()

        # Ring attention tests
        if self.rank == 0:
            print("Testing ring attention...")
        all_results["ring_attention"] = self.test_ring_attention()

        # Communication tests
        if self.rank == 0:
            print("Testing communication patterns...")
        all_results["communication"] = self.test_communication_patterns()

        # Memory scaling tests
        if self.rank == 0:
            print("Testing memory scaling...")
        all_results["memory_scaling"] = self.test_memory_scaling()

        # Gather results on rank 0
        gathered_results = self.gather_results(all_results)

        # Print summary on rank 0
        if self.rank == 0 and gathered_results:
            print("\n" + "=" * 80)
            print("DISTRIBUTED TEST SUMMARY")
            print("=" * 80)

            for rank_results in gathered_results:
                rank = rank_results["rank"]
                print(f"\nRank {rank}:")

                # Ring attention results
                if "ring_attention" in rank_results:
                    print("  Ring Attention:")
                    for config_name, config_results in rank_results[
                        "ring_attention"
                    ].items():
                        if config_results["success"]:
                            print(
                                f"    {config_name}: {config_results['time_ms']:.2f} ms, "
                                f"{config_results['memory_mb']:.2f} MB"
                            )
                        else:
                            print(
                                f"    {config_name}: Failed - {config_results.get('error', 'Unknown')}"
                            )

                # Communication results
                if "communication" in rank_results:
                    print("  Communication:")
                    for size_name, size_results in rank_results[
                        "communication"
                    ].items():
                        if isinstance(size_results, dict) and "ring_ms" in size_results:
                            print(
                                f"    {size_name}: Ring={size_results['ring_ms']:.2f} ms, "
                                f"Ratio={size_results['ratio']:.2f}"
                            )

        return all_results


def run_worker(rank: int, world_size: int, args: argparse.Namespace):
    """Run worker process."""
    # Create benchmark
    benchmark = DistributedTestSuite(
        rank=rank,
        world_size=world_size,
        dtype=args.dtype,
        backend=args.backend,
    )

    # Setup distributed
    benchmark.setup_distributed()

    try:
        # Run tests
        benchmark.run()
    finally:
        # Cleanup
        benchmark.cleanup_distributed()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed test suite")
    parser.add_argument(
        "--world-size", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend",
    )
    parser.add_argument(
        "--dtype", type=torch.dtype, default=torch.float32, help="Data type"
    )

    args = parser.parse_args()

    if args.world_size == 1:
        # Single GPU
        run_worker(0, 1, args)
    else:
        # Multi-GPU
        mp.spawn(
            run_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
