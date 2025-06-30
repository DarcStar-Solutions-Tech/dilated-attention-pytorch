"""
Quick dual-GPU benchmark comparing Ring V2 vs V3.

Tests simulated ring mode on 2 GPUs to see if V3's optimizations
help in multi-GPU scenarios.
"""

import torch
import time
from typing import Dict

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def benchmark_on_gpu(
    gpu_id: int,
    implementation: str,
    seq_len: int,
    ring_size: int,
    enable_cache: bool = True,
) -> Dict:
    """Benchmark on specific GPU."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Configuration
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Create model
    kwargs = {
        "segment_lengths": segment_lengths,
        "dilation_rates": dilation_rates,
        "ring_size": ring_size,
        "device": device,
        "dtype": torch.float16,
        "enable_memory_pool": True,
        "use_pattern_cache": enable_cache,
    }

    if implementation == "v2":
        model = RingDilatedAttentionV2(**kwargs)
    else:
        kwargs["cache_on_gpu"] = True
        model = RingDilatedAttentionV3(**kwargs)

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        _ = model(q, k, v)

    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated(device) / 1024 / 1024

    start_time = time.time()
    for _ in range(10):
        _ = model(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    memory_mb = peak_mem - start_mem
    time_ms = (end_time - start_time) * 100  # Average per iteration

    # Cleanup
    del model, q, k, v
    torch.cuda.empty_cache()

    return {
        "gpu_id": gpu_id,
        "implementation": implementation,
        "seq_len": seq_len,
        "ring_size": ring_size,
        "mode": "simulated" if ring_size > 1 else "single",
        "time_ms": time_ms,
        "memory_mb": memory_mb,
        "cache_enabled": enable_cache,
    }


def main():
    print("Ring Attention V2 vs V3 - Dual GPU Benchmark")
    print("=" * 60)

    sequence_lengths = [4096, 8192]
    ring_sizes = [1, 2]  # Single GPU vs Simulated 2-GPU ring

    results = []

    for seq_len in sequence_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        for ring_size in ring_sizes:
            for impl in ["v2", "v3"]:
                # Test on GPU 0
                try:
                    result = benchmark_on_gpu(
                        0, impl, seq_len, ring_size, enable_cache=True
                    )
                    results.append(result)
                    print(
                        f"  {impl.upper()} - Ring-{ring_size} (GPU 0): "
                        f"{result['time_ms']:.1f}ms, {result['memory_mb']:.1f}MB"
                    )
                except Exception as e:
                    print(f"  {impl.upper()} - Ring-{ring_size} (GPU 0): FAILED - {e}")

                # If ring_size=2, also test on GPU 1
                if ring_size == 2:
                    try:
                        result = benchmark_on_gpu(
                            1, impl, seq_len, ring_size, enable_cache=True
                        )
                        results.append(result)
                        print(
                            f"  {impl.upper()} - Ring-{ring_size} (GPU 1): "
                            f"{result['time_ms']:.1f}ms, {result['memory_mb']:.1f}MB"
                        )
                    except Exception as e:
                        print(
                            f"  {impl.upper()} - Ring-{ring_size} (GPU 1): FAILED - {e}"
                        )

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    for seq_len in sequence_lengths:
        print(f"\nSequence {seq_len}:")

        # Compare single GPU performance
        v2_single = next(
            (
                r
                for r in results
                if r["implementation"] == "v2"
                and r["seq_len"] == seq_len
                and r["ring_size"] == 1
            ),
            None,
        )
        v3_single = next(
            (
                r
                for r in results
                if r["implementation"] == "v3"
                and r["seq_len"] == seq_len
                and r["ring_size"] == 1
            ),
            None,
        )

        if v2_single and v3_single:
            speedup = v2_single["time_ms"] / v3_single["time_ms"]
            print(
                f"  Single GPU: V3 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than V2"
            )

        # Compare simulated ring performance
        v2_ring = [
            r
            for r in results
            if r["implementation"] == "v2"
            and r["seq_len"] == seq_len
            and r["ring_size"] == 2
        ]
        v3_ring = [
            r
            for r in results
            if r["implementation"] == "v3"
            and r["seq_len"] == seq_len
            and r["ring_size"] == 2
        ]

        if v2_ring and v3_ring:
            v2_avg = sum(r["time_ms"] for r in v2_ring) / len(v2_ring)
            v3_avg = sum(r["time_ms"] for r in v3_ring) / len(v3_ring)
            speedup = v2_avg / v3_avg
            print(
                f"  Ring-2 (avg): V3 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than V2"
            )

            # Memory comparison
            v2_mem = sum(r["memory_mb"] for r in v2_ring) / len(v2_ring)
            v3_mem = sum(r["memory_mb"] for r in v3_ring) / len(v3_ring)
            print(f"  Memory: V2={v2_mem:.1f}MB, V3={v3_mem:.1f}MB")

    print("\nConclusion:")
    print("-" * 40)

    # Overall performance comparison
    v2_times = [r["time_ms"] for r in results if r["implementation"] == "v2"]
    v3_times = [r["time_ms"] for r in results if r["implementation"] == "v3"]

    if v2_times and v3_times:
        v2_avg = sum(v2_times) / len(v2_times)
        v3_avg = sum(v3_times) / len(v3_times)
        overall_speedup = v2_avg / v3_avg

        print(
            f"Overall: V3 is {overall_speedup:.2f}x {'faster' if overall_speedup > 1 else 'slower'} than V2"
        )

        if overall_speedup < 1:
            print("\nV3's complex caching adds overhead that outweighs benefits,")
            print(
                "even in simulated multi-GPU scenarios. V2 remains the better choice."
            )
        else:
            print("\nV3 shows improvement in multi-GPU scenarios.")


if __name__ == "__main__":
    main()
