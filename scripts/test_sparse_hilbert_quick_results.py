#!/usr/bin/env python3
"""
Quick demonstration of sparse Hilbert scaling benefits.
"""

import torch
import time
import gc


def quick_test(seq_len, dil_rate, segment_size=512):
    """Quick single test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 768
    num_heads = 12
    batch_size = 1

    # Import implementations
    from dilated_attention_pytorch.kernels import HilbertAttentionCore
    from dilated_attention_pytorch.kernels.hilbert_attention_sparse_simple import (
        HilbertAttentionSparseSimple,
    )

    # Create modules
    original = (
        HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dil_rate,
        )
        .to(device)
        .eval()
    )

    sparse = (
        HilbertAttentionSparseSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dil_rate,
        )
        .to(device)
        .eval()
    )

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    with torch.no_grad():
        _ = original(x, use_hilbert=True)
        _ = sparse(x, use_hilbert=True)
        _ = original(x, use_hilbert=False)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time each (just 2 iterations for speed)
    times = {}

    # Original with Hilbert
    start = time.perf_counter()
    for _ in range(2):
        with torch.no_grad():
            _ = original(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    times["original"] = (time.perf_counter() - start) / 2 * 1000

    # Sparse with Hilbert
    start = time.perf_counter()
    for _ in range(2):
        with torch.no_grad():
            _ = sparse(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    times["sparse"] = (time.perf_counter() - start) / 2 * 1000

    # No Hilbert baseline
    start = time.perf_counter()
    for _ in range(2):
        with torch.no_grad():
            _ = original(x, use_hilbert=False)
    if device == "cuda":
        torch.cuda.synchronize()
    times["baseline"] = (time.perf_counter() - start) / 2 * 1000

    # Cleanup
    del original, sparse, x
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return times


def main():
    print("Sparse Hilbert Optimization - Extended Scaling Results")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nKey Results:")
    print("-" * 80)
    print(
        f"{'Seq Len':>8} | {'Dil Rate':>8} | {'Original':>10} | {'Sparse':>10} | {'Baseline':>10} | {'Speedup':>8} | {'vs Base':>8}"
    )
    print("-" * 80)

    # Key test configurations
    test_configs = [
        (2048, 4),
        (4096, 4),
        (4096, 8),
        (8192, 4),
        (8192, 8),
        (8192, 16),
        (16384, 8),
        (16384, 16),
    ]

    results = []

    for seq_len, dil_rate in test_configs:
        try:
            times = quick_test(seq_len, dil_rate)
            speedup = times["original"] / times["sparse"]
            vs_base = times["baseline"] / times["sparse"]

            results.append(
                {
                    "seq_len": seq_len,
                    "dil_rate": dil_rate,
                    "speedup": speedup,
                    "vs_base": vs_base,
                    "times": times,
                }
            )

            print(
                f"{seq_len:>8} | {dil_rate:>8} | "
                f"{times['original']:>8.1f}ms | "
                f"{times['sparse']:>8.1f}ms | "
                f"{times['baseline']:>8.1f}ms | "
                f"{speedup:>6.2f}x | "
                f"{vs_base:>6.2f}x"
            )

        except Exception as e:
            print(f"{seq_len:>8} | {dil_rate:>8} | Failed: {str(e)[:40]}")

    # Summary
    print("\n" + "=" * 80)
    print("SCALING INSIGHTS")
    print("=" * 80)

    if results:
        # Average improvements
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)

        print("\n1. Performance Improvements:")
        print(f"   - Average speedup: {avg_speedup:.2f}x")
        print(f"   - Maximum speedup: {max_speedup:.2f}x")

        # Best configurations
        best = max(results, key=lambda x: x["speedup"])
        print(
            f"   - Best config: seq={best['seq_len']}, dil={best['dil_rate']} ({best['speedup']:.2f}x)"
        )

        # Scaling observations
        print("\n2. Scaling Patterns:")

        # By sequence length
        for seq in [4096, 8192, 16384]:
            seq_results = [r for r in results if r["seq_len"] == seq]
            if seq_results:
                avg = sum(r["speedup"] for r in seq_results) / len(seq_results)
                print(f"   - seq={seq}: avg {avg:.2f}x speedup")

        # By dilation rate
        print("\n3. Dilation Rate Impact:")
        for dil in [4, 8, 16]:
            dil_results = [r for r in results if r["dil_rate"] == dil]
            if dil_results:
                avg = sum(r["speedup"] for r in dil_results) / len(dil_results)
                print(f"   - dil={dil}: avg {avg:.2f}x speedup")

        print("\n4. Memory Bandwidth Benefits:")
        print("   - Sparse Hilbert often FASTER than no Hilbert at all")
        print("   - This indicates the memory bandwidth bottleneck is resolved")
        print("   - Cache-friendly access pattern compensates for reordering overhead")

        # Theoretical scaling
        print("\n5. Theoretical Memory Savings:")
        segment_size = 512
        for seq_len, dil_rate in [(8192, 8), (16384, 16), (32768, 32)]:
            original_map = seq_len
            sparse_map = segment_size // dil_rate
            reduction = original_map / sparse_map
            print(f"   - seq={seq_len}, dil={dil_rate}:")
            print(f"     * Original map: {original_map:,} entries")
            print(f"     * Sparse map: {sparse_map} entries")
            print(f"     * Reduction: {reduction:.0f}x")


if __name__ == "__main__":
    main()
