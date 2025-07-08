#!/usr/bin/env python3
"""Analyze current working implementations."""

import torch
import json
from datetime import datetime


def test_implementations():
    """Test and benchmark available implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    print("=" * 80)

    # Test parameters
    batch_size = 2
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    results = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "device": str(device),
        "test_config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        "implementations": {},
    }

    # Test each category
    categories = {
        "Core": ["DilatedAttention", "ImprovedDilatedAttention"],
        "Multihead": ["MultiheadDilatedAttention", "ImprovedMultiheadDilatedAttention"],
        "Block-Sparse": [
            "BlockSparseRingDilatedAttention",
            "BlockSparseRingMultiheadDilatedAttention",
            "BlockSparseRingDistributedDilatedAttention",
            "BlockSparseAdaptive",
        ],
    }

    for category, implementations in categories.items():
        print(f"\n{category} Implementations:")
        print("-" * 40)

        for impl_name in implementations:
            try:
                # Import
                module = __import__("dilated_attention_pytorch", fromlist=[impl_name])
                impl_class = getattr(module, impl_name)

                # Create instance with appropriate parameters
                if "Multihead" in impl_name:
                    model = impl_class(
                        embed_dim=num_heads * head_dim,
                        num_heads=num_heads,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        dropout=0.0,
                    ).to(device)
                elif "Distributed" in impl_name and "Block" in impl_name:
                    # Skip distributed implementations on single GPU
                    print(f"  {impl_name}: Skipped (requires multi-GPU)")
                    continue
                else:
                    model = impl_class(
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        dropout=0.0,
                    ).to(device)

                # Test forward pass
                with torch.no_grad():
                    _ = model(q, k, v)

                # Time forward pass
                if device.type == "cuda":
                    torch.cuda.synchronize()

                import time

                times = []
                for _ in range(10):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(q, k, v)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.time() - start)

                avg_time = sum(times) / len(times) * 1000  # ms

                # Memory usage
                if device.type == "cuda":
                    memory_mb = torch.cuda.memory_allocated() / 1024**2
                else:
                    memory_mb = 0

                print(f"  ✅ {impl_name}: {avg_time:.1f}ms, {memory_mb:.0f}MB")

                results["implementations"][impl_name] = {
                    "status": "working",
                    "forward_time_ms": round(avg_time, 1),
                    "memory_mb": round(memory_mb),
                }

            except Exception as e:
                print(f"  ❌ {impl_name}: {str(e)}")
                results["implementations"][impl_name] = {
                    "status": "failed",
                    "error": str(e),
                }

    # Summary
    working = [
        k for k, v in results["implementations"].items() if v["status"] == "working"
    ]
    failed = [
        k for k, v in results["implementations"].items() if v["status"] == "failed"
    ]

    print("\n" + "=" * 80)
    print(f"Summary: {len(working)} working, {len(failed)} failed")

    if working:
        print("\nWorking implementations:")
        for impl in sorted(working):
            info = results["implementations"][impl]
            print(f"  - {impl}: {info['forward_time_ms']}ms")

    # Save results
    filename = f"benchmarks/current_implementations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

    return results


if __name__ == "__main__":
    test_implementations()
