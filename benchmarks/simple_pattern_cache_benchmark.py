"""
Simple benchmark to demonstrate pattern caching performance improvements.
"""

import time
import torch
import gc
from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention
from dilated_attention_pytorch.core import clear_global_cache, get_global_pattern_cache


def benchmark_pattern_caching():
    """Run a simple benchmark to show pattern caching benefits."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print("-" * 80)

    # Test configurations
    configs = [
        {
            "name": "Small (1K sequence)",
            "batch_size": 2,
            "seq_len": 1024,
            "num_heads": 8,
            "head_dim": 64,
            "segment_lengths": [256, 512],
            "dilation_rates": [1, 2],
        },
        {
            "name": "Medium (4K sequence)",
            "batch_size": 2,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "segment_lengths": [512, 1024, 2048],
            "dilation_rates": [1, 2, 4],
        },
        {
            "name": "Large (8K sequence)",
            "batch_size": 1,
            "seq_len": 8192,
            "num_heads": 8,
            "head_dim": 64,
            "segment_lengths": [1024, 2048, 4096],
            "dilation_rates": [1, 2, 4],
        },
    ]

    for attention_class in [DilatedAttention, ImprovedDilatedAttention]:
        print(f"\n{attention_class.__name__}")
        print("=" * 80)

        for config in configs:
            # Clear cache before each test
            clear_global_cache()
            cache = get_global_pattern_cache()

            # Create attention module
            attention = attention_class(
                segment_lengths=config["segment_lengths"],
                dilation_rates=config["dilation_rates"],
            )
            attention = attention.to(device)
            attention.eval()

            # Create input tensors
            q = torch.randn(
                config["batch_size"],
                config["seq_len"],
                config["num_heads"],
                config["head_dim"],
                device=device,
            )
            k = q.clone()
            v = q.clone()

            # Warmup
            for _ in range(3):
                _ = attention(q, k, v)

            if device == "cuda":
                torch.cuda.synchronize()

            # Measure cold cache (first run after clearing cache)
            clear_global_cache()
            gc.collect()

            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = attention(q, k, v)
            if device == "cuda":
                torch.cuda.synchronize()
            cold_cache_time = time.perf_counter() - start_time

            # Get cache stats after first run
            _ = cache.get_stats()

            # Measure warm cache (average of multiple runs)
            warm_times = []
            for _ in range(10):
                if device == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = attention(q, k, v)
                if device == "cuda":
                    torch.cuda.synchronize()
                warm_times.append(time.perf_counter() - start_time)

            warm_cache_time = sum(warm_times) / len(warm_times)

            # Get final cache stats
            cache_stats_final = cache.get_stats()

            # Calculate speedup
            speedup = cold_cache_time / warm_cache_time

            # Print results
            print(f"\n{config['name']}:")
            print(f"  Cold cache time: {cold_cache_time * 1000:.2f} ms")
            print(f"  Warm cache time: {warm_cache_time * 1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Patterns cached: {cache_stats_final['size']}")
            print(f"  Cache hit rate: {cache_stats_final['hit_rate']:.1%}")
            print(
                f"  Time saved per forward: {(cold_cache_time - warm_cache_time) * 1000:.2f} ms"
            )

    # Show global cache statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    cache = get_global_pattern_cache()
    final_stats = cache.get_stats()

    print(f"Total patterns cached: {final_stats['size']}")
    print(f"Total cache hits: {final_stats['hits']}")
    print(f"Total cache misses: {final_stats['misses']}")
    print(f"Overall hit rate: {final_stats['hit_rate']:.1%}")
    print(f"Cache evictions: {final_stats['evictions']}")

    print("\nPattern caching is most beneficial when:")
    print("- Running multiple forward passes with the same configuration")
    print("- Using consistent sequence lengths and attention parameters")
    print("- Memory usage for caching patterns is acceptable")

    print("\nNote: The speedup is most noticeable on GPU where the relative")
    print("cost of creating indices is higher compared to CPU.")


if __name__ == "__main__":
    benchmark_pattern_caching()
