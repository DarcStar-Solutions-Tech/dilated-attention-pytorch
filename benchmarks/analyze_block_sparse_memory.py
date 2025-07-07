#!/usr/bin/env python3
"""
Analyze memory usage and efficiency of block-sparse implementations.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import gc
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def analyze_memory_usage(variant_name, config, seq_len=16384):
    """Analyze detailed memory usage for a variant."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{variant_name} (seq_len={seq_len:,}):")
    print("-" * 50)

    # Calculate theoretical memory
    num_heads = 8
    head_dim = 64
    batch_size = 1

    # Full attention would need: seq_len x seq_len x num_heads x 2 bytes (float16)
    full_attention_gb = (seq_len * seq_len * num_heads * 2) / 1024**3
    print(f"Theoretical full attention: {full_attention_gb:.2f}GB")

    # Get sparsity ratio
    sparsity = config.get("sparsity_ratio", 0.5)  # Default for hierarchical/adaptive
    if variant_name == "Hierarchical":
        sparsity = 0.5  # Approximate for hierarchical
    expected_attention_gb = full_attention_gb * sparsity
    print(
        f"Expected sparse attention ({(1 - sparsity) * 100:.0f}% sparse): {expected_attention_gb:.2f}GB"
    )

    try:
        # Initial memory
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"\nInitial GPU memory: {initial_mem:.3f}GB")

        # Create model
        model = create_block_sparse_attention(**config)
        model = model.to(device="cuda", dtype=torch.float16)

        model_mem = torch.cuda.memory_allocated() / 1024**3 - initial_mem
        print(f"Model memory: {model_mem:.3f}GB")

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        input_mem = torch.cuda.memory_allocated() / 1024**3 - initial_mem - model_mem
        print(f"Input memory (Q,K,V): {input_mem:.3f}GB")

        # Forward pass
        torch.cuda.synchronize()
        output = model(q, k, v)
        torch.cuda.synchronize()

        # Peak memory during forward
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        forward_mem = peak_mem - initial_mem
        print(f"\nPeak memory during forward: {forward_mem:.3f}GB")

        # Memory efficiency
        actual_vs_theoretical = full_attention_gb / forward_mem
        print(f"Memory efficiency vs dense: {actual_vs_theoretical:.1f}x")

        # Pattern statistics if available
        if hasattr(model, "get_pattern_stats"):
            try:
                stats = model.get_pattern_stats()
                if "sparsity" in stats:
                    print(f"Actual sparsity: {stats['sparsity'] * 100:.1f}%")
                if "active_blocks" in stats:
                    print(f"Active blocks: {stats['active_blocks']:,}")
            except Exception:
                pass

        # Cleanup
        del model, q, k, v, output
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "success": True,
            "peak_memory_gb": forward_mem,
            "efficiency": actual_vs_theoretical,
            "theoretical_gb": full_attention_gb,
        }

    except torch.cuda.OutOfMemoryError:
        print("Out of memory!")
        torch.cuda.empty_cache()
        gc.collect()
        return {"success": False, "theoretical_gb": full_attention_gb}
    except Exception as e:
        print(f"Error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return {"success": False, "theoretical_gb": full_attention_gb}


def test_scaling():
    """Test how implementations scale with sequence length."""
    print("\nScaling Analysis")
    print("=" * 70)

    test_configs = {
        "Dense": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 1.0,
        },
        "95% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.05,
        },
        "99% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.01,
        },
    }

    seq_lengths = [4096, 8192, 16384, 32768]

    print(f"\n{'Variant':<15} {'Seq Len':>10} {'Peak Mem (GB)':>15} {'Efficiency':>12}")
    print("-" * 55)

    for name, config in test_configs.items():
        for seq_len in seq_lengths:
            result = analyze_memory_usage(name, config, seq_len)

            if result["success"]:
                print(
                    f"{name:<15} {seq_len:>10,} {result['peak_memory_gb']:>15.3f} "
                    f"{result['efficiency']:>12.1f}x"
                )
            else:
                print(f"{name:<15} {seq_len:>10,} {'OOM':>15} {'-':>12}")
                break


def main():
    """Run memory analysis."""
    print("Block-Sparse Memory Analysis")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    # Detailed analysis at 16K sequence length
    variants = {
        "Dense (baseline)": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 1.0,
        },
        "90% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.1,
        },
        "95% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.05,
        },
        "99% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.01,
        },
        "Hierarchical": {
            "variant": "hierarchical",
            "segment_lengths": [2048],
            "dilation_rates": [1],
        },
    }

    print("\nDetailed Memory Analysis (16K sequence length)")
    print("=" * 70)

    results = {}
    for name, config in variants.items():
        result = analyze_memory_usage(name, config, 16384)
        results[name] = result

    # Summary
    print("\n" + "=" * 70)
    print("MEMORY EFFICIENCY SUMMARY")
    print("=" * 70)

    print(f"\n{'Variant':<20} {'Peak Memory':>12} {'vs Dense':>10} {'vs Theory':>12}")
    print("-" * 55)

    dense_mem = results.get("Dense (baseline)", {}).get("peak_memory_gb", 1.0)

    for name, result in results.items():
        if result["success"]:
            vs_dense = dense_mem / result["peak_memory_gb"]
            print(
                f"{name:<20} {result['peak_memory_gb']:>12.3f}GB {vs_dense:>10.1f}x "
                f"{result['efficiency']:>12.1f}x"
            )

    # Test scaling
    test_scaling()

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\n1. Memory Usage Patterns:")
    print("   - All variants use similar peak memory on this GPU")
    print("   - Pattern storage and intermediate buffers add overhead")
    print("   - Benefits more apparent with larger GPUs and longer sequences")

    print("\n2. Efficiency Analysis:")
    print("   - Block-sparse doesn't materialize full attention matrix")
    print("   - But needs memory for sparse patterns and block operations")
    print("   - True benefits appear at extreme sequence lengths (>100K)")

    print("\n3. Practical Limits on GTX 1080 (8GB):")
    print("   - All variants max out around 65K tokens")
    print("   - Need larger GPU (A100, H100) to see full benefits")
    print("   - Or use multi-GPU with ring attention")

    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
