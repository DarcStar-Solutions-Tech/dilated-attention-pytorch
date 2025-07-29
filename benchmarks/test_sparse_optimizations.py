#!/usr/bin/env python3
"""
Test optimizations for sparse pattern performance.
"""

import torch
import time
import sys
import gc

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


class SparseOptimizedEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
    """Enhanced with sparse-specific optimizations."""

    def _get_optimal_config(self, seq_len: int):
        """Override config with sparse-specific optimizations."""
        config = {}

        # Check if we're on Pascal or newer GPU
        _ = self.compute_capability < 7

        # For sparse patterns, use adaptive configuration
        if self.dilation_rate > 1:
            effective_len = seq_len // self.dilation_rate
            sparsity = 1.0 - (1.0 / self.dilation_rate)

            # Very sparse (>= 75% sparse)
            if sparsity >= 0.75:
                config["block_m"] = 32
                config["block_n"] = 32
                config["block_d"] = min(32, self.head_dim)
                config["num_warps"] = 2
                config["use_fused_softmax"] = False

            # Moderately sparse (50-75%)
            elif sparsity >= 0.5:
                # Use asymmetric blocks for better efficiency
                if effective_len <= 2048:
                    config["block_m"] = 64
                    config["block_n"] = 32
                    config["block_d"] = min(32, self.head_dim)
                    config["num_warps"] = 4
                else:
                    config["block_m"] = 64
                    config["block_n"] = 64
                    config["block_d"] = min(64, self.head_dim)
                    config["num_warps"] = 4
                config["use_fused_softmax"] = False

            # Common sparse settings
            config["rows_per_block"] = 1
            config["fused_block_n"] = config["block_n"]
            config["enable_prefetch"] = False

            # Disable Hilbert for very sparse patterns
            self._sparse_hilbert_threshold = 8192 if sparsity >= 0.75 else 4096

            return config

        # Fall back to parent implementation for dense
        return super()._get_optimal_config(seq_len)

    def forward(
        self, x: torch.Tensor, use_hilbert: bool = True, is_causal: bool = False
    ) -> torch.Tensor:
        """Override forward to use PyTorch for very small sparse patterns."""
        B, M, D = x.shape

        # For very sparse patterns with small sequences, use PyTorch
        if self.dilation_rate >= 4 and M <= 2048:
            # Pad sequence
            M_padded = (
                (M + self.segment_size - 1) // self.segment_size
            ) * self.segment_size
            if M != M_padded:
                x = torch.nn.functional.pad(x, (0, 0, 0, M_padded - M))

            # QKV projection
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(B, M_padded, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Use PyTorch's SDPA
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            # Reshape and project
            out = out.transpose(1, 2).contiguous()
            out = out.view(B, M_padded, D)

            # Remove padding
            if M != M_padded:
                out = out[:, :M, :]

            out = self.out_proj(out)

            if self.dropout_layer is not None:
                out = self.dropout_layer(out)

            return out

        # Check if we should disable Hilbert for sparse
        if hasattr(self, "_sparse_hilbert_threshold"):
            M_padded = (
                (M + self.segment_size - 1) // self.segment_size
            ) * self.segment_size
            if M_padded <= self._sparse_hilbert_threshold:
                use_hilbert = False

        return super().forward(x, use_hilbert, is_causal)


def benchmark_sparse_optimizations():
    """Benchmark the sparse optimizations."""

    print("=== Testing Sparse Optimizations ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Test configurations focusing on problematic sparse patterns
    configs = [
        (2048, 2, "2K d=2"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (8192, 2, "8K d=2"),
        (8192, 4, "8K d=4"),
        (16384, 2, "16K d=2"),
        (16384, 4, "16K d=4"),
    ]

    # Common parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    print(
        f"{'Config':<10} | {'Unified':<10} | {'Original':<10} | {'Optimized':<10} | {'Orig Ratio':<12} | {'Opt Ratio':<12} | {'Improvement':<15}"
    )
    print("-" * 105)

    for seq_len, dilation_rate, desc in configs:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            # Create models
            unified = (
                UnifiedHilbertAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            original = (
                UnifiedHilbertAttentionOptimizedEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            optimized = (
                SparseOptimizedEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            # Create input
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
            )

            # Benchmark
            def benchmark(model, runs=5):
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(x)
                torch.cuda.synchronize()

                # Time
                times = []
                for _ in range(runs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = model(x)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)

                return sum(times) / len(times)

            unified_time = benchmark(unified)
            original_time = benchmark(original)
            optimized_time = benchmark(optimized)

            orig_ratio = original_time / unified_time
            opt_ratio = optimized_time / unified_time

            if opt_ratio < orig_ratio:
                improvement = (
                    f"{((orig_ratio - opt_ratio) / orig_ratio * 100):.0f}% better"
                )
            else:
                improvement = (
                    f"{((opt_ratio - orig_ratio) / orig_ratio * 100):.0f}% worse"
                )

            print(
                f"{desc:<10} | {unified_time:<10.2f} | {original_time:<10.2f} | {optimized_time:<10.2f} | "
                f"{orig_ratio:<12.2f}x | {opt_ratio:<12.2f}x | {improvement:<15}"
            )

        except Exception as e:
            print(f"{desc:<10} | Error: {str(e)}")


def verify_correctness():
    """Verify that optimizations produce correct results."""

    print("\n\n=== Verifying Correctness ===")

    # Test on 8K d=2 (problematic case)
    seq_len = 8192
    dilation_rate = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 1

    # Create models
    unified = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    optimized = (
        SparseOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        out_unified = unified(x)
        out_optimized = optimized(x)

    # Compare outputs
    diff = (out_unified - out_optimized).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Outputs match within tolerance")
    else:
        print("✗ Outputs differ significantly!")


def analyze_improvements():
    """Analyze what worked and what didn't."""

    print("\n\n=== Optimization Analysis ===")

    print("\n1. What Should Help:")
    print("   - Smaller blocks for high sparsity")
    print("   - Asymmetric blocks for moderate sparsity")
    print("   - Disabling fused softmax for sparse")
    print("   - Using PyTorch for very small sparse")
    print("   - Disabling Hilbert for very sparse")

    print("\n2. Expected Improvements:")
    print("   - 8K d=2: Should improve (was 2.63x slower)")
    print("   - 16K d=4: Should improve (was 4.82x slower)")
    print("   - Small sparse: Should use PyTorch path")


def main():
    benchmark_sparse_optimizations()
    verify_correctness()
    analyze_improvements()

    print("\n\n=== Summary ===")
    print("The sparse optimizations focus on:")
    print("1. Adaptive block sizing based on sparsity")
    print("2. Asymmetric blocks for 50% sparse patterns")
    print("3. PyTorch fallback for small very sparse")
    print("4. Disabling overhead features for sparse")


if __name__ == "__main__":
    main()
