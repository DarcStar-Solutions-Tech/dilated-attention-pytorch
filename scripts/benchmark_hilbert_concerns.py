#!/usr/bin/env python3
"""
Benchmark to address specific concerns about Hilbert attention implementation.

Tests:
1. Multi-GPU availability and ring attention
2. Float32 consistency (Pascal architecture)
3. SDPA vs manual attention with dilation masks
4. Actual implementation testing
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from typing import Dict, Tuple
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import implementations
try:
    from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core import (
        RingDilatedAttentionHilbertCore,
    )

    HILBERT_CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RingDilatedAttentionHilbertCore: {e}")
    HILBERT_CORE_AVAILABLE = False

try:
    from src.dilated_attention_pytorch.dilated_attention import DilatedAttention

    DILATED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DilatedAttention: {e}")
    DILATED_AVAILABLE = False


def check_multi_gpu() -> Tuple[bool, int]:
    """Check if multiple GPUs are available."""
    if not torch.cuda.is_available():
        return False, 0

    num_gpus = torch.cuda.device_count()
    return num_gpus > 1, num_gpus


def create_dilation_mask(
    seq_len: int,
    segment_length: int,
    dilation_rate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create attention mask for dilated attention pattern."""
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    # Number of segments
    num_segments = seq_len // segment_length

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_length
        seg_end = seg_start + segment_length

        # Within each segment, create dilated pattern
        for i in range(seg_start, seg_end):
            for j in range(seg_start, seg_end):
                if (i - seg_start) % dilation_rate == 0 and (
                    j - seg_start
                ) % dilation_rate == 0:
                    mask[i, j] = 1.0

    # Convert to attention mask (0 = attend, -inf = mask)
    attention_mask = torch.where(mask == 1, 0.0, float("-inf"))
    return attention_mask


def manual_dilated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_length: int,
    dilation_rate: int,
) -> torch.Tensor:
    """Manual implementation of dilated attention using matmul."""
    batch_size, seq_len, num_heads, head_dim = query.shape

    # Reshape for attention
    q = query.transpose(1, 2)  # [batch, heads, seq, dim]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

    # Apply dilation mask
    mask = create_dilation_mask(
        seq_len, segment_length, dilation_rate, query.device, query.dtype
    )
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
    scores = scores + mask

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, v)

    # Reshape back
    output = output.transpose(1, 2).contiguous()
    return output


def sdpa_dilated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_length: int,
    dilation_rate: int,
) -> torch.Tensor:
    """SDPA implementation with dilation mask."""
    batch_size, seq_len, num_heads, head_dim = query.shape

    # Create attention mask
    mask = create_dilation_mask(
        seq_len, segment_length, dilation_rate, query.device, query.dtype
    )

    # SDPA expects [batch, heads, seq, seq] mask
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

    # Use scaled_dot_product_attention
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),  # [batch, heads, seq, dim]
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )

    return output.transpose(1, 2).contiguous()


def benchmark_implementation(
    name: str,
    attention_fn,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_length: int,
    dilation_rate: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark a single attention implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Force float32 for Pascal compatibility

    # Create input tensors
    query = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Warmup
    for _ in range(num_warmup):
        try:
            if "segment_length" in attention_fn.__code__.co_varnames:
                output = attention_fn(query, key, value, segment_length, dilation_rate)
            else:
                output = attention_fn(query, key, value)
            if device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            return {
                "error": str(e),
                "forward_time": float("inf"),
                "backward_time": float("inf"),
            }

    # Time forward pass
    if device.type == "cuda":
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []

    for _ in range(num_runs):
        # Forward
        start = time.perf_counter()
        try:
            if "segment_length" in attention_fn.__code__.co_varnames:
                output = attention_fn(query, key, value, segment_length, dilation_rate)
            else:
                output = attention_fn(query, key, value)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - start
            forward_times.append(forward_time)

            # Backward
            grad_output = torch.randn_like(output)
            start = time.perf_counter()
            output.backward(grad_output, retain_graph=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            backward_time = time.perf_counter() - start
            backward_times.append(backward_time)

        except Exception as e:
            return {
                "error": str(e),
                "forward_time": float("inf"),
                "backward_time": float("inf"),
            }

    return {
        "forward_time": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "backward_time": np.mean(backward_times),
        "backward_std": np.std(backward_times),
        "total_time": np.mean(forward_times) + np.mean(backward_times),
    }


def test_multi_gpu_ring_attention():
    """Test ring attention with multiple GPUs if available."""
    multi_gpu, num_gpus = check_multi_gpu()

    if not multi_gpu:
        return f"Multi-GPU not available (found {num_gpus} GPU(s))"

    if not HILBERT_CORE_AVAILABLE:
        return "RingDilatedAttentionHilbertCore not available"

    # Test configurations
    batch_size = 2
    seq_len = 8192
    embed_dim = 768
    num_heads = 12
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    results = []

    for ring_size in [2, min(4, num_gpus)]:
        try:
            model = RingDilatedAttentionHilbertCore(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.0,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=ring_size,
                use_fa3=False,  # Disable FA3 for compatibility
            ).cuda()

            # Force float32
            model = model.float()

            # Create input
            x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32).cuda()

            # Time execution
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            results.append(f"Ring size {ring_size}: {elapsed:.4f}s")

        except Exception as e:
            results.append(f"Ring size {ring_size}: Error - {str(e)}")

    return "\n".join(results)


def run_comprehensive_benchmark():
    """Run comprehensive benchmark addressing all concerns."""
    print("=" * 80)
    print("Hilbert Attention Implementation Benchmark")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    # System info
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        multi_gpu, num_gpus = check_multi_gpu()
        print(f"Number of GPUs: {num_gpus}")
        print(f"Multi-GPU available: {multi_gpu}")

    # Test configurations
    configs = [
        {
            "batch_size": 2,
            "seq_len": 2048,
            "num_heads": 8,
            "head_dim": 64,
            "segment_length": 512,
            "dilation_rate": 1,
        },
        {
            "batch_size": 2,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "segment_length": 1024,
            "dilation_rate": 2,
        },
        {
            "batch_size": 1,
            "seq_len": 8192,
            "num_heads": 12,
            "head_dim": 64,
            "segment_length": 2048,
            "dilation_rate": 4,
        },
    ]

    # Test 1: Compare manual vs SDPA implementation
    print("\n" + "=" * 60)
    print("Test 1: Manual Attention vs SDPA with Dilation Mask")
    print("=" * 60)

    for i, config in enumerate(configs):
        print(f"\nConfiguration {i + 1}:")
        print(f"  Sequence length: {config['seq_len']}")
        print(f"  Segment length: {config['segment_length']}")
        print(f"  Dilation rate: {config['dilation_rate']}")

        # Manual implementation
        manual_results = benchmark_implementation(
            "Manual", manual_dilated_attention, **config
        )

        # SDPA implementation
        sdpa_results = benchmark_implementation(
            "SDPA", sdpa_dilated_attention, **config
        )

        print("\n  Manual Implementation:")
        if "error" in manual_results:
            print(f"    Error: {manual_results['error']}")
        else:
            print(
                f"    Forward: {manual_results['forward_time'] * 1000:.2f}ms ± {manual_results['forward_std'] * 1000:.2f}ms"
            )
            print(
                f"    Backward: {manual_results['backward_time'] * 1000:.2f}ms ± {manual_results['backward_std'] * 1000:.2f}ms"
            )

        print("\n  SDPA Implementation:")
        if "error" in sdpa_results:
            print(f"    Error: {sdpa_results['error']}")
        else:
            print(
                f"    Forward: {sdpa_results['forward_time'] * 1000:.2f}ms ± {sdpa_results['forward_std'] * 1000:.2f}ms"
            )
            print(
                f"    Backward: {sdpa_results['backward_time'] * 1000:.2f}ms ± {sdpa_results['backward_std'] * 1000:.2f}ms"
            )

        if "error" not in manual_results and "error" not in sdpa_results:
            speedup = manual_results["total_time"] / sdpa_results["total_time"]
            print(f"\n  SDPA Speedup: {speedup:.2f}x")

    # Test 2: Multi-GPU Ring Attention
    print("\n" + "=" * 60)
    print("Test 2: Multi-GPU Ring Attention")
    print("=" * 60)

    multi_gpu_results = test_multi_gpu_ring_attention()
    print(multi_gpu_results)

    # Test 3: Actual Implementation Testing
    print("\n" + "=" * 60)
    print("Test 3: Actual Implementation Comparison")
    print("=" * 60)

    if HILBERT_CORE_AVAILABLE and DILATED_AVAILABLE:
        batch_size = 2
        seq_len = 4096
        embed_dim = 768
        num_heads = 12
        segment_lengths = [1024, 2048]
        dilation_rates = [1, 2]

        # Test Hilbert Core
        try:
            hilbert_model = RingDilatedAttentionHilbertCore(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.0,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_fa3=False,
            ).float()  # Force float32

            if torch.cuda.is_available():
                hilbert_model = hilbert_model.cuda()

            x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
            if torch.cuda.is_available():
                x = x.cuda()

            # Benchmark
            start = time.perf_counter()
            for _ in range(5):
                output = hilbert_model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            hilbert_time = (time.perf_counter() - start) / 5

            print("\nHilbert Core Implementation:")
            print(f"  Time per forward pass: {hilbert_time * 1000:.2f}ms")
            print(f"  Output shape: {output.shape}")

        except Exception as e:
            print(f"\nHilbert Core Implementation Error: {e}")

        # Test standard dilated attention
        try:
            standard_model = DilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                causal=False,
                use_xpos=False,
                use_rel_pos_bias=False,
            ).float()  # Force float32

            if torch.cuda.is_available():
                standard_model = standard_model.cuda()

            # Create input in right format
            x_dilated = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                embed_dim // num_heads,
                dtype=torch.float32,
            )
            if torch.cuda.is_available():
                x_dilated = x_dilated.cuda()

            # Benchmark
            start = time.perf_counter()
            for _ in range(5):
                output = standard_model(x_dilated, x_dilated, x_dilated)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            standard_time = (time.perf_counter() - start) / 5

            print("\nStandard Dilated Attention:")
            print(f"  Time per forward pass: {standard_time * 1000:.2f}ms")
            print(f"  Output shape: {output.shape}")

            if hilbert_time > 0 and standard_time > 0:
                print(f"\nSpeedup: {standard_time / hilbert_time:.2f}x")

        except Exception as e:
            print(f"\nStandard Dilated Attention Error: {e}")
    else:
        print("\nSkipping actual implementation tests - modules not available")

    # Test 4: Memory usage comparison
    print("\n" + "=" * 60)
    print("Test 4: Memory Usage Analysis")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Test memory with different approaches
        batch_size = 1
        seq_len = 16384
        num_heads = 12
        head_dim = 64

        # Manual attention memory
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()

        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        try:
            output = manual_dilated_attention(query, key, value, 4096, 2)
            torch.cuda.synchronize()
            manual_peak = torch.cuda.max_memory_allocated() - start_mem
            print(f"\nManual Attention Peak Memory: {manual_peak / 1024**2:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            print("\nManual Attention: Out of Memory")

        # SDPA memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        try:
            output = sdpa_dilated_attention(query, key, value, 4096, 2)
            torch.cuda.synchronize()
            sdpa_peak = torch.cuda.max_memory_allocated() - start_mem
            print(f"SDPA Attention Peak Memory: {sdpa_peak / 1024**2:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            print("SDPA Attention: Out of Memory")

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
