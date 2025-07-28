"""
Test the optimization of RingDilatedAttention
"""

import time
import torch

from dilated_attention_pytorch import DilatedAttention
from dilated_attention_pytorch.ring import HilbertRingAttention, RingAttentionConfig


def test_optimization():
    print("Testing HilbertRingAttention Optimization")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test parameters
    batch_size = 1
    num_heads = 8
    head_dim = 64

    test_configs = [
        (2048, "Short sequence"),
        (8192, "Medium sequence"),
        (32768, "Long sequence"),
    ]

    for seq_len, desc in test_configs:
        print(f"\n{desc} ({seq_len:,} tokens):")

        segments = [
            min(1024, seq_len // 4),
            min(2048, seq_len // 2),
            min(4096, seq_len),
        ]
        dilation_rates = [1, 2, 4]

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Standard DilatedAttention
        standard = DilatedAttention(
            segment_lengths=segments,
            dilation_rates=dilation_rates,
            dropout=0.0,
        ).to(device)

        # Ring Dilated Attention with Hilbert
        config = RingAttentionConfig(
            segment_lengths=segments,
            dilation_rates=dilation_rates,
            dropout=0.0,
            use_hilbert=True,
            hilbert_curve_level=8,
        )
        ring = HilbertRingAttention(config, device=device, dtype=dtype)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = standard(q, k, v)
                _ = ring(q, k, v)

        # Sync
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timing for standard
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                out_standard = standard(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_standard = (time.perf_counter() - start) / 10

        # Timing for ring
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                out_ring = ring(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_ring = (time.perf_counter() - start) / 10

        # Check outputs are close
        diff = (out_standard - out_ring).abs().mean().item()

        print(f"  Standard time: {time_standard * 1000:.2f} ms")
        print(f"  Ring time: {time_ring * 1000:.2f} ms")
        print(f"  Speedup: {time_standard / time_ring:.2f}x")
        print(f"  Output difference: {diff:.6f}")

        # For single GPU, ring should be slightly slower due to overhead
        # But it enables much longer sequences with multi-GPU
        assert diff < 0.01, f"Output difference too large: {diff}"


if __name__ == "__main__":
    test_optimization()
