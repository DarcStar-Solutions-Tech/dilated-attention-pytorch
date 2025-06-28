"""
Analysis of why RingDilatedAttention with ring_size=1 is slower than base implementations
"""

import time

import torch

from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention


def analyze_performance_differences():
    print("Performance Analysis: Ring vs Base Dilated Attention")
    print("=" * 80)

    print("\nKEY DIFFERENCES FOUND:")

    print("\n1. TENSOR RESHAPING OVERHEAD:")
    print("   - Base DilatedAttention: Uses einops.rearrange (optimized)")
    print("   - Ring fallback: Manual view/reshape operations with contiguous() calls")
    print("   - Extra operations: Multiple .contiguous() calls force memory copies")

    print("\n2. SEGMENTATION APPROACH:")
    print("   - Base: Direct rearrange 'b (n s) h d -> b n s h d'")
    print("   - Ring: Custom _segment_tensor with padding logic")
    print("   - Ring adds: Padding calculations, trimming, extra views")

    print("\n3. DILATION HANDLING:")
    print("   - Base: Simple slicing q_seg[:, :, offset::r, hmin:hmax, :]")
    print("   - Ring: index_select operations with cached indices")
    print("   - Ring adds: Device checks, index caching overhead")

    print("\n4. ADDITIONAL OVERHEAD IN RING:")
    print("   - Repeat operations for mismatched segments (lines 523-526)")
    print("   - index_copy_ operations for reconstruction (line 573)")
    print("   - Extra device checks and transfers")
    print("   - Pre-compute patterns that aren't used with ring_size=1")

    print("\n5. MEMORY ACCESS PATTERNS:")
    print("   - Base: Direct operations on views")
    print("   - Ring: More intermediate tensors and copies")


def benchmark_operations():
    """Benchmark specific operations to show overhead"""
    print("\n\nBENCHMARKING SPECIFIC OPERATIONS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test tensor
    b, n, h, d = 2, 8192, 8, 64
    x = torch.randn(b, n, h, d, device=device, dtype=dtype)

    # Benchmark einops vs manual reshape
    print("\n1. Reshape operations (1000 iterations):")

    # Einops-style (what base uses)
    from einops import rearrange

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = rearrange(x, "b (n s) h d -> b n s h d", s=2048)
    torch.cuda.synchronize()
    einops_time = (time.time() - start) * 1000
    print(f"   einops.rearrange: {einops_time:.2f}ms")

    # Manual reshape with contiguous (what ring uses)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        x_cont = x.contiguous()
        _ = x_cont.view(b, n // 2048, 2048, h, d)
    torch.cuda.synchronize()
    manual_time = (time.time() - start) * 1000
    print(f"   contiguous + view: {manual_time:.2f}ms")
    print(f"   Overhead: {(manual_time / einops_time - 1) * 100:.0f}%")

    # Benchmark index operations
    print("\n2. Dilation operations (1000 iterations):")

    # Direct slicing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = x[:, 0::2, :, :]  # Simple stride
    torch.cuda.synchronize()
    slice_time = (time.time() - start) * 1000
    print(f"   Direct slicing: {slice_time:.2f}ms")

    # Index select (what ring uses)
    idx = torch.arange(0, n, 2, device=device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = x.index_select(1, idx)
    torch.cuda.synchronize()
    index_time = (time.time() - start) * 1000
    print(f"   index_select: {index_time:.2f}ms")
    print(f"   Overhead: {(index_time / slice_time - 1) * 100:.0f}%")


def compare_implementations():
    """Direct comparison of implementations"""
    print("\n\nDIRECT IMPLEMENTATION COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test parameters
    seq_lens = [2048, 8192, 32768]
    batch_size = 1
    num_heads = 8
    head_dim = 64

    for seq_len in seq_lens:
        print(f"\nSequence length: {seq_len}")

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

        # Test each implementation
        implementations = [
            ("DilatedAttention", DilatedAttention(segments, dilation_rates, 0.0)),
            (
                "ImprovedDilatedAttention",
                ImprovedDilatedAttention(segments, dilation_rates, 0.0),
            ),
            (
                "RingDilated(ring_size=1)",
                RingDilatedAttention(segments, dilation_rates, 0.0, ring_size=1),
            ),
        ]

        for name, module in implementations:
            module = module.to(device, dtype)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = module(q, k, v)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            for _ in range(10):
                with torch.no_grad():
                    _ = module(q, k, v)

            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 10 * 1000

            print(f"  {name:30} {elapsed:8.2f}ms")

            del module

        # Cleanup
        del q, k, v
        torch.cuda.empty_cache()


def main():
    analyze_performance_differences()
    benchmark_operations()
    compare_implementations()

    print("\n\nCONCLUSION:")
    print("=" * 80)
    print("RingDilatedAttention with ring_size=1 is slower because:")
    print("1. It uses less optimized tensor operations (manual reshape vs einops)")
    print(
        "2. Additional overhead from ring-specific logic not needed for single device"
    )
    print("3. Extra memory copies from .contiguous() calls")
    print("4. index_select is slower than direct slicing for dilation")
    print(
        "5. The implementation is optimized for distributed ring operation, not single device"
    )
    print("\nRecommendation: Use base DilatedAttention for single device scenarios")


if __name__ == "__main__":
    main()
