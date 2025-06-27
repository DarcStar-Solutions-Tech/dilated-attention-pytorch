"""
Demonstrate key findings from benchmarking
"""

import time

import torch

from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention


def test_max_sequence_length():
    """Show that ImprovedDilatedAttention can handle 2x longer sequences"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Maximum Sequence Length Demonstration")
    print("=" * 60)

    # Test parameters
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Test sequence lengths
    test_lengths = [65536, 131072, 262144, 524288]

    for seq_len in test_lengths:
        print(f"\nTesting sequence length: {seq_len:,} tokens")

        # Adjust segments
        segments = [
            min(2048, seq_len // 4),
            min(4096, seq_len // 2),
            min(8192, seq_len),
        ]
        dilation_rates = [1, 2, 4]

        # Test DilatedAttention
        try:
            module = DilatedAttention(segments, dilation_rates, 0.0).to(device, dtype)
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                output = module(q, k, v)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            print(f"  DilatedAttention: ✓ Success ({elapsed:.1f}ms)")
            del module, q, k, v, output
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  DilatedAttention: ✗ Failed - {str(e)[:50]}...")

        # Test ImprovedDilatedAttention
        try:
            module = ImprovedDilatedAttention(segments, dilation_rates, 0.0).to(
                device, dtype
            )
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                output = module(q, k, v)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            # Get memory usage
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"  ImprovedDilatedAttention: ✓ Success ({elapsed:.1f}ms, {memory_gb:.2f}GB)"
            )
            del module, q, k, v, output
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ImprovedDilatedAttention: ✗ Failed - {str(e)[:50]}...")


def main():
    print("\n" + "=" * 60)
    print("KEY FINDINGS FROM BENCHMARKING")
    print("=" * 60)

    print("\n1. ImprovedDilatedAttention achieves 2x longer sequences than base")
    print("2. Block sparse implementations need further optimization")
    print("3. Memory efficiency: Improved > Ring > Base > Block Sparse")
    print("4. Speed ranking: Base > Improved > Ring > Block Sparse")

    print("\nDemonstrating maximum sequence lengths...\n")
    test_max_sequence_length()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("• For maximum sequence length (512K): Use ImprovedDilatedAttention")
    print("• For maximum speed (<256K): Use DilatedAttention")
    print("• For balanced performance: Use ImprovedDilatedAttention")
    print("• Avoid block sparse until optimized further")


if __name__ == "__main__":
    main()
