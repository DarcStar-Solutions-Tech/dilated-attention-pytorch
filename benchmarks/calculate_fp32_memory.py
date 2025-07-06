#!/usr/bin/env python3
"""
Calculate expected memory usage for different sequence lengths with FP32.
"""


def calculate_memory(seq_len, batch_size=1, num_heads=8, head_dim=64):
    """Calculate memory usage for Q, K, V tensors."""
    # Elements per tensor
    elements = batch_size * seq_len * num_heads * head_dim

    # Bytes per element (4 for float32)
    bytes_per_element = 4

    # Memory for one tensor
    memory_per_tensor = elements * bytes_per_element

    # Total for Q, K, V
    total_memory = 3 * memory_per_tensor

    return {
        "elements_per_tensor": elements,
        "memory_per_tensor_mb": memory_per_tensor / (1024**2),
        "total_memory_mb": total_memory / (1024**2),
        "total_memory_gb": total_memory / (1024**3),
    }


def main():
    print("Memory Usage Calculation for FP32 (4 bytes per element)")
    print("=" * 60)
    print("Configuration: batch_size=1, num_heads=8, head_dim=64")
    print("=" * 60)

    seq_lengths = [16384, 32768, 65536, 98304, 131072, 262144, 524288, 1048576]

    print(
        f"{'Seq Length':>12} | {'Per Tensor':>12} | {'Total (Q,K,V)':>12} | {'Total GB':>10}"
    )
    print("-" * 60)

    for seq_len in seq_lengths:
        info = calculate_memory(seq_len)
        print(
            f"{seq_len:>12,} | {info['memory_per_tensor_mb']:>10.1f} MB | "
            f"{info['total_memory_mb']:>10.1f} MB | {info['total_memory_gb']:>8.2f} GB"
        )

    print("\nNotes:")
    print("- This is just for Q, K, V tensors")
    print("- Actual memory usage includes:")
    print("  - Attention scores (Q @ K^T)")
    print("  - Intermediate activations")
    print("  - Pattern buffers for dilated attention")
    print("  - Communication buffers for ring attention")
    print("\nWith extreme dilation (8,16), effective memory is reduced significantly")
    print("due to sparse attention patterns.")


if __name__ == "__main__":
    main()
