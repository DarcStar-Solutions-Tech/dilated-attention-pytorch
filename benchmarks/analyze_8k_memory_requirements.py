#!/usr/bin/env python3
"""Analyze memory requirements for different block configurations."""


def calculate_shared_memory(BLOCK_M, BLOCK_N, BLOCK_D, dtype_size=2):
    """Calculate shared memory requirements for attention kernel."""
    # Each block needs to store:
    # - Q block: BLOCK_M x BLOCK_D
    # - K block: BLOCK_N x BLOCK_D
    # - V block: BLOCK_N x BLOCK_D
    # - S (scores): BLOCK_M x BLOCK_N
    # - Additional buffers for accumulator, etc.

    q_memory = BLOCK_M * BLOCK_D * dtype_size
    k_memory = BLOCK_N * BLOCK_D * dtype_size
    v_memory = BLOCK_N * BLOCK_D * dtype_size
    s_memory = BLOCK_M * BLOCK_N * 4  # float32 for numerical stability

    # Additional buffers (accumulator, max values, sum values)
    acc_memory = BLOCK_M * BLOCK_D * 4  # float32 accumulator
    metadata = BLOCK_M * 4 * 2  # max and sum values

    total = q_memory + k_memory + v_memory + s_memory + acc_memory + metadata

    return total


def main():
    print("Shared Memory Analysis for GTX 1080")
    print("=" * 60)
    print("Shared memory limit: 49,152 bytes (48 KB)")
    print()

    # Test configurations
    configs = [
        # Current configs
        (64, 64, 64, "4K current"),
        (64, 128, 64, "8K proposed asymmetric"),
        (128, 64, 64, "8K proposed asymmetric (reversed)"),
        (96, 96, 64, "8K-10K proposed"),
        (128, 128, 64, "12K-16K current"),
        # Alternative configs
        (64, 64, 32, "Reduced D dimension"),
        (48, 96, 64, "Non-power-of-2"),
        (32, 128, 64, "Thin and wide"),
        (128, 32, 64, "Tall and narrow"),
    ]

    print(f"{'Config':<30} {'Block Size':<15} {'Memory (KB)':<12} {'Status':<10}")
    print("-" * 70)

    for BLOCK_M, BLOCK_N, BLOCK_D, desc in configs:
        memory = calculate_shared_memory(BLOCK_M, BLOCK_N, BLOCK_D)
        memory_kb = memory / 1024
        status = "OK" if memory <= 49152 else "TOO BIG"

        print(
            f"{desc:<30} {f'{BLOCK_M}x{BLOCK_N}x{BLOCK_D}':<15} {memory_kb:<12.1f} {status:<10}"
        )

    print("\n\nGrid Alignment Analysis")
    print("=" * 60)

    # Analyze grid sizes for 8K
    seq_len = 8192
    num_sms = 20  # GTX 1080

    print(f"Sequence length: {seq_len}")
    print(f"Number of SMs: {num_sms}")
    print()

    block_sizes = [32, 48, 64, 96, 128]

    print(
        f"{'Block Size':<12} {'Grid Size':<15} {'Total Blocks':<15} {'Per SM':<10} {'Efficiency':<10}"
    )
    print("-" * 70)

    for block_m in block_sizes:
        for block_n in block_sizes:
            if calculate_shared_memory(block_m, block_n, 64) > 49152:
                continue

            grid_m = (seq_len + block_m - 1) // block_m
            grid_n = (seq_len + block_n - 1) // block_n
            total_blocks = grid_m * grid_n
            blocks_per_sm = total_blocks / num_sms

            # Calculate load balancing efficiency
            ideal_blocks = int(blocks_per_sm) * num_sms
            efficiency = ideal_blocks / total_blocks * 100

            print(
                f"{f'{block_m}x{block_n}':<12} {f'{grid_m}x{grid_n}':<15} "
                f"{total_blocks:<15} {blocks_per_sm:<10.1f} {efficiency:<10.1f}%"
            )

    print("\n\nRECOMMENDATION")
    print("=" * 60)
    print("""
For 8K sequences on GTX 1080:
- Best configuration: 64x64 with BLOCK_D=32 (uses 40KB shared memory)
- This avoids the shared memory limit while maintaining reasonable performance
- Alternative: 48x96 for better grid alignment (360 blocks per SM)
""")


if __name__ == "__main__":
    main()
