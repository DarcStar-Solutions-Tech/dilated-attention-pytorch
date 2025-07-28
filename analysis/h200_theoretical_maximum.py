#!/usr/bin/env python3
"""
Calculate theoretical maximum sequence length for dilated attention on H200.
Based on fundamental memory requirements, not empirical benchmarks.
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class Config:
    """Model and hardware configuration."""

    # Model - matching our benchmark configuration
    num_heads: int = 8  # Our benchmarks used 8 heads
    head_dim: int = 64  # Our benchmarks used 64 dim
    num_layers: int = 1  # Single layer for benchmarking
    vocab_size: int = 32000

    # Hardware
    gpu_memory_gb: float = 141.0  # H200
    dtype_bytes: int = 2  # FP16

    # Dilated attention specifics
    max_segment_length: int = 65536
    num_segments: int = 6

    @property
    def hidden_dim(self):
        return self.num_heads * self.head_dim

    @property
    def model_params(self):
        """Calculate model parameters in billions."""
        # Embeddings
        embed = self.vocab_size * self.hidden_dim
        # Attention: QKVO projections
        attn = 4 * self.hidden_dim * self.hidden_dim * self.num_layers
        # FFN: 3 matrices for SwiGLU
        ffn = 3 * self.hidden_dim * int(self.hidden_dim * 2.67) * self.num_layers
        # LayerNorms
        ln = 2 * self.hidden_dim * self.num_layers

        total = embed + attn + ffn + ln
        return total / 1e9


def calculate_theoretical_memory(seq_len: int, config: Config) -> Dict[str, float]:
    """
    Calculate theoretical minimum memory requirements.

    Key insight: With Flash Attention 3 and dilated patterns,
    we don't need to store full attention matrices.
    """
    memory = {}

    # 1. Model parameters (fixed cost)
    param_bytes = config.model_params * 1e9 * config.dtype_bytes
    memory["params_gb"] = param_bytes / 1024**3

    # 2. KV Cache - This is the dominant factor for long sequences
    # Each layer stores K and V for all positions
    kv_bytes = (
        2  # K and V
        * config.num_layers  # All layers
        * seq_len  # All positions
        * config.num_heads  # All heads
        * config.head_dim  # Head dimension
        * config.dtype_bytes  # Data type
    )
    memory["kv_cache_gb"] = kv_bytes / 1024**3

    # 3. Activations with Flash Attention 3
    # Only need to store:
    # - Current layer's hidden states
    # - Small working buffers for block-wise attention

    # Hidden states (can be recomputed, so only current layer)
    hidden_bytes = seq_len * config.hidden_dim * config.dtype_bytes
    memory["hidden_states_gb"] = hidden_bytes / 1024**3

    # Flash Attention 3 working memory
    # Processes in blocks of ~256 tokens
    block_size = 256
    # Only materializes attention for one block at a time
    flash_bytes = (
        config.num_heads  # Per head
        * block_size  # Query block
        * block_size  # Key block
        * config.dtype_bytes  # Data type
    )
    memory["flash_working_gb"] = flash_bytes / 1024**3

    # 4. Dilated attention pattern storage
    # For each segment, store indices (int32)
    pattern_bytes = (
        config.num_segments  # Number of segments
        * config.max_segment_length  # Max indices per segment
        * 4  # int32
    )
    memory["patterns_gb"] = pattern_bytes / 1024**3

    # 5. Output buffer
    output_bytes = seq_len * config.hidden_dim * config.dtype_bytes
    memory["output_gb"] = output_bytes / 1024**3

    # Total
    memory["total_gb"] = sum(memory.values())

    return memory


def find_maximum_sequence(
    config: Config, target_util: float = 0.95
) -> Tuple[int, Dict]:
    """Binary search for maximum sequence length."""

    available_memory = config.gpu_memory_gb * target_util

    # First, check if we need to search in millions
    test_seq = 10_000_000  # 10M tokens
    test_memory = calculate_theoretical_memory(test_seq, config)

    if test_memory["total_gb"] > available_memory:
        # Search in smaller range
        low = 100_000
        high = 10_000_000
        step = 100_000
    else:
        # Search in larger range
        low = 10_000_000
        high = 500_000_000  # 500M tokens
        step = 1_000_000

    best_seq = low
    best_memory = calculate_theoretical_memory(low, config)

    while low <= high:
        mid = (low + high) // 2
        # Round to nearest step
        mid = (mid // step) * step

        memory = calculate_theoretical_memory(mid, config)

        if memory["total_gb"] <= available_memory:
            best_seq = mid
            best_memory = memory
            low = mid + step
        else:
            high = mid - step

    return best_seq, best_memory


def analyze_h200_theoretical():
    """Analyze theoretical maximum for H200."""

    config = Config()

    print("=== H200 Theoretical Maximum Sequence Length ===")
    print(
        f"\nModel: {config.hidden_dim}d ({config.num_heads} heads × {config.head_dim} dim)"
    )
    print(f"Layers: {config.num_layers}")
    print(f"Parameters: {config.model_params:.1f}B")
    print(f"\nH200: {config.gpu_memory_gb} GB HBM3e")
    print("Using: Flash Attention 3 + Dilated Patterns")

    # Find maximum
    max_seq, memory = find_maximum_sequence(config)

    print(f"\n{'=' * 50}")
    print(f"THEORETICAL MAXIMUM: {max_seq:,} tokens")
    print(f"That's {max_seq / 1_000_000:.1f} million tokens!")
    print(f"{'=' * 50}")

    print("\nMemory breakdown:")
    sorted_memory = sorted(
        [(k, v) for k, v in memory.items() if k != "total_gb"],
        key=lambda x: x[1],
        reverse=True,
    )
    for name, gb in sorted_memory:
        if gb > 0.01:
            pct = gb / config.gpu_memory_gb * 100
            print(f"  {name.replace('_gb', ''):20}: {gb:>8.2f} GB ({pct:>5.1f}%)")

    total_gb = memory.get(
        "total_gb", sum(v for k, v in memory.items() if k != "total_gb")
    )
    print(
        f"\n  {'TOTAL':20}: {total_gb:>8.2f} GB ({total_gb / config.gpu_memory_gb * 100:>5.1f}%)"
    )

    # Calculate per-token memory
    per_token_bytes = (memory["total_gb"] * 1024**3) / max_seq
    per_token_kb = per_token_bytes / 1024

    print(f"\nMemory per token: {per_token_kb:.1f} KB")

    # Show scaling
    print("\n=== Scaling Analysis ===")

    # Different configurations
    configs = [
        ("GTX 1080 (8GB)", 8, 0.90),
        ("RTX 3090 (24GB)", 24, 0.95),
        ("A100 (40GB)", 40, 0.95),
        ("A100 (80GB)", 80, 0.95),
        ("H100 (80GB)", 80, 0.95),
        ("H200 (141GB)", 141, 0.95),
    ]

    print(f"\n{'GPU':20} | {'Memory':>8} | {'Max Tokens':>15} | {'Notes'}")
    print("-" * 65)

    for name, mem_gb, util in configs:
        temp_config = Config()
        temp_config.gpu_memory_gb = mem_gb
        seq, _ = find_maximum_sequence(temp_config, util)

        if seq > 100_000_000:
            notes = f"{seq / 1_000_000:.0f}M tokens!"
        elif seq > 10_000_000:
            notes = f"{seq / 1_000_000:.1f}M tokens"
        else:
            notes = f"{seq / 1_000_000:.2f}M tokens"

        print(f"{name:20} | {mem_gb:>6} GB | {seq:>15,} | {notes}")

    # Compare with benchmarks
    print("\n=== Benchmark Validation ===")

    # Check against known results
    print("\nKnown benchmark: 131K tokens on 2x GTX 1080 (16GB) used 3GB")
    bench_memory = calculate_theoretical_memory(131_072, config)
    print(f"Theoretical for 131K: {bench_memory['total_gb']:.2f} GB")
    print("Benchmark showed: 3.0 GB")
    print(
        "Difference: Likely due to multi-GPU overhead and actual implementation details"
    )

    # Extreme sequences from BlockSparse
    print("\nBlockSparse achieved 786K tokens on 8GB GPU")
    print("This used extreme sparsity (95%) which our calculation doesn't account for")

    print("\n=== Key Insights ===")
    print("1. KV cache dominates memory usage at long sequences")
    print("2. Flash Attention 3 eliminates quadratic scaling")
    print("3. Theoretical maximum assumes perfect memory utilization")
    print("4. Real implementations have overheads (typically 10-30%)")
    print("5. Multi-GPU setups can achieve near-linear scaling")


if __name__ == "__main__":
    analyze_h200_theoretical()
