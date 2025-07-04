#!/usr/bin/env python3
"""
Calculate theoretical maximum sequence length for dilated attention on H200 GPU.
H200 specs: 141GB HBM3e memory, full Flash Attention 3 support.
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Standard model configuration."""

    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32  # Typical for 7B model
    vocab_size: int = 32000
    ffn_mult: float = 2.67  # SwiGLU uses 8/3 ≈ 2.67

    @property
    def hidden_dim(self):
        return self.num_heads * self.head_dim

    @property
    def ffn_dim(self):
        return int(self.hidden_dim * self.ffn_mult)


def calculate_memory_requirements(
    seq_len: int,
    config: ModelConfig,
    batch_size: int = 1,
    use_flash3: bool = True,
    dtype_bytes: int = 2,  # FP16/BF16
) -> Dict[str, float]:
    """Calculate memory requirements for different components."""

    h = config.num_heads
    d = config.head_dim
    hidden = config.hidden_dim
    ffn = config.ffn_dim
    layers = config.num_layers

    memory = {}

    # 1. Model Parameters (constant, not dependent on sequence length)
    # Per layer: Q,K,V,O projections + FFN
    qkvo_params = 4 * hidden * hidden * dtype_bytes
    ffn_params = 3 * hidden * ffn * dtype_bytes  # up, gate, down for SwiGLU
    layer_norm_params = 2 * hidden * dtype_bytes * 2  # 2 layer norms

    params_per_layer = qkvo_params + ffn_params + layer_norm_params
    total_params = layers * params_per_layer + vocab_size * hidden * dtype_bytes
    memory["model_params_gb"] = total_params / 1024**3

    # 2. Activations and KV Cache
    if use_flash3:
        # Flash Attention 3 - O(n) memory complexity
        # No materialized attention matrix!

        # Per layer activations (recomputed, not stored)
        # Only need current layer's activations in memory
        hidden_states = batch_size * seq_len * hidden * dtype_bytes

        # Flash3 internal buffers (very small, O(sqrt(n)))
        # Typical block size is 64-128
        block_size = 128
        flash_buffer = batch_size * h * block_size * d * dtype_bytes * 3  # Q,K,V blocks

        # KV cache for all layers (if using KV cache for inference)
        # This is the main memory consumer for long sequences
        kv_cache = 2 * layers * batch_size * seq_len * h * d * dtype_bytes

        # FFN activations (can be recomputed)
        ffn_acts = batch_size * seq_len * ffn * dtype_bytes

        activations_per_layer = hidden_states + flash_buffer + ffn_acts
        memory["activations_gb"] = activations_per_layer / 1024**3
        memory["kv_cache_gb"] = kv_cache / 1024**3
        memory["flash_buffer_gb"] = flash_buffer * layers / 1024**3

    else:
        # Standard attention - O(n²) memory
        # Full attention matrix must be materialized
        attention_matrix = batch_size * h * seq_len * seq_len * dtype_bytes
        hidden_states = batch_size * seq_len * hidden * dtype_bytes
        kv_cache = 2 * layers * batch_size * seq_len * h * d * dtype_bytes
        ffn_acts = batch_size * seq_len * ffn * dtype_bytes

        # Per layer: attention matrix is the killer
        activations_per_layer = attention_matrix + hidden_states + ffn_acts
        memory["activations_gb"] = activations_per_layer * layers / 1024**3
        memory["kv_cache_gb"] = kv_cache / 1024**3
        memory["attention_matrix_gb"] = attention_matrix * layers / 1024**3

    # 3. Gradients (for training)
    # Roughly same size as activations + parameters
    if batch_size > 1 or seq_len > 100000:  # Assume training for large sequences
        memory["gradients_gb"] = memory["activations_gb"] + memory["model_params_gb"]
    else:
        memory["gradients_gb"] = 0  # Inference only

    # 4. Optimizer states (Adam needs 2x parameters)
    memory["optimizer_gb"] = (
        2 * memory["model_params_gb"] if memory["gradients_gb"] > 0 else 0
    )

    # 5. Temporary buffers and overhead (20% safety margin)
    subtotal = sum(memory.values())
    memory["overhead_gb"] = 0.2 * subtotal

    memory["total_gb"] = sum(memory.values())

    return memory


def find_max_sequence_length(
    gpu_memory_gb: float,
    config: ModelConfig,
    batch_size: int = 1,
    use_flash3: bool = True,
    dtype_bytes: int = 2,
    for_training: bool = False,
) -> Tuple[int, Dict[str, float]]:
    """Binary search to find maximum sequence length that fits in memory."""

    # Start with reasonable bounds
    if use_flash3:
        low, high = 1000, 50_000_000  # Flash3 can go very high
    else:
        low, high = 1000, 1_000_000  # Standard attention limited

    best_seq_len = low
    best_memory = {}

    # Account for training vs inference
    if for_training:
        # Training needs gradients and optimizer states
        # Effectively reduces available memory by ~3x
        effective_memory = gpu_memory_gb * 0.9  # 90% utilization
    else:
        effective_memory = gpu_memory_gb * 0.95  # 95% utilization

    while low <= high:
        mid = (low + high) // 2

        # Round to nearest 1K for cleaner numbers
        mid = (mid // 1000) * 1000

        memory = calculate_memory_requirements(
            mid, config, batch_size, use_flash3, dtype_bytes
        )

        if memory["total_gb"] <= effective_memory:
            best_seq_len = mid
            best_memory = memory
            low = mid + 1000
        else:
            high = mid - 1000

    return best_seq_len, best_memory


def analyze_h200_capabilities():
    """Analyze H200 capabilities with different configurations."""

    h200_memory = 141  # GB
    config = ModelConfig()

    print("=== H200 Dilated Attention Maximum Sequence Length Analysis ===")
    print("\nGPU: NVIDIA H200")
    print(f"Memory: {h200_memory} GB HBM3e")
    print("Architecture: Hopper (Full Flash Attention 3 support)")
    print(
        f"\nModel: {config.hidden_dim}d, {config.num_heads} heads, "
        f"{config.num_layers} layers\n"
    )

    # Different scenarios
    scenarios = [
        # (name, batch_size, use_flash3, dtype_bytes, for_training)
        ("Inference FP16 + Flash3", 1, True, 2, False),
        ("Inference FP32 + Flash3", 1, True, 4, False),
        ("Inference FP16 No Flash", 1, False, 2, False),
        ("Training FP16 + Flash3", 1, True, 2, True),
        ("Training FP16 + Flash3 BS=8", 8, True, 2, True),
        ("Inference Int8 + Flash3", 1, True, 1, False),
    ]

    results = []

    for name, bs, flash, dtype, training in scenarios:
        max_len, memory = find_max_sequence_length(
            h200_memory, config, bs, flash, dtype, training
        )
        results.append((name, max_len, memory))

    # Print results
    print(f"{'Scenario':35} | {'Max Seq Length':>15} | {'Memory Breakdown'}")
    print("-" * 100)

    for name, max_len, memory in results:
        print(f"{name:35} | {max_len:>15,} | ", end="")

        # Show key memory components
        if "attention_matrix_gb" in memory:
            print(f"Attn Matrix: {memory.get('attention_matrix_gb', 0):.1f}GB", end="")
        else:
            print(f"KV Cache: {memory.get('kv_cache_gb', 0):.1f}GB", end="")

        print(f", Acts: {memory.get('activations_gb', 0):.1f}GB", end="")

        if memory.get("gradients_gb", 0) > 0:
            print(f", Grads: {memory.get('gradients_gb', 0):.1f}GB", end="")

        print(f", Total: {memory['total_gb']:.1f}GB")

    # Detailed breakdown for best case
    print("\n=== Detailed Breakdown for Best Case (Inference FP16 + Flash3) ===")
    best_memory = results[0][2]
    max_seq = results[0][1]

    print(f"\nMaximum sequence length: {max_seq:,} tokens")
    print(f"That's {max_seq / 1_000_000:.1f} million tokens!\n")

    print("Memory usage breakdown:")
    for key, value in best_memory.items():
        if value > 0.01:  # Only show significant components
            print(
                f"  {key.replace('_gb', ''):20}: {value:>8.2f} GB ({value / h200_memory * 100:>5.1f}%)"
            )

    # Compare with dilated attention specifics
    print("\n=== Dilated Attention Specific Advantages ===")
    print("\n1. With Flash Attention 3:")
    print("   - O(n) memory complexity instead of O(n²)")
    print("   - 1.5-2x faster than Flash Attention 2")
    print("   - Up to 75% GPU utilization on H200")

    print("\n2. Dilated Attention patterns:")
    segment_lengths = [2048, 4096, 8192, 16384, 32768, 65536]
    print(f"   - Segment lengths: {segment_lengths}")
    print("   - Process segments independently → better memory locality")
    print("   - Can checkpoint between segments for even longer sequences")

    print("\n3. Additional optimizations possible:")
    print("   - Gradient checkpointing: 2-3x longer sequences")
    print("   - Mixed precision: Critical ops in FP16, rest in FP8")
    print("   - Streaming mode: Virtually unlimited with bounded memory")

    # Practical recommendations
    print("\n=== Practical Recommendations ===")
    print(f"\n1. For inference: {results[0][1]:,} tokens with Flash3")
    print(f"2. For training: {results[3][1]:,} tokens with Flash3")
    print(f"3. With Int8: {results[5][1]:,} tokens possible")
    print("\n4. Extreme sequences (10M+): Use streaming dilated attention")
    print("5. Multi-GPU: Head-parallel can scale this linearly")


if __name__ == "__main__":
    analyze_h200_capabilities()
