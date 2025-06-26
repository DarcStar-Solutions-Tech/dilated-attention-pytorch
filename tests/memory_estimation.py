#!/usr/bin/env python3
"""
Memory estimation tool for dilated attention implementations.
Estimates the maximum number of tokens that can be trained on 80GB VRAM.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a dilated attention model."""

    embed_dim: int
    num_heads: int
    num_layers: int
    vocab_size: int
    segment_lengths: list[int]
    dilation_rates: list[int]

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def max_segment_length(self) -> int:
        return max(self.segment_lengths)


@dataclass
class MemoryBreakdown:
    """Breakdown of memory usage components."""

    model_parameters: float  # GB
    activations: float  # GB
    gradients: float  # GB
    optimizer_states: float  # GB
    attention_matrices: float  # GB
    intermediate_tensors: float  # GB
    total: float  # GB


def calculate_model_parameters_memory(config: ModelConfig, dtype_bytes: int = 2) -> float:
    """Calculate memory for model parameters (weights and biases)."""

    # Embedding layers
    embedding_params = config.vocab_size * config.embed_dim

    # Each transformer layer has:
    # - Q, K, V projections: 3 * embed_dim^2
    # - Output projection: embed_dim^2
    # - Feed forward: 2 * embed_dim * (4 * embed_dim) = 8 * embed_dim^2
    # - Layer norms: 2 * embed_dim (negligible)
    params_per_layer = 3 * config.embed_dim**2 + config.embed_dim**2 + 8 * config.embed_dim**2
    params_per_layer = 12 * config.embed_dim**2

    total_transformer_params = config.num_layers * params_per_layer

    # Output head (if using language modeling)
    output_head_params = config.vocab_size * config.embed_dim

    total_params = embedding_params + total_transformer_params + output_head_params

    # Convert to GB
    return (total_params * dtype_bytes) / (1024**3)


def calculate_dilated_attention_memory(
    seq_len: int,
    config: ModelConfig,
    batch_size: int = 1,
    dtype_bytes: int = 2,
    implementation: str = "original",
) -> float:
    """Calculate memory for dilated attention computation."""

    embed_dim = config.embed_dim
    num_heads = config.num_heads
    head_dim = config.head_dim
    num_segments = len(config.segment_lengths)

    # Memory for Q, K, V tensors: (batch, seq_len, num_heads, head_dim)
    qkv_memory = 3 * batch_size * seq_len * num_heads * head_dim * dtype_bytes

    # Memory for attention computation per segment
    attention_memory = 0

    for i, (seg_len, dil_rate) in enumerate(
        zip(config.segment_lengths, config.dilation_rates, strict=False)
    ):
        if seq_len < seg_len:
            continue

        # Number of segments
        num_segs = seq_len // seg_len

        # Effective sequence length after dilation
        effective_seq_len = seg_len // dil_rate

        # Heads assigned to this segment
        heads_per_group = num_heads // num_segments
        if i < num_heads % num_segments:
            heads_per_group += 1

        if implementation == "original":
            # Original implementation memory
            # Attention matrix: (batch * num_segs, effective_seq_len, effective_seq_len)
            attn_matrix = batch_size * num_segs * effective_seq_len**2 * dtype_bytes

            # Intermediate tensors from rearrange operations
            intermediate = batch_size * seq_len * heads_per_group * head_dim * dtype_bytes * 3

            segment_memory = attn_matrix + intermediate

        else:  # improved implementation
            # Improved implementation is more memory efficient
            # Attention matrix: (batch * num_segs, effective_seq_len, effective_seq_len)
            attn_matrix = batch_size * num_segs * effective_seq_len**2 * dtype_bytes

            # Reduced intermediate tensors due to optimizations
            intermediate = batch_size * seq_len * heads_per_group * head_dim * dtype_bytes * 2

            segment_memory = attn_matrix + intermediate

        attention_memory += segment_memory

    total_memory = (qkv_memory + attention_memory) / (1024**3)  # Convert to GB
    return total_memory


def calculate_activation_memory(
    seq_len: int, config: ModelConfig, batch_size: int = 1, dtype_bytes: int = 2
) -> float:
    """Calculate memory for activations during forward pass."""

    embed_dim = config.embed_dim
    num_layers = config.num_layers

    # Activations stored for each layer (for backprop)
    # - Input to each layer: (batch, seq_len, embed_dim)
    # - Attention output: (batch, seq_len, embed_dim)
    # - FFN intermediate: (batch, seq_len, 4 * embed_dim)
    # - FFN output: (batch, seq_len, embed_dim)

    activations_per_layer = batch_size * seq_len * embed_dim * (1 + 1 + 4 + 1) * dtype_bytes
    total_activations = num_layers * activations_per_layer

    return total_activations / (1024**3)  # Convert to GB


def calculate_gradient_memory(
    seq_len: int, config: ModelConfig, batch_size: int = 1, dtype_bytes: int = 2
) -> float:
    """Calculate memory for gradients (same size as parameters)."""
    return calculate_model_parameters_memory(config, dtype_bytes)


def calculate_optimizer_memory(
    config: ModelConfig,
    optimizer_type: str = "adamw",
    dtype_bytes: int = 4,  # Optimizer states typically use fp32
) -> float:
    """Calculate memory for optimizer states."""

    param_memory = calculate_model_parameters_memory(config, dtype_bytes=4)  # fp32 for optimizer

    if optimizer_type.lower() == "adamw":
        # AdamW stores: momentum, variance, and original parameters
        return param_memory * 3
    elif optimizer_type.lower() == "sgd":
        # SGD with momentum stores: momentum
        return param_memory * 1
    else:
        # Conservative estimate
        return param_memory * 2


def estimate_total_memory(
    seq_len: int,
    config: ModelConfig,
    batch_size: int = 1,
    implementation: str = "original",
    optimizer_type: str = "adamw",
    dtype_bytes: int = 2,
) -> MemoryBreakdown:
    """Estimate total memory usage for training."""

    model_params = calculate_model_parameters_memory(config, dtype_bytes)
    activations = calculate_activation_memory(seq_len, config, batch_size, dtype_bytes)
    gradients = calculate_gradient_memory(seq_len, config, batch_size, dtype_bytes)
    optimizer_states = calculate_optimizer_memory(config, optimizer_type)
    attention_matrices = calculate_dilated_attention_memory(
        seq_len, config, batch_size, dtype_bytes, implementation
    )

    # Intermediate tensors (buffers, temporary allocations)
    intermediate_tensors = (model_params + activations) * 0.2  # 20% overhead

    total = (
        model_params
        + activations
        + gradients
        + optimizer_states
        + attention_matrices
        + intermediate_tensors
    )

    return MemoryBreakdown(
        model_parameters=model_params,
        activations=activations,
        gradients=gradients,
        optimizer_states=optimizer_states,
        attention_matrices=attention_matrices,
        intermediate_tensors=intermediate_tensors,
        total=total,
    )


def find_max_tokens(
    config: ModelConfig,
    max_memory_gb: float = 80,
    batch_size: int = 1,
    implementation: str = "original",
    optimizer_type: str = "adamw",
    dtype_bytes: int = 2,
) -> tuple[int, MemoryBreakdown]:
    """Binary search to find maximum sequence length for given memory budget."""

    # Start with a reasonable range
    min_seq_len = config.max_segment_length
    max_seq_len = 1000000  # 1M tokens

    best_seq_len = min_seq_len
    best_breakdown = estimate_total_memory(
        min_seq_len, config, batch_size, implementation, optimizer_type, dtype_bytes
    )

    # Binary search
    while min_seq_len <= max_seq_len:
        mid_seq_len = (min_seq_len + max_seq_len) // 2

        # Ensure sequence length is multiple of max segment length
        mid_seq_len = (mid_seq_len // config.max_segment_length) * config.max_segment_length

        mid_seq_len = max(mid_seq_len, config.max_segment_length)

        breakdown = estimate_total_memory(
            mid_seq_len, config, batch_size, implementation, optimizer_type, dtype_bytes
        )

        if breakdown.total <= max_memory_gb:
            best_seq_len = mid_seq_len
            best_breakdown = breakdown
            min_seq_len = mid_seq_len + config.max_segment_length
        else:
            max_seq_len = mid_seq_len - config.max_segment_length

    return best_seq_len, best_breakdown


def main():
    """Main function to estimate memory usage for different configurations."""

    print("Dilated Attention Memory Estimation for 80GB VRAM")
    print("=" * 60)

    # Define model configurations
    configs = {
        "Small (125M params)": ModelConfig(
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            vocab_size=50000,
            segment_lengths=[2048, 4096, 8192],
            dilation_rates=[1, 2, 4],
        ),
        "Medium (350M params)": ModelConfig(
            embed_dim=1024,
            num_heads=16,
            num_layers=24,
            vocab_size=50000,
            segment_lengths=[2048, 4096, 8192, 16384],
            dilation_rates=[1, 2, 4, 8],
        ),
        "Large (1.3B params)": ModelConfig(
            embed_dim=2048,
            num_heads=32,
            num_layers=24,
            vocab_size=50000,
            segment_lengths=[2048, 4096, 8192, 16384, 32768],
            dilation_rates=[1, 2, 4, 8, 16],
        ),
        "XL (2.7B params)": ModelConfig(
            embed_dim=2560,
            num_heads=32,
            num_layers=32,
            vocab_size=50000,
            segment_lengths=[2048, 4096, 8192, 16384, 32768],
            dilation_rates=[1, 2, 4, 8, 16],
        ),
    }

    implementations = ["original", "improved"]
    max_memory_gb = 80

    print(f"\nEstimating maximum sequence length for {max_memory_gb}GB VRAM:\n")

    results = {}

    for config_name, config in configs.items():
        print(f"\n{config_name}:")
        print(f"  Model: {config.embed_dim}d, {config.num_heads}h, {config.num_layers}L")
        print(f"  Segments: {config.segment_lengths}")
        print(f"  Dilations: {config.dilation_rates}")

        results[config_name] = {}

        for impl in implementations:
            max_seq_len, breakdown = find_max_tokens(
                config, max_memory_gb, batch_size=1, implementation=impl
            )

            results[config_name][impl] = {"max_tokens": max_seq_len, "breakdown": breakdown}

            print(f"\n  {impl.title()} Implementation:")
            print(f"    Max sequence length: {max_seq_len:,} tokens")
            print("    Memory breakdown:")
            print(f"      Model parameters: {breakdown.model_parameters:.1f} GB")
            print(f"      Activations: {breakdown.activations:.1f} GB")
            print(f"      Gradients: {breakdown.gradients:.1f} GB")
            print(f"      Optimizer states: {breakdown.optimizer_states:.1f} GB")
            print(f"      Attention matrices: {breakdown.attention_matrices:.1f} GB")
            print(f"      Intermediate tensors: {breakdown.intermediate_tensors:.1f} GB")
            print(f"      Total: {breakdown.total:.1f} GB")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    print(f"{'Model':<20} {'Original':<15} {'Improved':<15} {'Improvement':<12}")
    print("-" * 62)

    for config_name in configs:
        orig_tokens = results[config_name]["original"]["max_tokens"]
        imp_tokens = results[config_name]["improved"]["max_tokens"]
        improvement = (imp_tokens - orig_tokens) / orig_tokens * 100

        print(f"{config_name:<20} {orig_tokens:>10,} {imp_tokens:>10,} {improvement:>8.1f}%")

    # Detailed analysis
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    print("1. Memory Bottlenecks:")
    print("   - Optimizer states (AdamW) typically consume the most memory")
    print("   - Attention matrices scale quadratically with effective sequence length")
    print("   - Model parameters become dominant for larger models")

    print("\n2. ImprovedDilatedAttention Benefits:")
    print("   - More efficient attention computation reduces peak memory")
    print("   - Early exit for oversized segments saves memory")
    print("   - Better tensor operations reduce intermediate allocations")

    print("\n3. Scaling Recommendations:")
    print("   - Use gradient checkpointing to reduce activation memory")
    print("   - Consider mixed precision (fp16) for 2x memory reduction")
    print("   - Use gradient accumulation for larger effective batch sizes")
    print("   - Implement sequence parallelism for longer sequences")


if __name__ == "__main__":
    main()
