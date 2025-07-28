"""Example demonstrating the new standardized ring attention API.

This example shows how to use the new base classes and configuration
system for ring attention implementations.
"""

import torch
import torch.distributed as dist

from dilated_attention_pytorch.ring.base import (
    StandardRingAttention,
    RingAttentionConfig,
    get_preset_config,
    create_ring_config,
)


def setup_distributed():
    """Initialize distributed environment if available."""
    if dist.is_available() and not dist.is_initialized():
        # In real usage, these would come from environment
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        # Initialize with default or environment settings
        import os

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if world_size > 1:
            dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
            print(f"Initialized distributed: rank {rank}/{world_size}")


def example_basic_usage():
    """Basic usage of standardized ring attention."""
    print("\n=== Basic Ring Attention Usage ===")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create configuration
    config = RingAttentionConfig(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        dropout=0.1,
        use_memory_pool=True,
        enable_error_recovery=True,
    )

    # Create ring attention
    attention = StandardRingAttention(config, device=device)
    print(f"Created ring attention: {attention}")

    # Create inputs
    batch_size = 2
    seq_len = 4096  # Must be divisible by largest segment length
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Forward pass
    output = attention(q, k, v, is_causal=True)
    print(f"Output shape: {output.shape}")

    # Check communication stats if distributed
    if attention.is_distributed:
        stats = attention.get_communication_stats()
        print(f"Communication stats: {stats}")


def example_preset_configs():
    """Using preset configurations."""
    print("\n=== Preset Configurations ===")

    # Development preset - includes debugging features
    dev_config = get_preset_config("development")
    print(f"Development config: {dev_config}")

    # Production preset - optimized for performance
    prod_config = get_preset_config("production")
    print(f"Production config: {prod_config}")

    # Large scale preset - for billion token sequences
    large_config = get_preset_config("large_scale")
    print(f"Large scale config: {large_config}")

    # Hilbert optimized preset
    hilbert_config = get_preset_config("hilbert_optimized")
    print(f"Hilbert config: {hilbert_config}")


def example_custom_config():
    """Creating custom configurations."""
    print("\n=== Custom Configuration ===")

    # Use convenience function
    config = create_ring_config(
        segment_lengths=[1024, 2048, 4096, 8192],
        dilation_rates=[1, 2, 4, 8],
        # Custom settings
        use_hilbert=True,
        hilbert_curve_level=10,
        overlap_communication=True,
        compile_mode="max-autotune",
        checkpoint_frequency=50,
        enable_watchdog=True,
    )

    print(f"Custom config: {config}")

    # Convert to dict for serialization
    config_dict = config.to_dict()
    print(f"Config keys: {list(config_dict.keys())}")

    # Recreate from dict
    config2 = RingAttentionConfig.from_dict(config_dict)
    assert config.segment_lengths == config2.segment_lengths


def example_distributed_training():
    """Example of using ring attention in distributed training."""
    print("\n=== Distributed Training Example ===")

    # Initialize distributed
    setup_distributed()

    # Get rank and device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    print(f"Running on rank {rank}/{world_size}, device: {device}")

    # Create config with distributed settings
    config = RingAttentionConfig(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        ring_size=world_size,  # Use all available GPUs
        communication_backend="nccl" if torch.cuda.is_available() else "gloo",
        enable_profiling=True,
        log_communication_stats=True,
    )

    # Create model with ring attention
    attention = StandardRingAttention(config, device=device)

    # Training loop simulation
    for step in range(3):
        # Create batch (different data per rank)
        batch_size = 2
        seq_len = 4096
        num_heads = 8
        head_dim = 64

        # Simulate different data per rank
        torch.manual_seed(rank + step)
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )

        # Forward pass
        output = attention(q, k, v, is_causal=True)

        # Simulate loss and backward
        loss = output.mean()
        loss.backward()

        # Log stats
        if step == 0 and rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            if config.log_communication_stats:
                stats = attention.get_communication_stats()
                print(f"Communication stats: {stats}")

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


def example_advanced_features():
    """Demonstrate advanced features of ring attention."""
    print("\n=== Advanced Features ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Hilbert curve optimization
    print("\n1. Hilbert Curve Optimization:")
    hilbert_config = RingAttentionConfig(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        use_hilbert=True,
        hilbert_curve_level=10,
    )
    _ = StandardRingAttention(hilbert_config, device=device)
    print("Hilbert-optimized attention created")

    # 2. Memory pool optimization
    print("\n2. Memory Pool Optimization:")
    memory_config = RingAttentionConfig(
        segment_lengths=[4096, 8192],
        dilation_rates=[1, 2],
        use_memory_pool=True,
        preallocate_buffers=True,
        aggressive_memory_cleanup=True,
    )
    _ = StandardRingAttention(memory_config, device=device)
    print("Memory-optimized attention created")

    # 3. Error recovery
    print("\n3. Error Recovery:")
    robust_config = RingAttentionConfig(
        segment_lengths=[2048],
        dilation_rates=[1],
        enable_error_recovery=True,
        max_retry_attempts=5,
        retry_delay=0.2,
        enable_watchdog=True,
        watchdog_timeout=30.0,
    )
    _ = StandardRingAttention(robust_config, device=device)
    print("Robust attention with error recovery created")

    # 4. Performance optimization
    print("\n4. Performance Optimization:")
    perf_config = RingAttentionConfig(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        overlap_communication=True,
        use_fused_kernels=True,
        compile_mode="max-autotune",
    )
    _ = StandardRingAttention(perf_config, device=device)
    print("Performance-optimized attention created")


def main():
    """Run all examples."""
    print("Standardized Ring Attention Examples")
    print("=" * 50)

    # Run examples
    example_basic_usage()
    example_preset_configs()
    example_custom_config()
    example_advanced_features()

    # Only run distributed example if explicitly requested
    import sys

    if "--distributed" in sys.argv:
        example_distributed_training()
    else:
        print("\n(Skip distributed example - run with --distributed flag)")

    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()
