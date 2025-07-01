# Dilated Attention PyTorch - Benchmark Suite

This directory contains comprehensive benchmarks for all dilated attention implementations. The benchmarks are organized into core and specialized categories for easy navigation.

## Directory Structure

```
benchmarks/
├── core/                              # Core benchmark suite
│   ├── benchmark_implementations.py   # Compare all implementations
│   ├── benchmark_distributed.py       # Multi-GPU distributed testing
│   └── benchmark_backends.py          # Attention backend comparisons
├── specialized/                       # Specialized benchmarks
│   ├── benchmark_extreme_sequences.py # Test extreme sequence lengths
│   ├── benchmark_flash_attention_3.py # Flash Attention 3 specific
│   └── benchmark_ring_attention.py    # Ring Attention variants
└── README.md                          # This file
```

## Quick Start

### 1. Compare All Implementations

```bash
# Compare base, improved, and multihead variants
python benchmarks/core/benchmark_implementations.py

# Options:
# --batch-size: Batch size (default: 2)
# --seq-length: Sequence length (default: 8192)
# --num-heads: Number of attention heads (default: 8)
# --head-dim: Head dimension (default: 64)
```

### 2. Test Distributed Performance

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=2 benchmarks/core/benchmark_distributed.py

# Multiple nodes
torchrun --nnodes=2 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    benchmarks/core/benchmark_distributed.py
```

### 3. Compare Attention Backends

```bash
# Compare Flash Attention, xformers, SDPA, and standard attention
python benchmarks/core/benchmark_backends.py

# Options:
# --test-flash: Include Flash Attention (default: True)
# --test-xformers: Include xformers (default: True)
# --test-sdpa: Include PyTorch SDPA (default: True)
```

### 4. Test Extreme Sequence Lengths

```bash
# Test sequences up to 1M tokens
python benchmarks/specialized/benchmark_extreme_sequences.py

# Options:
# --max-seq-length: Maximum sequence length to test (default: 1048576)
# --implementation: Which implementation to test (default: 'all')
```

### 5. Benchmark Ring Attention

```bash
# Comprehensive Ring Attention benchmarks
python benchmarks/specialized/benchmark_ring_attention.py

# For multi-GPU Ring Attention:
torchrun --nproc_per_node=2 benchmarks/specialized/benchmark_ring_attention.py
```

### 6. Flash Attention 3 Benchmarks

```bash
# Compare FA3 vs FA2 vs standard attention
python benchmarks/specialized/benchmark_flash_attention_3.py

# Note: Requires Flash Attention 3 installation
```

## Benchmark Categories

### Core Benchmarks

#### `benchmark_implementations.py`
- **Purpose**: Compare all dilated attention implementations
- **Tests**: DilatedAttention, ImprovedDilatedAttention, MultiheadDilatedAttention variants
- **Metrics**: Throughput, memory usage, latency
- **Use when**: Choosing which implementation to use

#### `benchmark_distributed.py`
- **Purpose**: Test multi-GPU distributed performance
- **Tests**: DistributedImprovedDilatedAttention, Ring Attention variants
- **Metrics**: Scaling efficiency, communication overhead
- **Use when**: Planning distributed training

#### `benchmark_backends.py`
- **Purpose**: Compare different attention computation backends
- **Tests**: Flash Attention, xformers, PyTorch SDPA, standard attention
- **Metrics**: Backend-specific performance characteristics
- **Use when**: Optimizing for specific hardware

### Specialized Benchmarks

#### `benchmark_extreme_sequences.py`
- **Purpose**: Test performance with very long sequences
- **Tests**: All implementations with sequences up to 1M tokens
- **Metrics**: Memory efficiency, throughput at scale
- **Use when**: Working with long documents or sequences

#### `benchmark_ring_attention.py`
- **Purpose**: Comprehensive Ring Attention testing
- **Tests**: Optimization comparison, single vs multi-GPU
- **Metrics**: Impact of optimizations (pattern caching, memory pool, etc.)
- **Use when**: Using Ring Attention for extreme sequences

#### `benchmark_flash_attention_3.py`
- **Purpose**: Test Flash Attention 3 specific features
- **Tests**: FA3 vs FA2 performance, new features
- **Metrics**: Speedup from FA3, memory efficiency
- **Use when**: Using latest Flash Attention features

## Performance Tips

1. **GPU Selection**: Modern GPUs (A100, H100) show best performance
2. **Dtype Selection**: Use FP16/BF16 on modern GPUs, FP32 on Pascal
3. **Backend Selection**: Flash Attention > xformers > SDPA > standard
4. **Memory Pool**: Enable for sequences > 16K tokens
5. **Pattern Caching**: Always enable for multi-batch scenarios

## Interpreting Results

Each benchmark outputs:
- **Throughput**: Tokens processed per second
- **Latency**: Time per forward pass (ms)
- **Memory**: Peak GPU memory usage (MB)
- **Efficiency**: For distributed benchmarks

Example output:
```
Implementation: RingDilatedAttentionV2Collective
Throughput: 1,429,956 tokens/s
Latency: 2.86 ± 0.20 ms
Memory: 1578.0 MB
Backend: xformers
Dtype: torch.float32
```

## Custom Benchmarks

See `examples/example_custom_benchmark.py` for creating custom benchmarks.

## Troubleshooting

### Out of Memory Errors
- Reduce batch size or sequence length
- Enable gradient checkpointing
- Use Ring Attention for extreme sequences

### Distributed Errors
- Ensure all GPUs are visible: `nvidia-smi`
- Check NCCL environment variables
- Use `NCCL_DEBUG=INFO` for debugging

### Performance Issues
- Check GPU utilization: `nvidia-smi dmon`
- Verify correct backend is being used
- Ensure optimal dtype for your GPU architecture