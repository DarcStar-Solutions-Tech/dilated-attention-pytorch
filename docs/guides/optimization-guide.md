# Optimization Guide

This guide covers the optimization features available in the dilated attention implementation, including pattern caching and memory pooling.

## Pattern Caching

Pattern caching significantly improves performance by storing and reusing computed attention patterns across forward passes.

### Overview

Pattern caching eliminates redundant computation of dilated attention indices by:
- Storing computed patterns in a global LRU cache
- Reusing patterns when the same configuration is encountered
- Maintaining thread-safe access for multi-threaded environments

### Benefits

Based on benchmarking results:
- **2x speedup** for repeated forward passes
- **23.2% memory reduction** through CPU storage
- **90%+ cache hit rates** in typical training scenarios

### When to Use

Enable pattern caching when:
- Multiple forward passes with the same sequence configuration
- Using Ring Attention implementations (V2/V3)
- Training or inference with fixed sequence lengths
- Memory is not severely constrained

### How to Enable

#### Ring Attention V2/V3
```python
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2

model = RingDilatedAttentionV2(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    use_pattern_cache=True  # Enable pattern caching
)
```

#### Global Cache Management
```python
from dilated_attention_pytorch.core import (
    get_global_pattern_cache,
    reset_global_pattern_cache
)

# Get cache statistics
cache = get_global_pattern_cache()
stats = cache.get_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")

# Clear cache when needed
reset_global_pattern_cache()
```

### Configuration Options

The pattern cache can be configured through environment variables:
```bash
# Set maximum cache size (default: 1000 patterns)
export PATTERN_CACHE_MAX_SIZE=2000

# Enable cache profiling
export PATTERN_CACHE_ENABLE_PROFILING=1
```

## Memory Pooling

Memory pooling reduces allocation overhead and memory fragmentation for large tensor operations.

### Overview

The memory pool system:
- Pre-allocates and reuses large tensors
- Reduces CUDA allocation overhead
- Implements adaptive cleanup based on memory pressure
- Supports both CPU and GPU tensors

### Benefits

Based on benchmarking:
- **15-30% memory reduction** for long sequences
- **Reduced allocation overhead** through buffer reuse
- **Better memory locality** and reduced fragmentation
- **Automatic memory pressure handling**

### When to Use

Enable memory pooling when:
- Working with sequences > 32K tokens
- Training large models with limited GPU memory
- Experiencing frequent OOM errors
- Running long training sessions

### How to Enable

#### Basic Usage
```python
from dilated_attention_pytorch import DilatedAttention

model = DilatedAttention(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    enable_memory_pool=True  # Enable memory pooling
)
```

#### Advanced Configuration
```python
from dilated_attention_pytorch import ImprovedDilatedAttention

model = ImprovedDilatedAttention(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    enable_memory_pool=True,
    enable_profiling=True  # Enable pool profiling
)
```

### Memory Pool Management

```python
from dilated_attention_pytorch.core import (
    get_global_memory_pool,
    reset_global_memory_pool
)

# Get pool statistics
pool = get_global_memory_pool()
stats = pool.get_stats()
print(f"Total allocated: {stats['total_allocated'] / 1024 / 1024:.1f} MB")
print(f"Peak allocated: {stats['peak_allocated'] / 1024 / 1024:.1f} MB")
print(f"Allocation count: {stats['allocation_count']}")

# Clear pool when needed
reset_global_memory_pool()
```

### Configuration Options

Memory pool behavior can be configured:
```python
# Use lightweight pool for faster allocation
model = RingDilatedAttentionV2(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    enable_memory_pool=True,
    lightweight_pool=True  # Faster but less features
)
```

## Combined Optimizations

For best performance, combine both optimizations:

```python
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3

model = RingDilatedAttentionV3(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    use_pattern_cache=True,     # Enable pattern caching
    enable_memory_pool=True,    # Enable memory pooling
    cache_on_gpu=True          # Keep patterns on GPU (V3 only)
)
```

### Combined Benefits

When using both optimizations:
- **2.5-3x overall speedup** for repeated operations
- **40-50% memory reduction** for large sequences
- **Minimal overhead** for cache misses
- **Automatic resource management**

## Implementation Support

### Current Support Matrix

| Implementation | Pattern Cache | Memory Pool | Notes |
|----------------|---------------|-------------|-------|
| DilatedAttention | ❌ | ✅ | Basic implementation |
| ImprovedDilatedAttention | ❌ | ✅ | Enhanced features |
| MultiheadDilatedAttention | ❌ | ❌ | Planned for future |
| ImprovedMultiheadDilatedAttention | ❌ | ❌ | Planned for future |
| RingDilatedAttentionV2 | ✅ | ✅ | Full support |
| RingDilatedAttentionV3 | ✅ | ✅ | GPU-resident cache |
| BlockSparseRingDilatedAttention | ❌ | ❌ | Performance optimized |

### Adding Support

To add optimization support to new implementations:

1. **Pattern Caching**:
   ```python
   from dilated_attention_pytorch.core import get_global_pattern_cache
   
   class MyAttention(nn.Module):
       def __init__(self, ..., use_pattern_cache=False):
           self.use_pattern_cache = use_pattern_cache
           self._pattern_cache = get_global_pattern_cache() if use_pattern_cache else None
   ```

2. **Memory Pooling**:
   ```python
   from dilated_attention_pytorch.core import get_global_memory_pool
   
   class MyAttention(nn.Module):
       def __init__(self, ..., enable_memory_pool=False):
           self.enable_memory_pool = enable_memory_pool
           self._memory_pool = get_global_memory_pool() if enable_memory_pool else None
   ```

## Performance Guidelines

### Sequence Length Recommendations

| Sequence Length | Pattern Cache | Memory Pool | Notes |
|-----------------|---------------|-------------|-------|
| < 4K tokens | Optional | No | Overhead may exceed benefits |
| 4K-32K tokens | Yes | Optional | Cache provides main benefit |
| 32K-128K tokens | Yes | Yes | Both recommended |
| > 128K tokens | Yes | Required | Essential for stability |

### Hardware Considerations

#### GPU Memory
- **8GB**: Enable both optimizations for sequences > 16K
- **16GB**: Enable memory pool for sequences > 64K
- **24GB+**: Optional for most use cases, beneficial for > 128K

#### Multi-GPU
- Pattern cache is shared across GPUs
- Memory pools are per-GPU
- Distributed implementations handle synchronization

### Profiling and Monitoring

Enable profiling to understand optimization impact:

```python
# Enable profiling for both systems
model = RingDilatedAttentionV3(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    use_pattern_cache=True,
    enable_memory_pool=True,
    enable_profiling=True  # Enables detailed statistics
)

# After running some forward passes
cache_stats = model._pattern_cache.get_stats()
pool_stats = model._memory_pool.get_stats()

print(f"Cache efficiency: {cache_stats['hit_rate']:.1%}")
print(f"Memory reuse rate: {pool_stats.get('reuse_rate', 0):.1%}")
```

## Troubleshooting

### Common Issues

1. **High Cache Miss Rate**
   - Check if sequence configurations are changing
   - Verify cache size is sufficient
   - Consider increasing `PATTERN_CACHE_MAX_SIZE`

2. **Memory Pool Fragmentation**
   - Enable adaptive cleanup
   - Use lightweight pool for simple allocations
   - Monitor pool statistics

3. **OOM with Optimizations Enabled**
   - Reduce cache size
   - Enable aggressive memory cleanup
   - Use CPU offloading for patterns

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger("dilated_attention_pytorch").setLevel(logging.DEBUG)

# Check optimization status
print(f"Pattern cache enabled: {model.use_pattern_cache}")
print(f"Memory pool enabled: {model.enable_memory_pool}")
print(f"Cache size: {len(model._pattern_cache._cache)}")
```

## Best Practices

1. **Start Conservative**: Enable optimizations one at a time
2. **Monitor Metrics**: Track hit rates and memory usage
3. **Profile First**: Understand your workload characteristics
4. **Adjust Thresholds**: Tune based on your hardware
5. **Clear Periodically**: Reset caches between epochs if needed

## Future Enhancements

Planned optimization improvements:
- Distributed pattern cache synchronization
- Automatic cache size tuning
- GPU-CPU memory tiering
- Compression for cached patterns
- Integration with PyTorch's memory management