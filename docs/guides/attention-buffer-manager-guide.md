# Attention Buffer Manager Guide

## Overview

The Attention Buffer Manager is a specialized memory management system designed specifically for attention mechanisms. It provides type-aware buffer allocation with optimized strategies for different components of attention computation.

## Buffer Types

The system recognizes 10 distinct buffer types, each with specific characteristics:

### Core Attention Buffers
- **QUERY**: Query tensors (Q) - frequently reused, no zero-init needed
- **KEY**: Key tensors (K) - frequently reused, no zero-init needed  
- **VALUE**: Value tensors (V) - frequently reused, no zero-init needed
- **OUTPUT**: Attention output - must be zero-initialized for accumulation

### Intermediate Computation Buffers
- **SCORES**: Attention scores (QK^T) - large, ephemeral, NUMA-preferred
- **WEIGHTS**: Softmax weights - large, ephemeral, NUMA-preferred
- **TEMP**: Temporary workspace - small, highly reused

### Specialized Buffers
- **COMM**: Communication buffers for distributed training - persistent, pinned
- **MASK**: Attention masks (causal, padding) - small, persistent, aligned
- **CACHE**: KV cache for autoregressive models - large, persistent

## Usage

### Basic Usage

```python
from dilated_attention_pytorch.core.attention_buffer_manager import (
    create_attention_buffer_manager,
    BufferType
)

# Create buffer manager
manager = create_attention_buffer_manager(
    enable_reuse=True,         # Enable buffer caching
    enable_preallocation=False, # Pre-allocate common sizes
    lightweight=True           # Use fast configuration
)

# Allocate buffers
query = manager.allocate(BufferType.QUERY, shape=(2, 1024, 8, 64), 
                        dtype=torch.float32, device="cuda")
key = manager.allocate(BufferType.KEY, shape=(2, 1024, 8, 64))
value = manager.allocate(BufferType.VALUE, shape=(2, 1024, 8, 64))

# Use buffers...

# Return for reuse
manager.deallocate(query, BufferType.QUERY)
manager.deallocate(key, BufferType.KEY)
manager.deallocate(value, BufferType.VALUE)

# Next allocation will reuse cached buffers
query2 = manager.allocate(BufferType.QUERY, shape=(2, 1024, 8, 64))
# This reuses the previously deallocated buffer!
```

### Integration with Attention Modules

```python
from dilated_attention_pytorch.improved_dilated_attention_v2 import (
    ImprovedDilatedAttentionV2
)

# Attention module with integrated buffer management
attention = ImprovedDilatedAttentionV2(
    segment_lengths=[256, 512, 1024],
    dilation_rates=[1, 2, 4],
    enable_buffer_manager=True,
    enable_buffer_reuse=True,
    enable_preallocation=False  # Optional
)

# Forward pass automatically uses optimized allocation
output = attention(query, key, value)

# Check buffer statistics
stats = attention.get_buffer_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

### Custom Buffer Configuration

```python
from dilated_attention_pytorch.core.attention_buffer_manager import (
    AttentionBufferManager, 
    BufferConfig,
    BufferType
)

# Define custom configuration for a buffer type
custom_configs = {
    BufferType.SCORES: BufferConfig(
        typical_size_mb=16.0,      # Expected size
        reuse_frequency="high",    # Cache aggressively
        lifetime="iteration",      # Lives for one iteration
        prefer_bucketed=False,     # Too large for buckets
        prefer_numa=True,          # Use NUMA for large tensors
        zero_init=False,          # Don't waste time zeroing
    )
}

manager = AttentionBufferManager(
    custom_configs=custom_configs,
    enable_reuse=True
)
```

## Allocation Strategies

The buffer manager automatically selects the best allocation strategy:

1. **Bucketed Allocation**: For small buffers (<1MB) with high reuse
2. **NUMA-Aware Allocation**: For large buffers (>16MB) on multi-socket systems
3. **Fragment-Aware Allocation**: For buffers 2x larger than typical size
4. **Direct Allocation**: Fallback for other cases

## Performance Characteristics

### Buffer Reuse Benefits
- Eliminates allocation overhead for frequently used buffers
- Reduces memory fragmentation
- Improves cache locality

### Zero-Initialization Optimization
- OUTPUT buffers: Always zeroed (required for accumulation)
- QUERY/KEY/VALUE: Not zeroed (will be overwritten)
- Configurable per buffer type

### Memory Overhead
- Cache limited to 5-10 buffers per type
- Automatic eviction of least recently used buffers
- Minimal overhead for buffer tracking

## Best Practices

### 1. Enable Reuse for Iterative Workloads
```python
# Good for training loops
manager = create_attention_buffer_manager(enable_reuse=True)
```

### 2. Use Lightweight Mode for Speed
```python
# Faster allocation with less tracking
manager = create_attention_buffer_manager(lightweight=True)
```

### 3. Pre-allocate for Fixed Sizes
```python
# Pre-allocate common sizes
manager.preallocate_buffers(
    batch_size=2,
    seq_len=1024, 
    num_heads=8,
    head_dim=64
)
```

### 4. Clean Up Periodically
```python
# Clear caches to free memory
manager.clear_cache()  # Clear all
manager.clear_cache(BufferType.TEMP)  # Clear specific type
```

## Implementation Details

### Thread Safety
- Buffer cache operations are thread-safe
- Safe for use in multi-threaded data loading

### Device Consistency
- Handles device string representation inconsistencies
- Ensures proper cache key generation for CUDA devices

### Integration Points
- Works with existing enhanced memory pool infrastructure
- Compatible with all attention implementations
- Minimal code changes required for adoption

## Future Enhancements

1. **Adaptive Cache Sizing**: Dynamically adjust cache size based on memory pressure
2. **Pattern Learning**: Learn allocation patterns and pre-allocate accordingly
3. **Cross-Module Sharing**: Share buffers across multiple attention modules
4. **Profiling Integration**: Detailed per-buffer-type profiling