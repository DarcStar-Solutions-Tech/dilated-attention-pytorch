# Kernel Consolidation Guide

## Overview

As of v0.4.0, the dilated attention PyTorch implementation has consolidated 10 separate kernel implementations into a single, unified `UnifiedHilbertAttention` class. This consolidation reduces code duplication by ~60% while preserving all unique optimizations through configurable modes.

## Consolidated Implementations

The following 10 implementations have been unified:

1. **HilbertAttentionCore** → Standard implementation
2. **HilbertAttentionMemoryOptimized** → Memory-optimized mode
3. **HilbertAttentionOptimizedSelective** → Selective sparse mode
4. **HilbertAttentionSparseOptimized** → Direct sparse mode
5. **HilbertAttentionSparseSimple** → Sparse with PyTorch backend
6. **HilbertAttentionStrided** → Strided access mode
7. **HilbertAttentionStridedSimple** → Strided with PyTorch backend
8. **HilbertAttentionSimple** → PyTorch-only backend
9. **HilbertAttentionV2** → Optimized memory mode
10. **HilbertAttentionTester** → Testing configuration

## Key Improvements

### 1. Bounded Memory Caches
All implementations now use `BoundedCache` with LRU eviction to prevent unbounded memory growth:

```python
# Old (unbounded)
self._hilbert_cache = {}  # Could grow indefinitely!

# New (bounded)
self._hilbert_cache = BoundedCache(
    max_size=32,  # Maximum entries
    max_memory_mb=100.0  # Maximum memory usage
)
```

### 2. Unified Architecture

```python
# New unified implementation
from dilated_attention_pytorch.kernels import UnifiedHilbertAttention

# Replaces all 10 implementations with configuration options
attention = UnifiedHilbertAttention(
    hidden_dim=768,
    num_heads=12,
    memory_mode="optimized",  # Memory optimization level
    sparse_mode="direct",     # Sparse optimization strategy
    access_mode="strided",    # Memory access pattern
    backend="auto"           # Computation backend
)
```

### 3. Configuration Modes

#### Memory Modes
- `"standard"`: No special memory optimization (default)
- `"optimized"`: Moderate memory optimization
- `"aggressive"`: Aggressive memory optimization with FP16 accumulation

#### Sparse Modes
- `"standard"`: Full Hilbert mapping with filtering (default)
- `"selective"`: On-the-fly Hilbert for selected positions
- `"direct"`: Direct sparse position computation (fastest)

#### Access Modes
- `"standard"`: Block-based access (default)
- `"strided"`: Strided memory access for dilated patterns

#### Backend Modes
- `"auto"`: Automatically select based on hardware (default)
- `"triton"`: Force Triton kernels (requires CUDA)
- `"pytorch"`: Force PyTorch implementation

## Migration Guide

### Manual Migration Examples

#### 1. Basic Implementation
```python
# Old
from dilated_attention_pytorch.kernels import HilbertAttentionCore
attn = HilbertAttentionCore(hidden_dim=768, num_heads=12)

# New
from dilated_attention_pytorch.kernels import UnifiedHilbertAttention
attn = UnifiedHilbertAttention(hidden_dim=768, num_heads=12)
```

#### 2. Memory Optimized
```python
# Old
from dilated_attention_pytorch.kernels import HilbertAttentionMemoryOptimized
attn = HilbertAttentionMemoryOptimized(hidden_dim=768, num_heads=12)

# New
attn = UnifiedHilbertAttention(
    hidden_dim=768, 
    num_heads=12,
    memory_mode="aggressive"
)
```

#### 3. Sparse Optimized
```python
# Old
from dilated_attention_pytorch.kernels import HilbertAttentionSparseOptimized
attn = HilbertAttentionSparseOptimized(
    hidden_dim=768, 
    num_heads=12,
    dilation_rate=4
)

# New
attn = UnifiedHilbertAttention(
    hidden_dim=768,
    num_heads=12, 
    dilation_rate=4,
    sparse_mode="direct"
)
```

#### 4. Combined Optimizations
```python
# New unified approach - combine any optimizations
attn = UnifiedHilbertAttention(
    hidden_dim=768,
    num_heads=12,
    segment_size=128,
    dilation_rate=4,
    memory_mode="optimized",
    sparse_mode="direct",
    access_mode="strided",
    backend="triton"
)
```

### Factory Pattern

For convenience, use the factory function:

```python
from dilated_attention_pytorch.kernels import create_hilbert_attention

# Create specific implementations
attn = create_hilbert_attention("memory_optimized", 
    hidden_dim=768,
    num_heads=12
)

# Available types:
# - "standard": Basic implementation
# - "memory_optimized": Aggressive memory optimization
# - "sparse": Direct sparse optimization
# - "selective": Selective sparse optimization
# - "strided": Strided access pattern
# - "simple": PyTorch-only backend
```

## Benefits of Consolidation

1. **Reduced Code Duplication**: ~60% reduction in code size
2. **Easier Maintenance**: Single implementation to maintain and test
3. **Flexible Configuration**: Mix and match optimizations as needed
4. **Bounded Memory Usage**: Prevents memory leaks from unbounded caches
5. **Consistent API**: Same interface across all optimization modes
6. **Better Testing**: Comprehensive test coverage for all modes

## Performance Considerations

The unified implementation maintains the same performance characteristics as the original implementations:

- **Memory Mode**: Reduces memory usage by 15-30% with aggressive mode
- **Sparse Mode**: Provides up to 64x reduction in computation for high dilation rates
- **Strided Mode**: Optimizes memory bandwidth for dilated patterns
- **Backend Selection**: Automatically uses best backend for your hardware

## Cache Management

Monitor and manage caches:

```python
# Get cache statistics
stats = attn.get_cache_stats()
print(f"Cache size: {stats['size']} entries")
print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")

# Clear caches when needed
attn.clear_cache()
```

## Troubleshooting

### Issue: Out of Memory with Old Code
If you experience OOM errors with old implementations, migrate to the unified version which has bounded caches:

```python
# Replace unbounded implementation
old_attn = HilbertAttentionCore(...)  # May cause OOM

# With bounded unified implementation  
new_attn = UnifiedHilbertAttention(
    ...,
    cache_size=16,  # Limit cache entries
    cache_memory_mb=50.0  # Limit cache memory
)
```

### Issue: Performance Regression
If you notice performance differences after migration, try matching the exact optimization mode:

```python
# If using HilbertAttentionSparseOptimized before
attn = UnifiedHilbertAttention(
    ...,
    sparse_mode="direct",  # Match the optimization
    backend="triton"  # Force Triton if needed
)
```

## Future Development

The unified architecture makes it easier to add new optimizations:

1. New optimization modes can be added as mixins
2. Existing modes can be enhanced without affecting others
3. Testing is simplified with the unified interface

For questions or issues with the consolidation, please open an issue on GitHub.