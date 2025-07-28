# Hilbert Attention Kernels

This directory contains the simplified Hilbert attention implementation with automatic optimization.

## Main Implementation

### **HilbertAttention** (`hilbert_attention.py`)
A clean, efficient implementation that automatically selects the best computation strategy.

**Features:**
- Automatic optimization based on hardware and inputs
- Hilbert curve reordering for improved cache locality
- Efficient sparse/dilated attention support
- Bounded memory caching to prevent leaks
- Full gradient support for training
- Simple, intuitive API

**Usage:**
```python
from dilated_attention_pytorch.kernels import HilbertAttention

# Standard attention
attention = HilbertAttention(
    hidden_dim=768,
    num_heads=12
)

# Dilated attention (automatically sparse)
attention = HilbertAttention(
    hidden_dim=768,
    num_heads=12,
    dilation_rate=4,
    segment_size=256
)

# Forward pass
output = attention(x)                      # With Hilbert ordering
output = attention(x, use_hilbert=False)   # Without Hilbert
output = attention(x, is_causal=True)      # With causal masking
```

## Supporting Files

### **BoundedCache** (`cache_manager.py`)
Memory-efficient cache implementation with LRU eviction.
- Prevents unbounded memory growth
- Configurable size and memory limits
- Thread-safe operations

### **Triton Kernels** (`hilbert_attention_core.py`)
Optional GPU-accelerated kernels (used automatically when available).
- Efficient forward pass implementation
- Custom backward pass for training
- Automatically used when CUDA is available

### **PyTorch Reference** (`hilbert_attention_simple.py`)
Pure PyTorch implementation used as fallback.
- No external dependencies
- Works on all devices (CPU/GPU)
- Reference for understanding the algorithm

## Key Improvements

1. **Simplified API** - No complex configuration needed
2. **Automatic Optimization** - Best strategy selected automatically
3. **Reduced Code** - ~400 lines vs ~3000 lines previously
4. **Better Performance** - Smart caching and optimization
5. **Easier Maintenance** - Single implementation to update

## Performance

The implementation automatically:
- Uses Triton kernels on CUDA devices
- Falls back to optimized PyTorch ops
- Applies sparse attention for `dilation_rate > 1`
- Manages memory efficiently
- Handles edge cases gracefully

## Cache Management

Monitor and control memory usage:

```python
# Check cache statistics
stats = attention.get_cache_stats()
print(f"Cache entries: {stats['size']}")
print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")

# Clear cache if needed
attention.clear_cache()
```

## Migration from Old Code

All previous implementations have been consolidated:
- `HilbertAttentionCore` → `HilbertAttention`
- `HilbertAttentionMemoryOptimized` → `HilbertAttention`
- `HilbertAttentionSparse*` → `HilbertAttention` with `dilation_rate`
- `UnifiedHilbertAttention` → `HilbertAttention`

Simply use `HilbertAttention` - it handles all cases automatically.