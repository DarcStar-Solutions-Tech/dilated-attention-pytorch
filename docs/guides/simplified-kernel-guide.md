# Simplified Kernel Implementation Guide

## Overview

As of the latest update, the Hilbert attention kernel implementation has been dramatically simplified. We've removed all the unnecessary configuration options and optimization modes, resulting in a clean, efficient implementation that automatically selects the best computation strategy.

## Key Simplifications

### Before: Complex Configuration
```python
# Old approach - too many options!
attn = UnifiedHilbertAttention(
    hidden_dim=768,
    num_heads=12,
    memory_mode="aggressive",    # What does this mean?
    sparse_mode="direct",        # vs "selective"? 
    access_mode="strided",       # vs "standard"?
    backend="triton",           # Do I need to know this?
)
```

### After: Simple and Automatic
```python
# New approach - just works!
attn = HilbertAttention(
    hidden_dim=768,
    num_heads=12,
    dilation_rate=4  # Optional: for sparse attention
)
```

## What Was Removed

1. **Memory Optimization Modes** - The implementation now automatically manages memory efficiently
2. **Sparse Modes** - Sparse attention is automatically applied when `dilation_rate > 1`
3. **Access Modes** - Memory access patterns are optimized automatically
4. **Backend Selection** - Uses Triton when available, PyTorch fallback otherwise
5. **Complex Mixins** - Removed multiple inheritance complexity
6. **Configuration Classes** - No more configuration objects to manage

## What Remains

### Core Features
- **Hilbert Curve Reordering** - For improved cache locality
- **Dilated/Sparse Attention** - Essential for long sequences
- **Bounded Memory Caches** - Prevents memory leaks
- **Automatic Optimization** - Selects best strategy based on inputs

### Simple API
```python
from dilated_attention_pytorch.kernels import HilbertAttention

# Standard attention
attn = HilbertAttention(hidden_dim=768, num_heads=12)

# Dilated attention (automatically sparse)
attn = HilbertAttention(
    hidden_dim=768,
    num_heads=12,
    dilation_rate=4,
    segment_size=256
)

# Forward pass
output = attn(x)                        # With Hilbert ordering
output = attn(x, use_hilbert=False)     # Without Hilbert ordering  
output = attn(x, is_causal=True)        # With causal masking
```

## Performance

The simplified implementation maintains excellent performance:
- Automatically uses optimized PyTorch ops when available
- Falls back to efficient custom implementation
- Smart caching of Hilbert mappings
- Efficient sparse attention for dilated patterns

## Code Reduction

- **10 implementations → 1 implementation**
- **~3000 lines → ~400 lines** 
- **Complex configuration → Simple parameters**
- **Multiple files → Single file**

## Benefits

1. **Easier to Use** - No need to understand optimization modes
2. **Easier to Maintain** - Single implementation to update
3. **Better Performance** - Automatic optimization selection
4. **Cleaner Code** - No complex inheritance hierarchies
5. **Fewer Bugs** - Less code means fewer places for bugs

## Migration from Old Code

If you were using the old implementations:

```python
# Old
from dilated_attention_pytorch.kernels import (
    HilbertAttentionCore,
    HilbertAttentionMemoryOptimized,
    UnifiedHilbertAttention
)

# New - just use HilbertAttention
from dilated_attention_pytorch.kernels import HilbertAttention
```

All functionality is preserved - the implementation just makes better decisions automatically.

## Cache Management

The implementation includes smart cache management:

```python
# Check cache usage
stats = attn.get_cache_stats()
print(f"Cache entries: {stats['size']}")
print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")

# Clear cache if needed
attn.clear_cache()
```

## Summary

The new simplified implementation provides all the benefits of Hilbert-ordered attention with dilated/sparse support, but without the complexity. It automatically makes optimal choices based on your hardware and inputs, letting you focus on your model rather than implementation details.