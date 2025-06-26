# Migration Guide: v0.1.x to v0.2.0

This guide helps you upgrade from dilated-attention-pytorch v0.1.x to v0.2.0, which includes a major refactoring for better performance and maintainability.

## Overview of Changes

### Core Architecture Refactoring
- **New `core` module** with shared base classes and utilities
- **Factory pattern** for creating attention modules
- **Type-safe configuration** system
- **Unified memory management** with adaptive cleanup
- **Utils directory** with reorganized utility functions

### Key Benefits
- 50-60% code reduction through shared implementations
- Auto-selection of optimal implementation for your hardware
- Better memory efficiency and error handling
- Cleaner, more maintainable codebase
- Full backward compatibility

## Quick Start

### Before (v0.1.x)
```python
from dilated_attention_pytorch.improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention

attention = ImprovedMultiheadDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)
```

### After (v0.2.0) - Recommended
```python
from dilated_attention_pytorch.core import create_multihead_dilated_attention

# Auto-select best implementation
attention = create_multihead_dilated_attention("auto",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)

# Or explicitly choose implementation
attention = create_multihead_dilated_attention("improved",  # same as before
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)
```

## Backward Compatibility

**All existing code continues to work!** The old import paths are maintained for backward compatibility:

```python
# These still work in v0.2.0
from dilated_attention_pytorch.dilated_attention import DilatedAttention
from dilated_attention_pytorch.multihead_dilated_attention import MultiheadDilatedAttention
from dilated_attention_pytorch.improved_dilated_attention import ImprovedDilatedAttention
from dilated_attention_pytorch.improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
# ... etc
```

## New Features in v0.2.0

### 1. Factory Pattern

The factory pattern allows easy creation and auto-selection:

```python
from dilated_attention_pytorch.core import (
    create_dilated_attention,
    create_multihead_dilated_attention
)

# For base dilated attention
attention = create_dilated_attention("auto",  # or "standard", "improved", "ring"
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)

# For multihead attention
mha = create_multihead_dilated_attention("auto",  # auto-selects best
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

Available implementations:
- `"auto"` - Automatically selects based on your hardware
- `"standard"` - Basic dilated attention
- `"improved"` - Optimized with Flash Attention support
- `"ring"` - Ring attention for extreme sequence lengths
- `"distributed"` - Multi-GPU distributed attention
- `"block_sparse"` - Block-sparse attention (5-50x speedup)

### 2. Type-Safe Configuration

Use configuration dataclasses for better validation:

```python
from dilated_attention_pytorch.core import (
    DilatedAttentionConfig,
    MultiheadConfig,
    create_multihead_dilated_attention
)

# Create configurations
attention_config = DilatedAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1,
    use_tf32=True  # New optimization flag
)

multihead_config = MultiheadConfig(
    embed_dim=768,
    num_heads=12,
    bias=True,
    layer_norm=True,
    layer_norm_eps=1e-5,
    gamma_init=1.0  # MAGNETO initialization
)

# Use with factory
attention = create_multihead_dilated_attention(
    "improved",
    multihead_config=multihead_config,
    attention_config=attention_config
)
```

### 3. Unified Memory Management

All implementations now share an adaptive memory pool:

```python
# Memory management is automatic, but you can monitor it
from dilated_attention_pytorch.core import get_memory_pool_info

info = get_memory_pool_info()
print(f"Allocated buffers: {info['total_buffers']}")
print(f"Memory usage: {info['total_memory_mb']:.1f} MB")
```

### 4. Better Error Messages

The new validation system provides clearer error messages:

```python
# Before: ValueError: segment_lengths and dilation_rates must have same length
# After: ValueError: segment_lengths and dilation_rates must have same length: got 3 and 2

# Before: ValueError: embed_dim must be divisible by num_heads
# After: ValueError: embed_dim (768) must be divisible by num_heads (13). Got head_dim=59.08
```

## Migration Steps

### Step 1: Update Imports (Optional but Recommended)

Replace direct imports with factory imports:

```python
# Old
from dilated_attention_pytorch.improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention
from dilated_attention_pytorch.ring_multihead_dilated_attention import RingMultiheadDilatedAttention

# New (recommended)
from dilated_attention_pytorch.core import create_multihead_dilated_attention
```

### Step 2: Use Factory Methods

Replace direct instantiation with factory calls:

```python
# Old
attention = ImprovedMultiheadDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)

# New
attention = create_multihead_dilated_attention("improved",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

### Step 3: Leverage Auto-Selection

Let the factory choose the best implementation:

```python
# Automatically selects:
# - "improved" if Flash Attention is available
# - "ring" for very long sequences
# - "distributed" if multiple GPUs detected
# - "standard" as fallback
attention = create_multihead_dilated_attention("auto",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4]
)
```

### Step 4: Use Configuration Objects (Optional)

For complex configurations, use the config classes:

```python
from dilated_attention_pytorch.core import (
    DilatedAttentionConfig,
    MultiheadConfig,
    create_multihead_dilated_attention
)

# Easier to manage and reuse
base_config = DilatedAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)

# Create variants
attention_12h = create_multihead_dilated_attention("auto",
    embed_dim=768, num_heads=12, attention_config=base_config)

attention_16h = create_multihead_dilated_attention("auto",
    embed_dim=1024, num_heads=16, attention_config=base_config)
```

## Common Patterns

### Pattern 1: Conditional Implementation Selection

```python
def create_model_attention(seq_len, num_gpus=1):
    if seq_len > 100_000:
        impl = "ring"  # Use ring attention for very long sequences
    elif num_gpus > 1:
        impl = "distributed"  # Use distributed for multi-GPU
    else:
        impl = "improved"  # Default to improved
    
    return create_multihead_dilated_attention(impl,
        embed_dim=768,
        num_heads=12,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4]
    )
```

### Pattern 2: Experiment with Different Implementations

```python
implementations = ["standard", "improved", "ring"]
results = {}

for impl in implementations:
    attention = create_multihead_dilated_attention(impl,
        embed_dim=768,
        num_heads=12,
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2]
    )
    
    # Benchmark or test
    results[impl] = benchmark(attention)
```

### Pattern 3: Custom Configuration Presets

```python
# Define presets
PRESETS = {
    "small": {
        "segment_lengths": [1024, 2048],
        "dilation_rates": [1, 2],
    },
    "medium": {
        "segment_lengths": [2048, 4096, 8192],
        "dilation_rates": [1, 2, 4],
    },
    "large": {
        "segment_lengths": [4096, 8192, 16384, 32768],
        "dilation_rates": [1, 2, 4, 8],
    }
}

def create_attention_from_preset(preset_name, embed_dim, num_heads):
    preset = PRESETS[preset_name]
    return create_multihead_dilated_attention("auto",
        embed_dim=embed_dim,
        num_heads=num_heads,
        **preset
    )
```

## Troubleshooting

### Import Errors

If you get import errors after upgrading:

```python
# If this fails:
from dilated_attention_pytorch.core import create_multihead_dilated_attention

# Fall back to:
from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention
```

### Performance Differences

The refactored code should have identical or better performance. If you notice degradation:

1. Ensure you're using the same implementation:
   ```python
   # Explicitly use "improved" instead of "auto"
   attention = create_multihead_dilated_attention("improved", ...)
   ```

2. Check PyTorch and CUDA versions - the auto-selection optimizes for your environment

3. Verify Flash Attention is properly installed if you were using it before

### Memory Usage

The new unified memory pool is more efficient but has different allocation patterns:

```python
# Monitor memory usage
from dilated_attention_pytorch.core import get_memory_pool_info, clear_memory_pool

# Check current usage
info = get_memory_pool_info()
print(f"Memory usage: {info['total_memory_mb']:.1f} MB")

# Clear if needed (automatic in most cases)
clear_memory_pool()
```

## Summary

The v0.2.0 refactoring provides:
- ✅ Full backward compatibility
- ✅ Cleaner API with factory pattern
- ✅ Auto-selection of optimal implementation
- ✅ Better memory management
- ✅ Type-safe configuration
- ✅ 50-60% code reduction internally

While your existing code will continue to work, we recommend adopting the factory pattern for new code to benefit from auto-selection and future optimizations.

For questions or issues, please open an issue on GitHub.