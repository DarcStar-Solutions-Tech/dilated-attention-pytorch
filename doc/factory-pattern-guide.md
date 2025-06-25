# Factory Pattern Guide

This guide explains how to use the factory pattern introduced in v0.2.0 for creating dilated attention modules with automatic implementation selection.

## Overview

The factory pattern provides a simple, unified interface for creating attention modules while automatically selecting the best implementation for your hardware and use case.

## Benefits

- **Auto-selection**: Automatically chooses the optimal implementation
- **Simplified API**: One function call instead of importing specific classes
- **Future-proof**: New optimizations are automatically available
- **Type-safe**: Validates parameters at creation time
- **Backward compatible**: Old imports still work

## Basic Usage

### Creating Attention Modules

```python
from dilated_attention_pytorch.core import (
    create_dilated_attention,
    create_multihead_dilated_attention
)

# Auto-select best implementation
attention = create_multihead_dilated_attention("auto",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4]
)

# Explicitly choose implementation
attention = create_multihead_dilated_attention("improved",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4]
)
```

## Available Implementations

### For `create_dilated_attention`:

| Implementation | Description | Best For |
|---------------|-------------|----------|
| `"auto"` | Automatically selects based on hardware | General use |
| `"standard"` | Basic dilated attention | Baseline, debugging |
| `"improved"` | Optimized with Flash Attention | Most GPUs |
| `"ring"` | Ring attention with O(n) memory | Very long sequences |

### For `create_multihead_dilated_attention`:

| Implementation | Description | Best For |
|---------------|-------------|----------|
| `"auto"` | Automatically selects based on hardware | General use |
| `"standard"` | Basic multihead dilated attention | Baseline, debugging |
| `"improved"` | Optimized with fused QKV projections | Most GPUs |
| `"ring"` | Ring attention multihead | Very long sequences |
| `"distributed"` | Multi-GPU distributed | Large models |
| `"block_sparse"` | Block-sparse optimization | Extreme speedup |

## Auto-Selection Logic

When using `"auto"`, the factory considers:

1. **Available hardware**:
   - H100 GPUs → Flash Attention 3 optimizations
   - A100 GPUs → Flash Attention 2 optimizations
   - Consumer GPUs → Standard optimizations
   - CPU → CPU-optimized implementation

2. **Sequence length**:
   - >100K tokens → Ring attention
   - >50K tokens + multi-GPU → Distributed
   - <50K tokens → Improved or standard

3. **Available libraries**:
   - Flash Attention available → Use it
   - xFormers available → Use it
   - Neither → PyTorch native

## Advanced Usage

### Using Configuration Objects

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
    use_tf32=True
)

multihead_config = MultiheadConfig(
    embed_dim=768,
    num_heads=12,
    bias=True,
    layer_norm=True,
    gamma_init=1.0
)

# Create module with configs
attention = create_multihead_dilated_attention(
    "auto",
    multihead_config=multihead_config,
    attention_config=attention_config
)
```

### Custom Implementation Selection

```python
def create_optimal_attention(seq_len, batch_size, num_gpus=1):
    """Custom logic for implementation selection."""
    
    # Very long sequences need ring attention
    if seq_len > 100_000:
        impl = "ring"
    
    # Multi-GPU with medium sequences
    elif seq_len > 50_000 and num_gpus > 1:
        impl = "distributed"
    
    # Large batch sizes benefit from block-sparse
    elif batch_size > 32 and seq_len > 10_000:
        impl = "block_sparse"
    
    # Otherwise use auto-selection
    else:
        impl = "auto"
    
    return create_multihead_dilated_attention(impl,
        embed_dim=768,
        num_heads=12,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4]
    )
```

### Registering Custom Implementations

You can register your own implementations:

```python
from dilated_attention_pytorch.core import register_attention_implementation
from dilated_attention_pytorch.core import BaseDilatedAttention

class MyCustomAttention(BaseDilatedAttention):
    """Your custom implementation."""
    def forward(self, q, k, v, **kwargs):
        # Your implementation
        pass

# Register it
register_attention_implementation("custom", MyCustomAttention)

# Now you can use it
attention = create_dilated_attention("custom",
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

## Implementation Details

### How Auto-Selection Works

```python
# Simplified auto-selection logic
def auto_select_implementation(seq_len=None, num_gpus=None):
    # Check for Flash Attention 3
    if has_flash_attention_3() and is_h100():
        return "improved"  # Uses FA3 internally
    
    # Check for very long sequences
    if seq_len and seq_len > 100_000:
        return "ring"
    
    # Check for multi-GPU
    if num_gpus and num_gpus > 1 and seq_len > 50_000:
        return "distributed"
    
    # Check for Flash Attention 2
    if has_flash_attention():
        return "improved"
    
    # Default
    return "standard"
```

### Performance Characteristics

| Implementation | Memory | Speed | Quality |
|---------------|--------|-------|---------|
| Standard | O(n²) | 1x | 100% |
| Improved | O(n²) | 2-5x | 100% |
| Ring | O(n) | 0.8-1x | 100% |
| Distributed | O(n²/p) | 1-2x | 100% |
| Block Sparse | O(n×s) | 5-50x | 95-99% |

Where:
- n = sequence length
- p = number of GPUs
- s = sparsity ratio

## Best Practices

### 1. Start with Auto-Selection

```python
# Let the factory decide
attention = create_multihead_dilated_attention("auto", ...)
```

### 2. Profile Before Optimizing

```python
implementations = ["auto", "standard", "improved", "ring"]
for impl in implementations:
    attention = create_multihead_dilated_attention(impl, ...)
    # Benchmark your specific use case
```

### 3. Use Configs for Complex Setups

```python
# Reusable configurations
base_config = DilatedAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)

# Create variants
small_model = create_multihead_dilated_attention("auto",
    embed_dim=512, num_heads=8, attention_config=base_config)

large_model = create_multihead_dilated_attention("auto",
    embed_dim=1024, num_heads=16, attention_config=base_config)
```

### 4. Consider Memory vs Speed Tradeoffs

```python
# For memory-constrained environments
if torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3:  # 16GB
    impl = "ring"  # O(n) memory
else:
    impl = "improved"  # Faster but O(n²) memory
```

## Troubleshooting

### Implementation Not Available

```python
try:
    attention = create_multihead_dilated_attention("distributed", ...)
except ValueError as e:
    print(f"Distributed not available: {e}")
    # Fall back to auto
    attention = create_multihead_dilated_attention("auto", ...)
```

### Checking Available Implementations

```python
from dilated_attention_pytorch.core import get_available_implementations

# List all available implementations
available = get_available_implementations()
print(f"Available implementations: {available}")
```

### Getting Implementation Info

```python
from dilated_attention_pytorch.core import get_implementation_info

# Get details about an implementation
info = get_implementation_info("improved")
print(f"Improved attention info: {info}")
```

## Migration from Direct Imports

### Old Way
```python
from dilated_attention_pytorch.improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention

attention = ImprovedMultiheadDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

### New Way (Recommended)
```python
from dilated_attention_pytorch.core import create_multihead_dilated_attention

attention = create_multihead_dilated_attention("improved",  # or "auto"
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

## Summary

The factory pattern simplifies creating dilated attention modules while providing:

- Automatic optimization selection
- Cleaner, more maintainable code
- Future-proof implementation
- Type-safe configuration
- Full backward compatibility

Start with `"auto"` and let the factory optimize for your hardware!