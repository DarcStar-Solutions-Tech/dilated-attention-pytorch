# Dilated Attention Kernels

This directory contains production-ready kernel implementations for dilated attention with full gradient support.

## Available Implementations

### 1. **HilbertAttentionCore** (`hilbert_attention_core.py`)
The main unified Hilbert attention implementation with all optimizations.

**Features:**
- Efficient Triton kernels for forward pass
- Optimized PyTorch-based backward pass with full gradient support
- Custom autograd function (`HilbertAttentionFunction`)
- Configurable custom backward (can be disabled for debugging)
- Hilbert mapping caching for efficiency
- Support for both Hilbert-ordered and standard attention

**Key Components:**
- `hilbert_attention_kernel`: Triton kernel with Hilbert curve reordering
- `standard_attention_kernel`: Triton kernel without reordering (for comparison)
- `HilbertAttentionFunction`: Custom autograd with optimized backward pass
- `HilbertAttentionCore`: Main module with QKV projections

**Usage:**
```python
from dilated_attention_pytorch.kernels import HilbertAttentionCore

attention = HilbertAttentionCore(
    hidden_dim=768,
    num_heads=12,
    segment_size=128,
    dilation_rate=2,
    dropout=0.1,
    use_custom_backward=True  # Enable optimized backward
)

# Forward pass
output = attention(x, use_hilbert=True)
```

### 2. **HilbertAttentionTritonWrapper** (`hilbert_attention_triton_wrapper.py`)
A wrapper that adapts HilbertAttentionCore to accept separate q, k, v tensors.

**Purpose:**
- Provides compatibility with benchmark interfaces expecting `forward(q, k, v)`
- Wraps HilbertAttentionCore while maintaining its gradient support
- Includes `HilbertAttentionTritonFixed` alias for backward compatibility

**Usage:**
```python
from dilated_attention_pytorch.kernels import HilbertAttentionTritonFixed

attention = HilbertAttentionTritonFixed(
    segment_lengths=[128, 256],
    dilation_rates=[1, 2],
    dropout=0.1,
    num_heads=8,
    head_dim=64
)

# Forward pass with separate q, k, v
output = attention(q, k, v)
```

## Gradient Support

Both implementations have full gradient support for training:

1. **HilbertAttentionCore**: 
   - Custom backward pass optimized for Hilbert-ordered tensors
   - Efficient gradient computation using pre-reordered tensors
   - Supports gradient checkpointing

2. **HilbertAttentionTritonWrapper**: 
   - Inherits gradient support from HilbertAttentionCore
   - Compatible with standard PyTorch autograd

## Performance Considerations

- The Hilbert curve reordering improves cache locality for long sequences
- Custom backward pass is ~2x faster than PyTorch's automatic differentiation
- Use `use_hilbert=False` to compare against standard attention
- Segment size and dilation rate significantly impact performance

## Implementation Details

### Hilbert Curve Mapping
The implementation uses a snake pattern approximation of Hilbert curves:
- Provides similar cache locality benefits
- Simpler to compute than true Hilbert curves
- Works well with power-of-2 and non-power-of-2 sequence lengths

### Memory Efficiency
- Reuses buffers where possible
- Caches Hilbert mappings to avoid recomputation
- Optimized for different sequence length ranges

### Triton Kernel Optimization
- Block sizes optimized for different GPU architectures
- Efficient memory access patterns
- Minimal synchronization overhead

## Removed Implementations

The following implementations were removed due to lack of gradient support:
- `hilbert_attention_core_fixed.py` - Simplified version without backward pass
- `hilbert_attention_kernel_simple.py` - Kernel-only implementation
- `hilbert_attention_triton_v2.py` - Experimental version with PyTorch fallback
- `hilbert_attention_triton_v2_simple.py` - Simplified experimental version
- `hilbert_dilated_attention_triton_fixed.py` - Earlier fixed version

These implementations are not suitable for training and have been removed to maintain code quality.