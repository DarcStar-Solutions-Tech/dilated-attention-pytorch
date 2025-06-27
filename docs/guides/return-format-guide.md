# Return Format Guide for Multihead Attention Implementations

## Overview

Different multihead dilated attention implementations in this library have varying return formats. This guide explains the behavior of each implementation and how to ensure consistent returns across all variants.

## Implementation Return Behaviors

### 1. MultiheadDilatedAttention
- **Default**: Returns tensor when `need_weights=False`, tuple when `need_weights=True`
- **Configurable**: Set `module._always_return_tuple = True` to always return tuple

### 2. ImprovedMultiheadDilatedAttention
- **Default**: Always returns tuple `(output, None)`
- **Note**: Ignores `need_weights` parameter for return format

### 3. RingMultiheadDilatedAttention
- **Default**: Always returns tuple `(output, None)`
- **Note**: Ring attention cannot compute attention weights

### 4. BlockSparseRingMultiheadDilatedAttention
- **Default**: Always returns tuple `(output, attention_weights)`
- **Note**: Only implementation that can return actual attention weights

### 5. DistributedImprovedMultiheadDilatedAttention
- **Default**: Returns tensor when `need_weights=False`, tuple when `need_weights=True`
- **Configurable**: Set `module._always_return_tuple = True` to always return tuple

## Ensuring Consistent Returns

### Option 1: Use Return Standardizer Utility

```python
from dilated_attention_pytorch.utils.return_standardizer import standardize_attention_output

# Call any attention module
output = attention_module(q, k, v, need_weights=False)

# Standardize the output
output, weights = standardize_attention_output(output, force_tuple=True)
```

### Option 2: Use MultiheadAttentionWrapper

```python
from dilated_attention_pytorch.utils.return_standardizer import MultiheadAttentionWrapper

# Wrap any attention module
wrapped_attention = MultiheadAttentionWrapper(
    attention_module,
    always_return_tuple=True
)

# Always returns tuple
output, weights = wrapped_attention(q, k, v)
```

### Option 3: Configure Module Directly

```python
# For MultiheadDilatedAttention and DistributedImprovedMultiheadDilatedAttention
attention._always_return_tuple = True

# Now always returns tuple
output, weights = attention(q, k, v, need_weights=False)
```

## Best Practices

1. **For Drop-in Replacement**: Use `MultiheadAttentionWrapper` to ensure your module behaves exactly like `nn.MultiheadAttention`

2. **For New Code**: Always expect tuple returns and unpack accordingly:
   ```python
   output, _ = attention(q, k, v)
   ```

3. **For Flexibility**: Use the standardizer utility when you need to handle multiple implementations:
   ```python
   def process_attention(module, q, k, v):
       result = module(q, k, v)
       output, weights = standardize_attention_output(result, force_tuple=True)
       return output
   ```

## Attention Weights Support

**Important**: Due to the nature of dilated and ring attention patterns, most implementations cannot compute traditional attention weights:

- ✅ **BlockSparseRingMultiheadDilatedAttention**: Can return sparse attention weights
- ❌ **All other implementations**: Return `None` for weights

If you need attention weights for visualization or analysis, use `BlockSparseRingMultiheadDilatedAttention` or consider using hooks to capture intermediate activations.

## Migration Guide

If you're migrating from `nn.MultiheadAttention`:

```python
# Old code with nn.MultiheadAttention
attn = nn.MultiheadAttention(embed_dim, num_heads)
output, weights = attn(q, k, v)

# New code with dilated attention
from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention
attn = ImprovedMultiheadDilatedAttention(
    embed_dim, num_heads,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
output, weights = attn(q, k, v)  # weights will be None
```

No code changes needed - the interface is compatible!