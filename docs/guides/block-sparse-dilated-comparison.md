# BlockSparseAttention vs BlockSparseDilatedAttention

This guide compares the two implementations to help you choose the right one for your use case.

## Overview

Both implementations apply **block-level sparsity** to reduce computation, but they differ in how attention is computed **within** each active block.

## BlockSparseAttention

```python
from dilated_attention_pytorch import BlockSparseAttention, SparsePatternConfig

model = BlockSparseAttention(
    sparse_config=SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.1,  # 90% sparse
        block_size=256,
    )
)
```

### How it works:
1. Divides sequence into blocks (e.g., 256 tokens each)
2. Determines which blocks can attend to each other (sparse pattern)
3. Within each active block pair: computes **full dense attention**

### Characteristics:
- **Sparsity**: Block-level only
- **Attention**: Dense within blocks
- **Complexity**: O(active_blocks × block_size²)
- **Memory**: O(active_blocks × block_size²)

## BlockSparseDilatedAttention

```python
from dilated_attention_pytorch import BlockSparseDilatedAttention, SparsePatternConfig

model = BlockSparseDilatedAttention(
    segment_lengths=[128, 128],  # Two segments per block
    dilation_rates=[1, 2],       # Second segment dilated by 2
    sparse_config=SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.1,  # 90% sparse at block level
        block_size=256,
    )
)
```

### How it works:
1. Divides sequence into blocks (same as BlockSparseAttention)
2. Determines which blocks can attend to each other (same sparse pattern)
3. Within each active block pair: applies **dilated attention** with segments

### Characteristics:
- **Sparsity**: Block-level + Token-level (via dilation)
- **Attention**: Dilated within blocks (multi-scale)
- **Complexity**: O(active_blocks × block_size × effective_size)
- **Memory**: Similar to BlockSparseAttention

## Key Differences

### 1. Attention Pattern Within Blocks

**BlockSparseAttention:**
```
Block (256 tokens):
[================================]
 Every token attends to every token
 Total: 256 × 256 = 65,536 interactions
```

**BlockSparseDilatedAttention:**
```
Block (256 tokens) with segments [128, 128] and dilations [1, 2]:
[================][================]
 Segment 1 (128)    Segment 2 (128)
 Dilation: 1        Dilation: 2
 (all tokens)       (every 2nd token)
 
 Effective interactions: ~49,152 (25% reduction)
```

### 2. Multi-Scale Modeling

**BlockSparseAttention:**
- Single scale of attention
- All tokens in a block are treated equally
- Good for uniform patterns

**BlockSparseDilatedAttention:**
- Multi-scale attention (different dilation rates)
- Near tokens: fine-grained attention (dilation 1)
- Far tokens: coarse-grained attention (dilation 2+)
- Better for hierarchical patterns

### 3. Computational Efficiency

For a sequence of 16,384 tokens with 256-token blocks and 90% sparsity:

**BlockSparseAttention:**
- 64×64 = 4,096 total blocks
- ~410 active blocks (10%)
- Each block: 256² = 65,536 operations
- Total: 26.8M operations

**BlockSparseDilatedAttention:**
- Same 410 active blocks
- Each block: ~49,152 operations (with dilation)
- Total: 20.1M operations
- **25% fewer operations**

### 4. Memory Access Patterns

**BlockSparseAttention:**
- Dense memory access within blocks
- Good cache utilization for small blocks
- May suffer cache misses for large blocks

**BlockSparseDilatedAttention:**
- Strided memory access due to dilation
- Better for very large blocks
- Slightly more complex access patterns

## Performance Comparison

| Aspect | BlockSparseAttention | BlockSparseDilatedAttention |
|--------|---------------------|---------------------------|
| Speed | Fastest | ~10-20% overhead |
| Memory | Baseline | Similar |
| FLOPs | Baseline | ~25% reduction |
| Multi-scale | No | Yes |
| Implementation | Simple | More complex |

## When to Use Which?

### Use BlockSparseAttention when:
- ✅ Maximum speed is critical
- ✅ Working with uniform data (images, structured data)
- ✅ All interactions within a region are equally important
- ✅ Simple implementation preferred
- ✅ Block sizes are relatively small (<512)

### Use BlockSparseDilatedAttention when:
- ✅ Working with hierarchical data (text, time series, audio)
- ✅ Need multi-scale pattern recognition
- ✅ Processing very long sequences (>16K tokens)
- ✅ Want both local detail and global context
- ✅ Block sizes are large (≥256)
- ✅ Can afford 10-20% runtime overhead for better modeling

## Example Use Cases

### BlockSparseAttention:
- Image generation models
- Protein structure prediction
- Dense sensor data processing
- Real-time applications

### BlockSparseDilatedAttention:
- Language models (capturing local + distant dependencies)
- Document understanding (word → sentence → paragraph)
- Time series forecasting (short + long-term patterns)
- Music generation (notes → measures → phrases)

## Code Example: Comparing Outputs

```python
import torch
from dilated_attention_pytorch import (
    BlockSparseAttention, 
    BlockSparseDilatedAttention,
    SparsePatternConfig
)

# Configuration
seq_len = 2048
batch_size = 2
num_heads = 8
head_dim = 64

# Both use same block-sparse pattern
sparse_config = SparsePatternConfig(
    pattern_type="local_window",
    sparsity_ratio=0.1,
    block_size=256,
)

# Create models
block_sparse = BlockSparseAttention(sparse_config=sparse_config)
block_sparse_dilated = BlockSparseDilatedAttention(
    segment_lengths=[128, 128],
    dilation_rates=[1, 2],
    sparse_config=sparse_config,
)

# Input
x = torch.randn(batch_size, seq_len, num_heads, head_dim)

# Forward pass
output1 = block_sparse(x, x, x)        # Dense within blocks
output2 = block_sparse_dilated(x, x, x)  # Dilated within blocks

print(f"BlockSparse output shape: {output1.shape}")
print(f"BlockSparseDilated output shape: {output2.shape}")
print(f"Outputs are different: {not torch.allclose(output1, output2)}")
```

## Summary

- **BlockSparseAttention**: Faster, simpler, single-scale
- **BlockSparseDilatedAttention**: Multi-scale, hierarchical, ~25% fewer FLOPs

Choose based on your data characteristics and performance requirements. For most applications, the standard BlockSparseAttention is sufficient. Use BlockSparseDilatedAttention when you need multi-scale modeling and can afford the slight overhead.