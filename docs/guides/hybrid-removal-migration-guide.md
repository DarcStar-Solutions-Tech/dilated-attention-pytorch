# Migration Guide: Hybrid Implementation Removal

## Overview

As of version 0.3.0, the Hybrid Ring Attention implementations have been removed due to poor performance compared to alternatives. This guide helps you migrate to better-performing alternatives.

## Removed Components

The following components have been removed:
- `RingDilatedAttentionHybrid`
- `RingDilatedAttentionTrue` (alias for Hybrid)
- `RingMultiheadDilatedAttentionHybrid`
- `create_ring_multihead_attention_hybrid`
- `RingDilatedAttentionHilbertOptimized` (depended on Hybrid)

## Performance Comparison

Benchmarks showed the Hybrid implementation was significantly slower:
- At 4096 tokens: 61,261 tokens/sec (Hybrid) vs 565,618 tokens/sec (Hilbert)
- At 8192 tokens: 80,491 tokens/sec (Hybrid) vs 382,531 tokens/sec (Hilbert)

## Migration Options

### Option 1: Use Hilbert-Optimized Implementation (Recommended)

For best performance with good memory efficiency:

```python
# Old code
from dilated_attention_pytorch import RingDilatedAttentionHybrid
model = RingDilatedAttentionHybrid(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)

# New code - using standardized API
from dilated_attention_pytorch.core import create_standardized_ring_attention
model = create_standardized_ring_attention(
    "hilbert",
    dim=64,
    heads=8,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)
```

### Option 2: Use Production Implementation

For well-tested, stable performance:

```python
# New code
from dilated_attention_pytorch import RingDilatedAttentionProduction
model = RingDilatedAttentionProduction(
    RingAttentionConfig(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        dropout=0.1
    )
)
```

### Option 3: Use Block-Sparse for Memory Savings

If memory is a primary concern:

```python
# New code
from dilated_attention_pytorch import BlockSparseRingDilatedAttention
model = BlockSparseRingDilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    sparsity_ratio=0.1,  # 90% sparse
    dropout=0.1
)
```

## Fixed Implementations Available

If you need the standardized API wrappers:

```python
# Import fixed implementations
from dilated_attention_pytorch.ring_dilated_attention_production_fixed import (
    RingDilatedAttentionProductionFixed
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_fixed import (
    BlockSparseRingDilatedAttentionFixed
)

# Use with StandardizedRingConfig
from dilated_attention_pytorch.core.standardized_api import StandardizedRingConfig

config = StandardizedRingConfig(
    dim=64,
    heads=8,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)

# Create model
model = RingDilatedAttentionHilbertOptimizedFixed(config)
```

## Key Improvements in Alternatives

1. **Hilbert Implementation**:
   - 5-9x faster than Hybrid
   - Better cache locality through Hilbert curve ordering
   - Efficient memory usage

2. **Production Implementation**:
   - Stable and well-tested
   - Good performance across all sequence lengths
   - Comprehensive error recovery

3. **Block-Sparse Implementation**:
   - 60-80% memory reduction
   - Good for very long sequences
   - Trade compute for memory

## Conclusion

The removal of Hybrid implementations simplifies the codebase while directing users to better-performing alternatives. The Hilbert-optimized implementation provides the best overall performance and should be the default choice for most use cases.