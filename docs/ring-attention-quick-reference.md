# Ring Attention Quick Reference Guide

## Overview

Ring Attention implementations are now properly exported at the top level for easy access. All implementations provide O(n/k) memory scaling where k is the number of GPUs.

## Available Implementations

### 1. **StandardRingAttention** - Clean Reference Implementation
```python
from dilated_attention_pytorch import StandardRingAttention

attention = StandardRingAttention(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.0,
    ring_size=world_size  # Number of GPUs
)
```
- Best for: Learning, reference, simple use cases
- Features: Clean implementation of ring attention algorithm

### 2. **DistributedRingAttention** - Enterprise Features
```python
from dilated_attention_pytorch import DistributedRingAttention

attention = DistributedRingAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.0,
    ring_size=world_size,
    use_deepspeed=True,
    enable_monitoring=True
)
```
- Best for: Production deployments, large-scale training
- Features: DeepSpeed, fault tolerance, monitoring, checkpointing

### 3. **HilbertRingAttention** - Cache-Optimized
```python
from dilated_attention_pytorch import HilbertRingAttention

attention = HilbertRingAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.0,
    ring_size=world_size,
    hilbert_level=8
)
```
- Best for: Maximum performance, cache efficiency
- Features: Hilbert curve reordering, GPU optimization

### 4. **RingBlockSparseAttention** - Sparse + Ring
```python
from dilated_attention_pytorch import RingBlockSparseAttention

attention = RingBlockSparseAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    sparsity_ratio=0.9,  # 90% sparse
    pattern_type="dilated_sparse"
)
```
- Best for: Extremely long sequences, maximum speedup
- Features: O(n/k) memory + 5-50x additional speedup from sparsity

### 5. **RingDilatedAttentionCorrect** - Reference Implementation
```python
from dilated_attention_pytorch import RingDilatedAttentionCorrect

attention = RingDilatedAttentionCorrect(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.0,
    ring_size=world_size
)
```
- Best for: Understanding correct implementation
- Features: Splits sequences BEFORE projection (critical for memory savings)

### 6. **RingDilatedAttentionSDPA** - PyTorch SDPA Backend
```python
from dilated_attention_pytorch import RingDilatedAttentionSDPA

attention = RingDilatedAttentionSDPA(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.0,
    ring_size=world_size
)
```
- Best for: Modern PyTorch versions, automatic optimization
- Features: Uses torch.nn.functional.scaled_dot_product_attention

## Factory Pattern (Recommended)

```python
from dilated_attention_pytorch import create_ring_attention, RingAttentionConfig

# Create with preset
attention = create_ring_attention("standard", world_size=4)

# Create with custom config
config = RingAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1,
    ring_size=world_size
)
attention = create_ring_attention("distributed", config=config)
```

## Memory Scaling Example

With 4 GPUs processing 100K tokens:
- **Without Ring**: Each GPU uses 1GB (4GB total)
- **With Ring**: Each GPU uses 250MB (1GB total)

## Multi-GPU Usage

Always use `torchrun` for multi-GPU execution:
```bash
# 2 GPUs
torchrun --nproc_per_node=2 your_script.py

# 4 GPUs
torchrun --nproc_per_node=4 your_script.py
```

## Performance Tips

1. **Sequence Length**: Must be divisible by largest segment length
2. **Ring Size**: Set to world_size for distributed training
3. **Memory**: Enable gradient checkpointing for very long sequences
4. **Communication**: Use NCCL backend for best performance

## Migration from Old Names

- `RingDilatedAttentionProduction` → Use `StandardRingAttention`
- `RingDistributedDilatedAttention` → Use `EnterpriseDistributedDilatedAttention` (not ring!)
- `RingDilatedAttentionHilbertGPUOptimized` → Still available, or use `HilbertRingAttention`

## Which Implementation to Choose?

- **New project**: Use `StandardRingAttention` or factory pattern
- **Production**: Use `DistributedRingAttention` for enterprise features
- **Maximum performance**: Use `HilbertRingAttention` or `RingBlockSparseAttention`
- **Learning/debugging**: Use `RingDilatedAttentionCorrect` to understand the algorithm