# Dilated Attention Implementation Overview

**Last Updated**: 2025-07-07

This guide provides a comprehensive overview of all available dilated attention implementations in this library.

## 📊 Implementation Count: 22 Distinct Implementations

The library provides 22 different implementations of dilated attention, each optimized for specific use cases. Recent consolidation has merged redundant implementations while preserving unique functionality.

## 🎯 Quick Selection Guide

```python
from dilated_attention_pytorch import create_multihead_dilated_attention

# Let the factory choose the best implementation
attention = create_multihead_dilated_attention("auto", 
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4]
)
```

### Selection Criteria:
- **Sequence Length < 50K**: Use standard implementations
- **Sequence Length 50K-1M**: Use Ring Attention (O(n) memory)
- **Need 5-50x speedup**: Use Block-Sparse variants
- **Multi-GPU**: Use distributed implementations
- **Memory constrained**: Use memory-optimized variants

## 📚 Complete Implementation List

### 1. Core Implementations

#### DilatedAttention
- **File**: `dilated_attention_pytorch/dilated_attention.py`
- **Use Case**: Basic dilated attention from LongNet paper
- **Features**: Configurable segment lengths and dilation rates
- **Memory**: O(n²/d) where d is average dilation rate

```python
from dilated_attention_pytorch import DilatedAttention

attn = DilatedAttention(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4]
)
```

#### ImprovedDilatedAttention
- **File**: `dilated_attention_pytorch/improved_dilated_attention.py`
- **Use Case**: When you need better numerical stability and performance
- **Features**: Enhanced gradient flow, better memory management
- **Memory**: Same as base but with pooling optimizations

#### MultiheadDilatedAttention
- **File**: `dilated_attention_pytorch/multihead_dilated_attention.py`
- **Use Case**: Drop-in replacement for `nn.MultiheadAttention`
- **Features**: Compatible with existing transformer code
- **Memory**: O(n²/d) per head

#### ImprovedMultiheadDilatedAttention
- **File**: `dilated_attention_pytorch/improved_multihead_dilated_attention.py`
- **Use Case**: Production deployments needing stability
- **Features**: Layer normalization, MAGNETO improvements
- **Memory**: Optimized with pattern caching

### 2. Ring Attention Variants (O(n) Memory)

#### RingDilatedAttentionHybrid
- **File**: `dilated_attention_pytorch/ring_dilated_attention_hybrid.py`
- **Aliases**: `RingDilatedAttention`, `RingDilatedAttentionTrue`
- **Use Case**: Sequences up to 1B tokens
- **Features**: True O(n/p) memory scaling, Flash Attention integration
- **Memory**: O(n/p) where p is number of GPUs

```python
from dilated_attention_pytorch import RingDilatedAttention

# Works with 1M+ token sequences
attn = RingDilatedAttention(
    segment_lengths=[4096, 8192, 16384],
    dilation_rates=[1, 2, 4],
    ring_size=8  # Number of GPUs/processes
)
```

#### [REMOVED] RingDilatedAttentionProduction
- **Removed**: July 2025 - Was not actually implementing ring attention
- **Issue**: Used O(n²) memory instead of O(n), defeating the purpose
- **Details**: See `docs/reports/ring-production-not-ring-attention-2025-07-08-0327-UTC.md`

#### RingMultiheadDilatedAttentionHybrid
- **File**: `dilated_attention_pytorch/ring_multihead_dilated_attention_hybrid.py`
- **Use Case**: Multihead attention with ring communication
- **Features**: Fused QKV projections, buffer reuse
- **Memory**: O(n/p) per head

### 3. Block-Sparse Variants (5-50x Speedup)

#### BlockSparseRingDilatedAttention (Enhanced)
- **File**: `dilated_attention_pytorch/block_sparse_ring_dilated_attention.py`
- **Use Case**: When you need extreme speedup
- **Features**: 
  - 90%+ sparsity, multiple pattern types
  - **NEW**: Device-aware pattern caching with LRU eviction
  - **NEW**: Batched block operations (32+ blocks)
  - **NEW**: Smart buffer reuse strategies
  - Flash Attention 3 integration
- **Memory**: O(n × sparsity_ratio)

```python
from dilated_attention_pytorch import BlockSparseRingDilatedAttention, SparsePatternConfig

config = SparsePatternConfig(
    pattern_type='dilated_sparse',  # or 'local_window', 'global_local'
    sparsity_ratio=0.1,  # 90% sparse = 10x speedup
    block_size=128
)

attn = BlockSparseRingDilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    sparse_config=config
)
```

#### BlockSparseRingMultiheadDilatedAttention
- **File**: `dilated_attention_pytorch/block_sparse_ring_multihead_dilated_attention.py`
- **Use Case**: Drop-in sparse replacement for transformers
- **Features**: Compatible with existing code
- **Memory**: O(n × sparsity_ratio) per head

#### BlockSparseRingDistributedDilatedAttention
- **File**: `dilated_attention_pytorch/block_sparse_ring_distributed_dilated_attention.py`
- **Use Case**: Multi-node training with sparsity
- **Features**: Hierarchical sparsity patterns
- **Memory**: O(n × sparsity_ratio / p)

### 4. Specialized Block-Sparse Variants

> **Note**: BlockSparseOptimized has been merged into BlockSparseRingDilatedAttention, and BlockSparseTorchSparse has been removed as it provided no benefits over the base implementation.

#### BlockSparseHierarchical
- **File**: `dilated_attention_pytorch/block_sparse_hierarchical.py`
- **Use Case**: Multi-scale attention patterns
- **Features**: Local + global + intermediate patterns
- **Memory**: Adaptive based on hierarchy

#### BlockSparseAdaptive
- **File**: `dilated_attention_pytorch/block_sparse_adaptive.py`
- **Use Case**: Content-aware sparsity
- **Features**: Learns optimal sparse patterns
- **Memory**: Dynamic based on content

### 5. Distributed Variants

#### DistributedMultiheadDilatedAttention
- **File**: `dilated_attention_pytorch/distributed_dilated_attention.py`
- **Use Case**: PyTorch Lightning distributed training
- **Features**: Automatic DDP handling
- **Memory**: Standard dilated attention per GPU

#### RingDistributedDilatedAttention
- **File**: `dilated_attention_pytorch/ring_distributed_dilated_attention.py`
- **Use Case**: Enterprise distributed deployments
- **Features**: DeepSpeed integration, advanced monitoring
- **Memory**: O(n/p) with enterprise features

### 6. Special Variants

#### BlockSparseRingDilatedAttentionHilbertPostPattern
- **File**: `dilated_attention_pytorch/block_sparse_ring_dilated_attention_hilbert_post_pattern.py`
- **Use Case**: Optimized processing order for better cache locality
- **Features**: Post-pattern Hilbert optimization (up to 2.53x speedup)
- **Memory**: Standard block-sparse with optimized access patterns

```python
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
    create_post_pattern_hilbert_attention
)

# Optimized for sequences ≥ 4K tokens
attn = create_post_pattern_hilbert_attention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    sparsity_ratio=0.1
)
```

#### HeadParallelDilatedAttentionOptimized
- **File**: `dilated_attention_pytorch/head_parallel_dilated_attention_optimized.py`
- **Use Case**: When heads can be processed independently
- **Features**: Parallel head processing
- **Memory**: Standard with parallel efficiency

### 7. Kernel Implementations

#### HilbertDilatedAttention
- **File**: `dilated_attention_pytorch/kernels/hilbert_dilated_attention.py`
- **Use Case**: Custom kernel for Hilbert ordering
- **Features**: Low-level optimizations
- **Memory**: Highly optimized

#### HilbertAttentionTritonFixed
- **File**: `dilated_attention_pytorch/kernels/hilbert_dilated_attention_triton_fixed.py`
- **Use Case**: Triton kernel implementation
- **Features**: GPU-specific optimizations
- **Memory**: Kernel-level efficiency

## 🔧 Configuration Options

All implementations support these core parameters:

```python
# Common parameters
segment_lengths = [2048, 4096, 8192]  # Must be increasing
dilation_rates = [1, 2, 4]            # Must match segment_lengths
dropout = 0.1                         # Attention dropout
attention_scale = None                # None for 1/sqrt(d)

# Multihead parameters
embed_dim = 768                       # Model dimension
num_heads = 12                        # Number of attention heads
bias = True                          # Use bias in projections
add_bias_kv = False                  # Add bias to keys/values
add_zero_attn = False                # Add zero attention

# Ring attention parameters
ring_size = 8                        # Number of processes
use_flash_attn = True               # Use Flash Attention if available
use_gradient_checkpointing = True   # Save memory during training

# Block-sparse parameters
sparsity_ratio = 0.1                # Fraction of blocks to compute
block_size = 128                    # Size of each block
pattern_type = 'dilated_sparse'     # Sparsity pattern
```

## 📈 Performance Comparison

| Implementation | Memory | Speed | Best For |
|----------------|--------|-------|----------|
| DilatedAttention | O(n²/d) | 1x | Small sequences (<10K) |
| ImprovedDilatedAttention | O(n²/d) | 1.2x | Better stability |
| RingDilatedAttention | O(n/p) | 0.8x | Very long sequences |
| BlockSparseRingDilatedAttention (Enhanced) | O(n×s) | 5-50x | Speed critical, now with better caching |
| BlockSparseHierarchical | O(n×s) | 3-20x | Multi-scale patterns |
| BlockSparseAdaptive | O(n×s) | Variable | Content-dependent sparsity |
| RingDilatedAttentionHilbertOptimized | O(n/p) | 1.5x | Cache efficiency |

## 🚀 Getting Started

1. **For most users**: Use the factory pattern
2. **For long sequences**: Use Ring variants
3. **For speed**: Use Block-Sparse variants
4. **For production**: Use Production/Improved variants
5. **For research**: Experiment with different variants

## 🔍 How to Choose

```python
def choose_implementation(seq_len, num_gpus=1, need_speed=False):
    if need_speed and seq_len > 10_000:
        return "block_sparse"
    elif seq_len > 100_000:
        return "ring"
    elif seq_len > 50_000 and num_gpus > 1:
        return "distributed"
    else:
        return "improved"

# Use with factory
impl_type = choose_implementation(seq_len=1_000_000, num_gpus=8)
attention = create_multihead_dilated_attention(impl_type, ...)
```

## 📚 Further Reading

- [Ring Attention Guide](ring-attention-guide.md)
- [Block-Sparse Attention Guide](block-sparse-attention-guide.md)
- [Distributed Training Guide](distributed-training-guide.md)
- [Factory Pattern Guide](factory-pattern-guide.md)
- [Migration Guide](migration-v0.3.0.md)