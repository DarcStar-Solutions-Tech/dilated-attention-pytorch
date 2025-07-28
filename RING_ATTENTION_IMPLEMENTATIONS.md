# Complete List of Ring Attention Implementations

## Overview

This document provides a comprehensive list of all ring attention implementations in the dilated-attention-pytorch project, organized by category and purpose.

## Main Ring Attention Implementations

### 1. **StandardRingAttention** (`ring/standard_ring_attention.py`)
- **Location**: `src/dilated_attention_pytorch/ring/standard_ring_attention.py`
- **Description**: The basic ring attention implementation that provides O(n/k) memory scaling through proper sequence splitting and ring communication
- **Features**:
  - Standard ring communication pattern using isend/irecv
  - Proper LSE accumulation for numerical stability
  - Memory-efficient processing of long sequences
- **Aliases**: `RingDilatedAttention`, `RingDilatedAttentionProduction`

### 2. **HilbertRingAttention** (`ring/hilbert_ring_attention.py`)
- **Location**: `src/dilated_attention_pytorch/ring/hilbert_ring_attention.py`
- **Description**: Ring attention with Hilbert curve optimization for improved cache locality
- **Features**:
  - Per-segment Hilbert curve reordering
  - Maintains O(n/k) memory benefits
  - Improved cache efficiency for better performance

### 3. **DistributedRingAttention** (`ring/distributed_ring_attention.py`)
- **Location**: `src/dilated_attention_pytorch/ring/distributed_ring_attention.py`
- **Description**: Enterprise-grade distributed ring attention with advanced features
- **Features**:
  - DeepSpeed integration for ZeRO optimization
  - Fault tolerance with automatic recovery
  - Advanced monitoring and profiling
  - Gradient compression and optimization

### 4. **BlockSparseRingAttention** (`ring/block_sparse_ring_attention.py`)
- **Location**: `src/dilated_attention_pytorch/ring/block_sparse_ring_attention.py`
- **Description**: Ring attention combined with block-sparse patterns
- **Features**:
  - O(n/k) memory complexity from ring attention
  - Additional speedup from block-sparse patterns
  - Multiple sparse pattern types (local, dilated, global-local)

## Legacy/Specialized Ring Implementations

### 5. **RingDilatedAttentionHilbertGPUOptimized** (`ring/hilbert/ring_dilated_attention_hilbert_gpu_optimized.py`)
- **Location**: `src/dilated_attention_pytorch/ring/hilbert/ring_dilated_attention_hilbert_gpu_optimized.py`
- **Description**: GPU-optimized ring attention with per-segment Hilbert optimization
- **Features**:
  - Automatic GPU detection and backend selection
  - Optimized for specific GPU architectures
  - Per-segment Hilbert curve optimization
- **Status**: Legacy implementation, use `HilbertRingAttention` for new code

### 6. **RingDilatedAttentionCorrect** (`ring/base/ring_dilated_attention_correct.py`)
- **Location**: `src/dilated_attention_pytorch/ring/base/ring_dilated_attention_correct.py`
- **Description**: Reference implementation with correctness focus
- **Features**:
  - Emphasis on numerical correctness
  - Includes RingAttentionWrapper for compatibility
  - Used as baseline for testing
- **Status**: Legacy implementation, mainly for reference

### 7. **RingDilatedAttentionSDPA** (`ring/base/ring_dilated_attention_sdpa.py`)
- **Location**: `src/dilated_attention_pytorch/ring/base/ring_dilated_attention_sdpa.py`
- **Description**: Ring attention using PyTorch's scaled_dot_product_attention
- **Features**:
  - Leverages PyTorch's optimized SDPA backend
  - Automatic backend selection (Flash Attention, Math, etc.)
  - Simplified implementation
- **Status**: Legacy implementation

## Distributed Ring Implementations

### 8. **EnterpriseDistributedDilatedAttention** (`ring/distributed/ring_distributed_dilated_attention.py`)
- **Location**: `src/dilated_attention_pytorch/ring/distributed/ring_distributed_dilated_attention.py`
- **Description**: Base class for enterprise distributed features
- **Features**:
  - Comprehensive error recovery mechanisms
  - Memory pool management
  - Gradient optimization and compression
  - Multi-level fault tolerance

### 9. **RingDistributedDilatedAttention** (`ring/distributed/ring_distributed_dilated_attention.py`)
- **Location**: `src/dilated_attention_pytorch/ring/distributed/ring_distributed_dilated_attention.py`
- **Description**: Full-featured distributed ring attention (inherits from EnterpriseDistributedDilatedAttention)
- **Features**:
  - All enterprise features
  - Backward compatibility wrapper
  - Production-ready for large-scale training
- **Note**: This is different from `DistributedRingAttention` in the main ring module

## Block-Sparse Ring Variants

### 10. **BlockSparseRingDilatedAttention** (`sparse/block_sparse_ring_dilated_attention.py`)
- **Location**: `src/dilated_attention_pytorch/sparse/block_sparse_ring_dilated_attention.py`
- **Description**: Enhanced block-sparse ring dilated attention with merged optimizations
- **Features**:
  - Memory-efficient block-sparse attention patterns
  - Multiple sparse pattern types
  - Optimized for long sequences
  - Not refactored by design for performance

### 11. **BlockSparseRingDilatedAttentionFixed** (`sparse/block_sparse_ring_dilated_attention_fixed.py`)
- **Location**: `src/dilated_attention_pytorch/sparse/block_sparse_ring_dilated_attention_fixed.py`
- **Description**: Standardized API wrapper for BlockSparseRingDilatedAttention
- **Features**:
  - Consistent API across implementations
  - Maintains efficient block-sparse implementation
  - Drop-in replacement compatibility

### 12. **BlockSparseRingDilatedAttentionHilbertPostPattern** (`sparse/block_sparse_ring_dilated_attention_hilbert_post_pattern.py`)
- **Location**: `src/dilated_attention_pytorch/sparse/block_sparse_ring_dilated_attention_hilbert_post_pattern.py`
- **Description**: Block-sparse ring attention with Hilbert curve optimization applied after pattern generation
- **Features**:
  - Up to 2.53x speedup from Hilbert optimization
  - Post-pattern Hilbert reordering
  - Optimized block processing order

### 13. **BlockSparseRingMultiheadDilatedAttention** (`sparse/block_sparse_ring_multihead_dilated_attention.py`)
- **Location**: `src/dilated_attention_pytorch/sparse/block_sparse_ring_multihead_dilated_attention.py`
- **Description**: Multihead version of block-sparse ring attention
- **Features**:
  - Drop-in replacement for nn.MultiheadAttention
  - Block-sparse optimization
  - Memory-efficient for long sequences

### 14. **BlockSparseRingDistributedDilatedAttention** (`sparse/block_sparse_ring_distributed_dilated_attention.py`)
- **Location**: `src/dilated_attention_pytorch/sparse/block_sparse_ring_distributed_dilated_attention.py`
- **Description**: Enterprise-grade block-sparse ring distributed attention
- **Features**:
  - Inherits from RingDistributedDilatedAttention
  - Combines block-sparse patterns with distributed training
  - Hierarchical sparsity patterns
  - Maximum scalability for large models

## Supporting Classes and Utilities

### Base Classes
- **BaseRingAttention** (`ring/base/base_ring_attention.py`): Abstract base class for all ring implementations
- **RingCommunicationMixin** (`ring/base/ring_communication_mixin.py`): Mixin providing ring communication patterns
- **RingCommunicationMixin** (`ring/utils/ring_communication_mixin.py`): Extended version with more features

### Configuration Classes
- **RingAttentionConfig** (`ring/base/ring_config.py`): Configuration for ring attention
- **RingAttentionConfig** (`core/config.py`): Extended configuration in core module
- **StandardizedRingConfig** (`core/standardized_api.py`): Standardized configuration for all implementations

### Utility Classes
- **RingAttentionFunction** (`ring/utils/ring_attention_autograd.py`): Custom autograd function for ring attention
- **StableRingAccumulator** (`ring/utils/ring_attention_lse.py`): Numerically stable accumulator for LSE
- **AsyncRingCommunicator** (`ring/utils/ring_communication_mixin.py`): Helper for async ring communications
- **RingAttentionState** (`ring/base/base_ring_attention.py`): State management for ring attention
- **RingCommunicationStats** (`ring/base/ring_config.py`): Statistics tracking for ring communication

### Mixin Classes
- **HilbertAttentionMixin** (`utils/hilbert_attention_mixin.py`): Mixin for Hilbert curve functionality
- **StandardizedRingAttentionMixin** (`core/standardized_api.py`): Mixin for standardized API support

## Factory Functions

The recommended way to create ring attention instances is through factory functions:

```python
from dilated_attention_pytorch.ring import create_ring_attention

# Auto-select best implementation
attention = create_ring_attention("auto", config)

# Specific implementations
standard = create_ring_attention("standard", config)
hilbert = create_ring_attention("hilbert", config)
distributed = create_ring_attention("distributed", config)
block_sparse = create_ring_attention("block_sparse", config)
```

## Removed Implementations

The following implementations were removed during cleanup (July 2025) due to poor performance or deprecated APIs:
- `ring_dilated_attention_v2_collective.py` - Used inefficient all_gather
- `ring_dilated_attention_refactored.py` - Merged into Production version
- `ring_hilbert_dilated_attention.py` - Functionality in HilbertOptimizedFixed
- `ring_dilated_attention_fixed.py` - Replaced by ProductionFixed
- `ring_multihead_dilated_attention.py` - Depended on deprecated V2Collective

## Summary

**Total Active Ring Attention Implementations**: 14
- 4 main standardized implementations (Standard, Hilbert, Distributed, BlockSparse)
- 4 legacy/specialized ring implementations
- 5 block-sparse ring variants
- 1 hybrid implementation (backup file)

**Key Characteristics**:
- All use O(n/k) memory scaling where k = world_size
- All use isend/irecv for efficient ring communication (no all_gather)
- Support for various optimizations (Hilbert curves, block-sparse, distributed)
- Enterprise features available in distributed variants
