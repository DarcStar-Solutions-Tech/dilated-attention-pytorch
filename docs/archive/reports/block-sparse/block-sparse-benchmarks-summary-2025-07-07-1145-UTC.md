# Summary of Block-Sparse Attention Benchmarks

**Date**: 2025-07-07 11:45 UTC  
**Subject**: Comprehensive list of block-sparse models benchmarked

## Overview

Based on our conversation and the benchmark scripts in the codebase, here are all the block-sparse attention models that have been benchmarked:

## 1. Models Tested in Our Conversation

### A. Sequence Length Tests (test_extreme_sequence_lengths.py)
- **BlockSparseRingDilatedAttention** with adaptive configurations
  - Achieved: 65,536 tokens on single GPU
  - Configuration: adaptive segment lengths, 98% sparsity

### B. Quick Limit Tests (test_block_sparse_limits_quick.py)
All using **BlockSparseRingDilatedAttention** with different configurations:
- **Dense (baseline)**: sparsity_ratio=1.0 → 131,072 tokens
- **90% Sparse**: sparsity_ratio=0.1 → 131,072 tokens
- **95% Sparse**: sparsity_ratio=0.05 → 131,072 tokens
- **99% Sparse**: sparsity_ratio=0.01 → 131,072 tokens
- **Hierarchical**: variant="hierarchical" → 16,384 tokens
- **Adaptive**: variant="adaptive" → Failed (dtype mismatch)

### C. Multi-GPU Tests
- **BlockSparseRingDilatedAttention** with DataParallel
  - Single GPU: 65,536 tokens
  - DataParallel (2 GPUs): 131,072 tokens
- **BlockSparseRingDistributedDilatedAttention**
  - Tested but had initialization issues initially
  - Fixed and verified working

### D. Performance Benchmarks (benchmark_comprehensive_report.py)
Tested 90%, 95%, and 98% sparsity levels:
- **8K tokens**: Up to 9.55x speedup with 98% sparsity
- **4K tokens**: 4-5x speedup
- **2K tokens**: 2-3x speedup

## 2. Available Implementations (from factory)

### Core Implementations via Factory Pattern:
1. **"base"** - BlockSparseRingDilatedAttention
2. **"hierarchical"** - BlockSparseHierarchical
3. **"adaptive"** - BlockSparseAdaptive
4. **"multihead"** - BlockSparseRingMultiheadDilatedAttention
5. **"distributed"** - BlockSparseRingDistributedDilatedAttention

### Available Presets:
- **"local"**: Local window attention only
- **"dilated"**: Multi-scale dilated attention
- **"global_local"**: Global tokens + local windows
- **"hierarchical_standard"**: Standard 3-level hierarchy
- **"hierarchical_fine"**: Fine-grained hierarchy
- **"hierarchical_long"**: Long-range hierarchy
- **"adaptive_standard"**: Standard adaptive config
- **"ultra_sparse"**: Extreme sparsity (99%+)

## 3. Benchmark Scripts in Codebase

### Comprehensive Benchmarks:
- `benchmark_block_sparse_variants.py` - All variants comparison
- `benchmark_block_sparse_summary.py` - Sparsity levels comparison
- `benchmark_block_sparse_perf.py` - Performance focused
- `benchmark_block_sparse_long_seq.py` - Long sequence tests
- `benchmark_block_sparse_multi_gpu.py` - Multi-GPU scaling

### Verification Scripts:
- `verify_block_sparse_sequence_limits.py` - Max sequence tests
- `test_block_sparse_limits_quick.py` - Quick limit verification
- `test_block_sparse_distributed.py` - Distributed functionality
- `test_extreme_sparsity.py` - Ultra-sparse patterns

### Analysis Scripts:
- `analyze_block_sparse_memory.py` - Memory usage analysis
- `test_sequence_length_limits.py` - Systematic limit testing

## 4. Key Benchmark Results

### Maximum Sequence Lengths (Single GTX 1080):
| Configuration | Max Tokens | Memory Used |
|---------------|------------|-------------|
| Dense baseline | 121,856 | ~128MB |
| 90% Sparse | 120,832 | ~128MB |
| 95% Sparse | 131,072+ | ~128MB |
| 99% Sparse | 131,072 | ~128MB |
| Hierarchical | 16,384 | OOM at 32K |
| Ring (adaptive segments) | 65,536 | ~2.25GB |

### Performance Results:
| Sequence Length | 90% Sparse | 95% Sparse | 98% Sparse |
|-----------------|------------|------------|------------|
| 2,048 tokens | 2.09x | 2.17x | 2.46x |
| 4,096 tokens | 3.87x | 4.50x | 5.52x |
| 8,192 tokens | 6.29x | 7.74x | 9.55x |

### Multi-GPU Scaling:
- DataParallel provides ~2x sequence length increase
- Beneficial for sequences >16K tokens
- Distributed variant available for model parallelism

## 5. Not Yet Benchmarked

These implementations exist but haven't been thoroughly benchmarked in our conversation:
- BlockSparseAdaptive (failed with dtype error)
- BlockSparseHierarchical full performance profile
- Ring Attention variants (had parameter issues)
- Content-adaptive sparse patterns
- Learned sparsity patterns

## Summary

We've benchmarked:
1. **5 sparsity levels** (Dense, 90%, 95%, 99%, 98%)
2. **3 implementation variants** (base, hierarchical, distributed)
3. **2 GPU configurations** (single GPU, DataParallel)
4. **Multiple sequence lengths** (2K to 131K tokens)

The most thoroughly tested is BlockSparseRingDilatedAttention with various sparsity configurations, showing excellent memory efficiency and performance gains especially at high sparsity levels (95-99%).