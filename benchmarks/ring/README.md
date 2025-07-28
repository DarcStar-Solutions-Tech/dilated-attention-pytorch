# Ring Attention Benchmarks

This directory contains benchmarks for ring attention implementations in the dilated-attention-pytorch library.

## Important Distinction

There are two types of "ring attention" implementations in this library:

### 1. Standard Ring Attention
- **What it does**: Computes FULL attention matrix, but splits computation across GPUs
- **Memory savings**: O(n/k) where k = number of GPUs (saves on attention matrix storage)
- **Computation**: Still O(n²) - computes all attention values
- **Examples**: `StandardRingAttention`, `HilbertRingAttention`, `DistributedRingAttention`

### 2. Dilated Ring Attention  
- **What it does**: Computes SPARSE dilated attention patterns AND uses ring communication
- **Memory savings**: O(n/k) from ring + additional savings from sparsity
- **Computation**: O(n×s) where s << n due to dilated/sparse patterns
- **Examples**: `RingDilatedAttentionSDPA`, `RingDilatedAttentionCorrect`

## Benchmark Scripts

### `dilated_vs_standard_ring_benchmark.py` (NEW)
Compares standard ring attention with dilated ring attention to show:
- Computational savings from dilated patterns
- Memory efficiency differences
- Throughput comparison
- Sparsity levels

Usage:
```bash
# Single GPU
python dilated_vs_standard_ring_benchmark.py

# Multi-GPU
torchrun --nproc_per_node=2 dilated_vs_standard_ring_benchmark.py
```

### `comprehensive_ring_benchmark.py`
Tests various standard ring attention implementations (NOT dilated).
Shows O(n/k) memory scaling across GPUs.

### `memory_scaling_analysis.py`
Analyzes memory usage patterns for ring attention implementations.

### `extreme_sequence_benchmark.py`
Tests with very long sequences (100K+ tokens).
Note: Must allocate full input tensors even with ring attention.

### `quick_multi_gpu_test.py`
Quick verification of multi-GPU setup and ring communication.

## Key Insights

1. **Ring attention alone** provides O(n/k) memory scaling by splitting sequences across k GPUs
2. **Dilated attention** provides computational savings by computing sparse patterns
3. **Dilated + Ring** combines both benefits: memory scaling AND computational efficiency
4. Standard ring attention still needs to compute the full O(n²) attention matrix (just distributed)

## Running Benchmarks

```bash
# Compare dilated vs standard (recommended first)
python dilated_vs_standard_ring_benchmark.py --seq-lengths 1024 2048 4096

# Test ring implementations  
torchrun --nproc_per_node=2 comprehensive_ring_benchmark.py

# Memory scaling analysis
torchrun --nproc_per_node=2 memory_scaling_analysis.py
```

## Understanding Results

When comparing implementations, consider:
- **Memory per GPU**: Should scale as O(n/k) with k GPUs
- **Computation time**: Dilated should be faster due to sparsity
- **Throughput**: Tokens/second processed
- **Sparsity**: Percentage of attention matrix NOT computed (dilated only)