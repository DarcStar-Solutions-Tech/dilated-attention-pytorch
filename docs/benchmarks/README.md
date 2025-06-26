# Benchmark Results and Performance Analysis

This directory contains benchmark results, performance measurements, and optimization findings for the dilated-attention-pytorch project.

## Contents

### Benchmark Results
- `benchmark-results-2025-06-26-1136-UTC.md` - Standard benchmark results
- `benchmark-results-comprehensive-2025-06-26-1136-UTC.md` - Comprehensive performance analysis
- `billion-token-benchmark-results-2025-06-26-1136-UTC.md` - Extreme scale benchmarks (1B tokens)

### Performance Visualizations
- `benchmark.png` - Main benchmark visualization
- `benchmark 64M Tokens.png` - 64M token benchmark results
- `benchmark-128M-tokens-2025-06-26-1136-UTC.png` - 128M token benchmark
- `benchmark-256M-tokens-2025-06-26-1136-UTC.png` - 256M token benchmark
- `benchmark-288M-tokens-2025-06-26-1136-UTC.png` - 288M token benchmark

### Optimization Studies
- `block-sparse-optimization-findings.md` - Block-sparse attention optimization results
- `advanced-ring-attention-optimizations.md` - Ring attention performance optimizations

## Key Performance Metrics

### Speed Improvements
- **Block-Sparse Attention**: 5-50x speedup over dense attention
- **Ring Attention**: O(n) memory complexity vs O(n²) for standard attention
- **Flash Attention Integration**: 2-4x speedup on compatible hardware

### Memory Efficiency
- **Standard Attention**: O(n²) memory usage
- **Ring Attention**: O(n) memory usage
- **Block-Sparse**: 75-95% memory reduction

### Scalability
- Successfully tested up to 1 billion tokens
- Linear scaling with sequence length for Ring Attention
- Sub-linear scaling for Block-Sparse implementations

## Running Benchmarks

To reproduce these benchmarks, use the scripts in the `benchmarks/` directory at the project root:

```bash
# Basic benchmark
python benchmarks/benchmark.py

# Comprehensive benchmark
python benchmarks/benchmark_all.py

# Billion-token benchmark
python benchmarks/benchmark_ring_billion_tokens.py
```

## Hardware Tested

- NVIDIA A100 (80GB)
- NVIDIA H100 (80GB)
- NVIDIA V100 (32GB)
- NVIDIA RTX 4090 (24GB)
- NVIDIA GTX 1080 (8GB)