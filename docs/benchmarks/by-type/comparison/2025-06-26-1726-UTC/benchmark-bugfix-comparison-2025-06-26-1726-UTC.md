# Benchmark Results - Bug Fix Comparison

Generated: 2025-06-26-1726-UTC

## Configuration

- Device: cuda
- Batch size: 2
- Sequence length: 8192
- Number of heads: 8
- Head dimension: 64
- Segment lengths: [2048, 4096]
- Dilation rates: [1, 2]

## Results

| Implementation | Time (seconds) | Std Dev |
|----------------|----------------|----------|
| DilatedAttention | 0.0105 | ±0.0010 |
| ImprovedDilatedAttention | 0.0103 | ±0.0005 |

**Speedup**: 1.02x

**Output difference (max)**: 0.000000

**Peak memory**: 280.0 MB

## Analysis

The bug fixes implemented in Phase 1.1 include:
1. Thread safety fix for cache access
2. Memory leak fix in buffer tracking
3. Ring size validation for distributed scenarios
4. Gradient normalization order correction

These fixes ensure correctness without significantly impacting performance.
