# Comprehensive Block-Sparse Benchmark

Generated: 2025-06-27T23:07:54.033409Z

## Configuration

- Device: cuda
- Data type: torch.float16
- Sequence length: 4096
- Batch size: 1
- Num heads: 8
- Head dim: 64
- Default sparsity: 0.9
- Segment lengths: [1024, 2048, 4096]
- Dilation rates: [1, 2, 4]

## Implementation Comparison

| Implementation | Time (ms) | Memory (MB) | Speedup vs Original |
|----------------|-----------|-------------|--------------------|
| Baseline | 29.72 | 28.27 | 2.30x |
| Original | 68.36 | 28.44 | 1.00x |
| Optimized | 28.23 | 148.13 | 2.42x |
| TorchSparse | 540.95 | 29.22 | 0.13x |

## Sparsity Analysis

| Sparsity | Active Blocks | Time (ms) | Memory (MB) |
|----------|---------------|-----------|-------------|
| 0.80 | 20% | 612.37 | 29.22 |
| 0.90 | 10% | 637.34 | 29.22 |
| 0.95 | 5% | 594.29 | 29.22 |
| 0.98 | 2% | 660.75 | 29.22 |

## Key Findings

- Best performance: Optimized (28.23 ms)
- Worst performance: TorchSparse (540.95 ms)
- Optimized improvement over Original: 58.7%
- TorchSparse improvement over Original: -691.4%
