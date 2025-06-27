# Adaptive Attention Patterns Benchmark

Generated: 2025-06-27T23:42:47.046191Z

## Configuration

- Device: cuda
- Data type: torch.float32
- Sequence length: 2048
- Batch size: 1
- Num heads: 8
- Head dim: 64

## Pattern Adaptation Results

| Input Type | Connections | Sparsity |
|------------|-------------|----------|
| Random | 96 | 90.6% |
| Periodic | 96 | 90.6% |
| Sparse | 96 | 90.6% |

## Performance Results

| Implementation | Time (ms) | Memory (MB) | Speedup vs Baseline |
|----------------|-----------|-------------|--------------------|
| Dense Baseline | 1.63 | 37.26 | 1.00x |
| Fixed Local Window | 28.41 | 28.75 | 0.06x |
| Fixed Hierarchical | 18.46 | 232.13 | 0.09x |
| Adaptive (default) | 87.29 | 40.05 | 0.02x |
| Adaptive (optimized) | 34.94 | 32.19 | 0.05x |

## Key Findings

- Fastest: Dense Baseline (1.63 ms)
- Most memory efficient: Fixed Local Window (28.75 MB)
- Adaptive is 207.3% slower than fixed local window

### Adaptive Pattern Advantages:
- Content-aware sparsity adapts to input characteristics
- Learnable patterns can capture task-specific dependencies
- Flexible sparsity ratio based on sequence complexity
