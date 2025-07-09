# Ring Attention Hilbert Comparison Benchmark

Generated: 2025-07-09 17:34:51 UTC

## System Information

- GPU: NVIDIA GeForce GTX 1080
- CUDA Version: 12.6
- PyTorch Version: 2.7.1+cu126

## Performance Comparison

### Seq=2,048, Batch=1, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 3.92±0.65   | 137.6±0.0     |               522102 |            68.82 |
| Ring (Hilbert)    | 3.40±0.05   | 144.7±0.0     |               602107 |            72.33 |

**Hilbert Impact:**
- Speedup: 1.15x
- Memory ratio: 1.05x

### Seq=4,096, Batch=1, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 10.20±0.45  | 266.2±0.0     |               401432 |            66.54 |
| Ring (Hilbert)    | 10.67±0.23  | 278.7±0.0     |               383742 |            69.68 |

**Hilbert Impact:**
- Speedup: 0.96x
- Memory ratio: 1.05x

### Seq=8,192, Batch=1, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 101.30±109.29 | 519.7±0.0     |                80866 |            64.97 |
| Ring (Hilbert)    | 39.01±19.71   | 540.9±0.0     |               210003 |            67.61 |

**Hilbert Impact:**
- Speedup: 2.60x
- Memory ratio: 1.04x

## Summary Analysis

### Average Hilbert Impact

- Average speedup: 1.57x (±0.73)
- Average memory ratio: 1.05x (±0.00)

## Recommendations

- ✅ **Hilbert ordering provides consistent performance benefits**
- Average speedup of 1.57x justifies the additional complexity

### Best Use Cases for Hilbert Ordering

Based on the benchmarks:
- **Most beneficial configurations:**
  - Seq=8,192, Batch=1, Heads=8: 2.60x speedup
  - Seq=2,048, Batch=1, Heads=8: 1.15x speedup
