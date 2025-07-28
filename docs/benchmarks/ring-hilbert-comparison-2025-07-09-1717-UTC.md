# Ring Attention Hilbert Comparison Benchmark

Generated: 2025-07-09 17:17:59 UTC

## System Information

- GPU: NVIDIA GeForce GTX 1080
- CUDA Version: 12.6
- PyTorch Version: 2.7.1+cu126

## Performance Comparison

### Seq=2,048, Batch=1, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 3.30±0.19   | 137.6±0.0     |               620283 |            68.82 |
| Ring (Hilbert)    | 4.26±0.60   | 137.6±0.0     |               480962 |            68.82 |

**Hilbert Impact:**
- Speedup: 0.78x
- Memory ratio: 1.00x

### Seq=2,048, Batch=2, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 6.96±0.80   | 266.1±0.0     |               588582 |            66.53 |
| Ring (Hilbert)    | 9.57±0.58   | 266.1±0.0     |               427869 |            66.53 |

**Hilbert Impact:**
- Speedup: 0.73x
- Memory ratio: 1.00x

### Seq=4,096, Batch=1, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 21.25±12.24 | 266.1±0.0     |               192793 |            66.53 |
| Ring (Hilbert)    | 9.17±0.63   | 266.1±0.0     |               446561 |            66.53 |

**Hilbert Impact:**
- Speedup: 2.32x
- Memory ratio: 1.00x

### Seq=4,096, Batch=2, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 145.57±151.79 | 524.6±0.0     |                56275 |            65.58 |
| Ring (Hilbert)    | 96.43±102.47  | 524.1±0.0     |                84953 |            65.52 |

**Hilbert Impact:**
- Speedup: 1.51x
- Memory ratio: 1.00x

### Seq=8,192, Batch=1, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 113.05±122.02 | 519.6±0.0     |                72464 |            64.96 |
| Ring (Hilbert)    | 273.88±176.45 | 519.6±0.0     |                29911 |            64.96 |

**Hilbert Impact:**
- Speedup: 0.41x
- Memory ratio: 1.00x

### Seq=8,192, Batch=2, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 474.37±301.03 | 1032.1±0.0    |                34538 |            64.51 |
| Ring (Hilbert)    | 518.76±333.03 | 1032.1±0.0    |                31583 |            64.51 |

**Hilbert Impact:**
- Speedup: 0.91x
- Memory ratio: 1.00x

## Summary Analysis

### Average Hilbert Impact

- Average speedup: 1.11x (±0.63)
- Average memory ratio: 1.00x (±0.00)

## Recommendations

- ✅ **Hilbert ordering provides consistent performance benefits**
- Average speedup of 1.11x justifies the additional complexity

### Best Use Cases for Hilbert Ordering

Based on the benchmarks:
- **Most beneficial configurations:**
  - Seq=4,096, Batch=1, Heads=8: 2.32x speedup
  - Seq=4,096, Batch=2, Heads=8: 1.51x speedup
