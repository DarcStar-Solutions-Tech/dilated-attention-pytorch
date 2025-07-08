# Ring Attention Hilbert Comparison Benchmark

Generated: 2025-07-08 20:26:24 UTC

## System Information

- GPU: NVIDIA GeForce GTX 1080
- CUDA Version: 12.6
- PyTorch Version: 2.7.1+cu126

## Performance Comparison

### Seq=2,048, Batch=1, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 4.33±0.28   | 105.8±0.9     |               473272 |            52.91 |
| Ring (Hilbert)    | 4.77±0.99   | 105.8±0.9     |               429065 |            52.91 |

**Hilbert Impact:**
- Speedup: 0.91x
- Memory ratio: 1.00x

### Seq=2,048, Batch=1, Heads=16

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 11.84±1.06  | 225.8±0.9     |               172983 |           112.91 |
| Ring (Hilbert)    | 61.99±72.87 | 225.8±0.9     |                33038 |           112.91 |

**Hilbert Impact:**
- Speedup: 0.19x
- Memory ratio: 1.00x

### Seq=2,048, Batch=2, Heads=8

| Implementation    | Time (ms)   | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 76.01±79.24 | 203.5±1.8     |                53885 |            50.88 |
| Ring (Hilbert)    | 68.36±79.54 | 203.5±1.8     |                59918 |            50.88 |

**Hilbert Impact:**
- Speedup: 1.11x
- Memory ratio: 1.00x

### Seq=2,048, Batch=2, Heads=16

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 198.46±170.10 | 443.5±1.8     |                20639 |           110.88 |
| Ring (Hilbert)    | 180.49±171.10 | 443.5±1.8     |                22693 |           110.88 |

**Hilbert Impact:**
- Speedup: 1.10x
- Memory ratio: 1.00x

### Seq=4,096, Batch=1, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 84.08±114.79  | 205.8±1.8     |                48717 |            51.45 |
| Ring (Hilbert)    | 130.06±128.69 | 205.8±1.8     |                31492 |            51.45 |

**Hilbert Impact:**
- Speedup: 0.65x
- Memory ratio: 1.00x

### Seq=4,096, Batch=1, Heads=16

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 340.74±269.25 | 446.4±1.8     |                12021 |           111.59 |
| Ring (Hilbert)    | 347.77±302.42 | 446.4±1.8     |                11778 |           111.59 |

**Hilbert Impact:**
- Speedup: 0.98x
- Memory ratio: 1.00x

### Seq=4,096, Batch=2, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 313.13±245.88 | 404.4±3.6     |                26162 |            50.56 |
| Ring (Hilbert)    | 249.73±261.16 | 404.4±3.6     |                32804 |            50.56 |

**Hilbert Impact:**
- Speedup: 1.25x
- Memory ratio: 1.00x

### Seq=4,096, Batch=2, Heads=16

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 711.65±507.57 | 884.6±3.6     |                11511 |           110.57 |
| Ring (Hilbert)    | 687.84±575.63 | 884.6±3.6     |                11910 |           110.57 |

**Hilbert Impact:**
- Speedup: 1.03x
- Memory ratio: 1.00x

### Seq=8,192, Batch=1, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 327.89±361.17 | 402.7±3.6     |                24984 |            50.34 |
| Ring (Hilbert)    | 350.63±333.67 | 402.7±3.6     |                23364 |            50.34 |

**Hilbert Impact:**
- Speedup: 0.94x
- Memory ratio: 1.00x

### Seq=8,192, Batch=1, Heads=16

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 979.79±662.36 | 883.6±3.6     |                 8361 |           110.45 |
| Ring (Hilbert)    | 717.34±662.87 | 883.6±3.6     |                11420 |           110.45 |

**Hilbert Impact:**
- Speedup: 1.37x
- Memory ratio: 1.00x

### Seq=8,192, Batch=2, Heads=8

| Implementation    | Time (ms)     | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|---------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 484.16±539.69 | 797.7±7.2     |                33840 |            49.86 |
| Ring (Hilbert)    | 994.27±624.79 | 797.7±7.2     |                16478 |            49.86 |

**Hilbert Impact:**
- Speedup: 0.49x
- Memory ratio: 1.00x

### Seq=8,192, Batch=2, Heads=16

| Implementation    | Time (ms)       | Memory (MB)   |   Throughput (tok/s) |   Mem/Token (KB) |
|-------------------|-----------------|---------------|----------------------|------------------|
| Ring (No Hilbert) | 1667.75±1054.55 | 1759.1±7.2    |                 9824 |           109.94 |
| Ring (Hilbert)    | 1895.49±1065.65 | 1759.1±7.2    |                 8644 |           109.94 |

**Hilbert Impact:**
- Speedup: 0.88x
- Memory ratio: 1.00x

## Summary Analysis

### Average Hilbert Impact

- Average speedup: 0.91x (±0.31)
- Average memory ratio: 1.00x (±0.00)

## Recommendations

- ⚠️ **Hilbert ordering shows minimal performance benefits**
- Consider disabling Hilbert ordering to reduce complexity

### Best Use Cases for Hilbert Ordering

Based on the benchmarks:
- **Most beneficial configurations:**
  - Seq=8,192, Batch=1, Heads=16: 1.37x speedup
  - Seq=4,096, Batch=2, Heads=8: 1.25x speedup
  - Seq=2,048, Batch=2, Heads=8: 1.11x speedup
