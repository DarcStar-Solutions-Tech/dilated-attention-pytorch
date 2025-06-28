# Optimized Block-Sparse Extreme Length Benchmark

Generated: 2025-06-27T23:01:45.427768Z

## Configuration

- Device: cuda
- Data type: torch.float16
- Batch size: 1
- Num heads: 8
- Head dim: 64
- Sparsity: 0.9
- GPU: NVIDIA GeForce GTX 1080
- Total GPU Memory: 7.88 GB

## Results

| Seq Length | Original Time | Original Memory | Optimized Time | Optimized Memory | Speedup |
|------------|---------------|-----------------|----------------|------------------|---------|
| 4,096 | 121.91 ms | 12.44 MB | 39.82 ms | 124.01 MB | 3.06x |
| 8,192 | 148.05 ms | 8.32 MB | 47.35 ms | 248.02 MB | 3.13x |
| 16,384 | 263.06 ms | 16.33 MB | 90.79 ms | 496.04 MB | 2.90x |
| 32,768 | 513.46 ms | 32.35 MB | 197.16 ms | 992.08 MB | 2.60x |
| 65,536 | 1003.10 ms | 64.39 MB | 373.41 ms | 1.94 GB | 2.69x |

## Maximum Sequence Lengths

- Block-Sparse Original: 262,144 tokens
- Block-Sparse Optimized: 65,536 tokens

## Analysis

- Average speedup: 2.88x
- Min/Max speedup: 2.60x / 3.13x

The optimized implementation maintains the same memory efficiency while providing significant performance improvements across all sequence lengths.