# Extreme Sequence Length Benchmark

Generated: 2025-06-27T22:53:40.315826Z

## Configuration

- Device: cuda
- Data type: torch.float16
- Batch size: 1
- Num heads: 8
- Head dim: 64
- GPU: NVIDIA GeForce GTX 1080
- Total GPU Memory: 7.88 GB

## Results Summary

| Implementation | Max Successful Length | Max Time (ms) | Max Memory | Failure Reason |
|----------------|----------------------|---------------|------------|----------------|
| DilatedAttention | 0 | 0.00 | 0.00 B | DIVISIBILITY |
| ImprovedDilatedAttention | 0 | 0.00 | 0.00 B | DIVISIBILITY |
| RingDilatedAttentionV2_r1 | 8,192 | 133.25 | 790.00 MB | OOM |
| RingDilatedAttentionV2_r4 | 32,768 | 1488.32 | 829.00 MB | OOM |
| RingDilatedAttentionV2_r8 | 65,536 | 2631.63 | 861.00 MB | OOM |
| RingDilatedAttentionV2_r16 | 131,072 | 8065.35 | 925.00 MB | OOM |
| RingDilatedAttentionProduction_r1 | 32,768 | 23.38 | 32.00 MB | OOM |
| RingDilatedAttentionProduction_r8 | 262,144 | 15.05 | 320.00 MB | N/A |
| RingDilatedAttentionProduction_r16 | 262,144 | 14.34 | 288.00 MB | N/A |
| BlockSparse_LocalWindow_0.9 | 262,144 | 4181.54 | 256.62 MB | N/A |
| BlockSparse_LocalWindow_0.95 | 262,144 | 4404.77 | 256.62 MB | N/A |

## Detailed Results

### DilatedAttention

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✗ | - | - | DIVISIBILITY |
| 8,192 | ✗ | - | - | DIVISIBILITY |
| 16,384 | ✗ | - | - | DIVISIBILITY |
| 32,768 | ✗ | - | - | DIVISIBILITY |
| 65,536 | ✗ | - | - | DIVISIBILITY |
| 131,072 | ✗ | - | - | DIVISIBILITY |
| 262,144 | ✗ | - | - | OOM |

### ImprovedDilatedAttention

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✗ | - | - | DIVISIBILITY |
| 8,192 | ✗ | - | - | DIVISIBILITY |
| 16,384 | ✗ | - | - | DIVISIBILITY |
| 32,768 | ✗ | - | - | DIVISIBILITY |
| 65,536 | ✗ | - | - | DIVISIBILITY |
| 131,072 | ✗ | - | - | DIVISIBILITY |
| 262,144 | ✗ | - | - | OTHER |

### RingDilatedAttentionV2_r1

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 133.25 | 211.19 MB | - |
| 8,192 | ✓ | 124.23 | 790.00 MB | - |
| 16,384 | ✗ | - | - | OOM |

### RingDilatedAttentionV2_r4

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 5.97 | 19.52 MB | - |
| 8,192 | ✓ | 20.66 | 63.03 MB | - |
| 16,384 | ✓ | 922.71 | 222.94 MB | - |
| 32,768 | ✓ | 1488.32 | 829.00 MB | - |
| 65,536 | ✗ | - | - | OOM |

### RingDilatedAttentionV2_r8

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 6.89 | 9.63 MB | - |
| 8,192 | ✓ | 11.88 | 23.52 MB | - |
| 16,384 | ✓ | 60.92 | 71.03 MB | - |
| 32,768 | ✓ | 1147.30 | 238.94 MB | - |
| 65,536 | ✓ | 2631.63 | 861.00 MB | - |
| 131,072 | ✗ | - | - | OOM |

### RingDilatedAttentionV2_r16

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 10.39 | 5.63 MB | - |
| 8,192 | ✓ | 10.39 | 13.63 MB | - |
| 16,384 | ✓ | 27.27 | 31.52 MB | - |
| 32,768 | ✓ | 132.10 | 87.03 MB | - |
| 65,536 | ✓ | 1779.83 | 270.94 MB | - |
| 131,072 | ✓ | 8065.35 | 925.00 MB | - |
| 262,144 | ✗ | - | - | OOM |

### RingDilatedAttentionProduction_r1

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 2.49 | 4.00 MB | - |
| 8,192 | ✓ | 23.38 | 8.00 MB | - |
| 16,384 | ✓ | 10.52 | 16.00 MB | - |
| 32,768 | ✓ | 19.51 | 32.00 MB | - |
| 65,536 | ✗ | - | - | OOM |

### RingDilatedAttentionProduction_r8

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 0.59 | 5.00 MB | - |
| 8,192 | ✓ | 0.73 | 10.00 MB | - |
| 16,384 | ✓ | 0.62 | 20.00 MB | - |
| 32,768 | ✓ | 1.35 | 40.00 MB | - |
| 65,536 | ✓ | 3.09 | 80.00 MB | - |
| 131,072 | ✓ | 6.93 | 160.00 MB | - |
| 262,144 | ✓ | 15.05 | 320.00 MB | - |

### RingDilatedAttentionProduction_r16

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 1.02 | 4.50 MB | - |
| 8,192 | ✓ | 1.27 | 9.00 MB | - |
| 16,384 | ✓ | 1.09 | 18.00 MB | - |
| 32,768 | ✓ | 1.39 | 36.00 MB | - |
| 65,536 | ✓ | 3.08 | 72.00 MB | - |
| 131,072 | ✓ | 6.95 | 144.00 MB | - |
| 262,144 | ✓ | 14.34 | 288.00 MB | - |

### BlockSparse_LocalWindow_0.9

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 73.87 | 4.32 MB | - |
| 8,192 | ✓ | 128.37 | 8.32 MB | - |
| 16,384 | ✓ | 239.89 | 16.33 MB | - |
| 32,768 | ✓ | 487.99 | 32.35 MB | - |
| 65,536 | ✓ | 989.59 | 64.39 MB | - |
| 131,072 | ✓ | 2307.70 | 128.47 MB | - |
| 262,144 | ✓ | 4181.54 | 256.62 MB | - |

### BlockSparse_LocalWindow_0.95

| Seq Length | Status | Time (ms) | Memory | Error |
|------------|--------|-----------|--------|-------|
| 4,096 | ✓ | 70.92 | 4.32 MB | - |
| 8,192 | ✓ | 116.52 | 8.32 MB | - |
| 16,384 | ✓ | 247.78 | 16.33 MB | - |
| 32,768 | ✓ | 507.09 | 32.35 MB | - |
| 65,536 | ✓ | 1002.07 | 64.39 MB | - |
| 131,072 | ✓ | 2118.80 | 128.47 MB | - |
| 262,144 | ✓ | 4404.77 | 256.62 MB | - |

## Analysis

- **Longest sequence achieved**: 262,144 tokens by RingDilatedAttentionProduction_r8

### Ring Attention Analysis

- RingDilatedAttentionV2_r1: up to 8,192 tokens
- RingDilatedAttentionV2_r4: up to 32,768 tokens
- RingDilatedAttentionV2_r8: up to 65,536 tokens
- RingDilatedAttentionV2_r16: up to 131,072 tokens
- RingDilatedAttentionProduction_r1: up to 32,768 tokens
- RingDilatedAttentionProduction_r8: up to 262,144 tokens
- RingDilatedAttentionProduction_r16: up to 262,144 tokens
