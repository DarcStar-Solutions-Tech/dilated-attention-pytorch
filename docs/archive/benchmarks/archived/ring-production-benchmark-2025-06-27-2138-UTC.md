# Ring Attention Production Benchmark

Generated: 2025-06-27T21:38:46.276445Z

## Configuration
- Device: cuda
- Data type: torch.float16
- Batch size: 2
- Num heads: 8
- Head dim: 64
- Segment lengths: [256, 512]
- Dilation rates: [1, 2]

## Results

### Sequence Length: 1024

| Model | Forward (ms) | Backward (ms) | Memory (GB) |
|-------|--------------|---------------|-------------|
| RingDilatedV2 | 1.31 ± 0.01 | 10.67 ± 13.49 | 0.078 |
| Production_NoCP | FAILED | FAILED | N/A |
| Production_WithCP | 1.89 ± 0.25 | 10.19 ± 9.01 | 0.084 |
| Production_Ring4 | 2.48 ± 0.02 | 6.39 ± 0.27 | 0.061 |
| ImprovedDilated | 12.86 ± 0.66 | 33.95 ± 13.75 | 0.094 |

### Sequence Length: 2048

| Model | Forward (ms) | Backward (ms) | Memory (GB) |
|-------|--------------|---------------|-------------|
| RingDilatedV2 | 2.25 ± 0.29 | 6.47 ± 0.24 | 0.156 |
| Production_NoCP | FAILED | FAILED | N/A |
| Production_WithCP | 5.19 ± 0.84 | 107.80 ± 182.26 | 0.165 |
| Production_Ring4 | 5.60 ± 0.11 | 25.59 ± 23.45 | 0.127 |
| ImprovedDilated | 26.54 ± 2.25 | 93.66 ± 24.46 | 0.185 |

