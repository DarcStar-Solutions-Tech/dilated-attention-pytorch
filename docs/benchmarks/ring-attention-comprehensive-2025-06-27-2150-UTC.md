# Comprehensive Ring Attention Benchmark

Generated: 2025-06-27T21:50:29.035427Z

## Summary

Performance comparison of all Ring Attention implementations:
- **RingDilatedV2**: Main implementation
- **RingDilatedV2_Ring4**: With ring size 4 (simulated)
- **Production_NoCP**: Production version without gradient checkpointing
- **Production_CP**: Production version with gradient checkpointing
- **Production_Ring4**: Production version with ring size 4
- **ImprovedDilated**: Baseline dilated attention

## Results by Sequence Length

### Sequence Length: 1,024

| Model | Time (ms) | Memory (GB) | Throughput (tokens/sec) | Relative Speed |
|-------|-----------|-------------|------------------------|----------------|
| Production_Ring4 | 0.30 | 0.014 | 3,461,393 | 23.55x |
| RingDilatedV2 | 1.44 | 0.030 | 713,219 | 4.85x |
| Production_NoCP | 1.51 | 0.032 | 676,866 | 4.61x |
| Production_CP | 1.70 | 0.032 | 604,122 | 4.11x |
| RingDilatedV2_Ring4 | 1.89 | 0.015 | 542,639 | 3.69x |
| ImprovedDilated | 6.97 | 0.017 | 146,953 | 1.00x |

### Sequence Length: 2,048

| Model | Time (ms) | Memory (GB) | Throughput (tokens/sec) | Relative Speed |
|-------|-----------|-------------|------------------------|----------------|
| RingDilatedV2_Ring4 | 1.88 | 0.024 | 1,087,576 | 7.32x |
| Production_Ring4 | 1.97 | 0.025 | 1,037,933 | 6.99x |
| RingDilatedV2 | 2.36 | 0.054 | 866,094 | 5.83x |
| Production_NoCP | 2.52 | 0.056 | 812,635 | 5.47x |
| Production_CP | 3.21 | 0.057 | 637,144 | 4.29x |
| ImprovedDilated | 13.78 | 0.025 | 148,571 | 1.00x |

### Sequence Length: 4,096

| Model | Time (ms) | Memory (GB) | Throughput (tokens/sec) | Relative Speed |
|-------|-----------|-------------|------------------------|----------------|
| Production_Ring4 | 2.76 | 0.044 | 1,482,814 | 10.10x |
| RingDilatedV2_Ring4 | 9.52 | 0.043 | 430,034 | 2.93x |
| RingDilatedV2 | 15.60 | 0.222 | 262,618 | 1.79x |
| ImprovedDilated | 27.90 | 0.042 | 146,787 | 1.00x |
| Production_CP | 31.46 | 0.231 | 130,196 | 0.89x |
| Production_NoCP | 49.98 | 0.228 | 81,951 | 0.56x |

### Sequence Length: 8,192

| Model | Time (ms) | Memory (GB) | Throughput (tokens/sec) | Relative Speed |
|-------|-----------|-------------|------------------------|----------------|
| Production_CP | 43.92 | 0.454 | 186,509 | 1.32x |
| RingDilatedV2 | 52.50 | 0.440 | 156,031 | 1.10x |
| ImprovedDilated | 57.90 | 0.078 | 141,473 | 1.00x |
| Production_NoCP | 69.18 | 0.448 | 118,414 | 0.84x |
| RingDilatedV2_Ring4 | 118.17 | 0.101 | 69,322 | 0.49x |
| Production_Ring4 | 167.79 | 0.117 | 48,823 | 0.35x |

## Key Findings

### Best Throughput by Sequence Length

- **1,024 tokens**: Production_Ring4 (3,461,393 tokens/sec)
- **2,048 tokens**: RingDilatedV2_Ring4 (1,087,576 tokens/sec)
- **4,096 tokens**: Production_Ring4 (1,482,814 tokens/sec)
- **8,192 tokens**: Production_CP (186,509 tokens/sec)

### Memory Efficiency

- **ImprovedDilated**: 0.078 GB (+0.0% vs baseline)
- **RingDilatedV2_Ring4**: 0.101 GB (-29.9% vs baseline)
- **Production_Ring4**: 0.117 GB (-50.2% vs baseline)
- **RingDilatedV2**: 0.440 GB (-466.6% vs baseline)
- **Production_NoCP**: 0.448 GB (-478.0% vs baseline)
- **Production_CP**: 0.454 GB (-485.6% vs baseline)
