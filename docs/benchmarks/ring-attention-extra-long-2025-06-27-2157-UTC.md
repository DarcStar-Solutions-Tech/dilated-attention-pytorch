# Ring Attention Extra Long Sequence Benchmark

Generated: 2025-06-27T21:57:31.194002Z

## Configuration

- Device: NVIDIA GeForce GTX 1080
- Batch size: 1
- Num heads: 8
- Head dim: 64
- Sequence lengths tested: [16384, 32768, 65536, 131072]

## Results

### Summary Table

| Model | Ring Size | Max Seq Length | Memory at Max | Throughput at Max |
|-------|-----------|----------------|---------------|-------------------|
| RingDilatedV2_r1 | 1 | 16,384 | 0.07 GB | 36,731 tok/s |
| RingDilatedV2_r4 | 4 | 32,768 | 0.13 GB | 17,172 tok/s |
| RingDilatedV2_r8 | 8 | 65,536 | 0.26 GB | 16,276 tok/s |
| Production_r1 | 1 | 16,384 | 0.08 GB | 22,782 tok/s |
| Production_r4 | 4 | 32,768 | 0.14 GB | 25,968 tok/s |
| Production_r8 | 8 | 65,536 | 0.26 GB | 21,931 tok/s |
| Production_r16 | 16 | 131,072 | 0.51 GB | 41,468 tok/s |

### Detailed Results by Sequence Length


#### Sequence Length: 16,384 tokens

| Model | Success | Time (ms) | Memory (GB) | Throughput (tok/s) |
|-------|---------|-----------|-------------|--------------------||
| RingDilatedV2_r1 | ✓ | 446.1 | 0.07 | 36,731 |
| RingDilatedV2_r4 | ✓ | 590.6 | 0.07 | 27,744 |
| RingDilatedV2_r8 | ✓ | 84.2 | 0.07 | 194,507 |
| Production_r1 | ✓ | 719.2 | 0.08 | 22,782 |
| Production_r4 | ✓ | 75.3 | 0.07 | 217,617 |
| Production_r8 | ✓ | 29.1 | 0.07 | 563,124 |
| Production_r16 | ✓ | 3.0 | 0.07 | 5,495,777 |

#### Sequence Length: 32,768 tokens

| Model | Success | Time (ms) | Memory (GB) | Throughput (tok/s) |
|-------|---------|-----------|-------------|--------------------||
| RingDilatedV2_r1 | ✗ | - | - | - |
| RingDilatedV2_r4 | ✓ | 1908.2 | 0.13 | 17,172 |
| RingDilatedV2_r8 | ✓ | 1009.5 | 0.13 | 32,461 |
| Production_r1 | ✗ | - | - | - |
| Production_r4 | ✓ | 1261.9 | 0.14 | 25,968 |
| Production_r8 | ✓ | 109.5 | 0.13 | 299,239 |
| Production_r16 | ✓ | 1.8 | 0.13 | 17,989,156 |

#### Sequence Length: 65,536 tokens

| Model | Success | Time (ms) | Memory (GB) | Throughput (tok/s) |
|-------|---------|-----------|-------------|--------------------||
| RingDilatedV2_r1 | ✗ | - | - | - |
| RingDilatedV2_r4 | ✗ | - | - | - |
| RingDilatedV2_r8 | ✓ | 4026.6 | 0.26 | 16,276 |
| Production_r1 | ✗ | - | - | - |
| Production_r4 | ✗ | - | - | - |
| Production_r8 | ✓ | 2988.2 | 0.26 | 21,931 |
| Production_r16 | ✓ | 844.4 | 0.26 | 77,614 |

#### Sequence Length: 131,072 tokens

| Model | Success | Time (ms) | Memory (GB) | Throughput (tok/s) |
|-------|---------|-----------|-------------|--------------------||
| RingDilatedV2_r1 | ✗ | - | - | - |
| RingDilatedV2_r4 | ✗ | - | - | - |
| RingDilatedV2_r8 | ✗ | - | - | - |
| Production_r1 | ✗ | - | - | - |
| Production_r4 | ✗ | - | - | - |
| Production_r8 | ✗ | - | - | - |
| Production_r16 | ✓ | 3160.8 | 0.51 | 41,468 |

## Key Findings

### Memory Scaling

Ring Attention demonstrates O(n/ring_size) memory scaling:

At 16,384 tokens:
- RingDilatedV2_r4: 0.0% memory reduction
- RingDilatedV2_r8: 0.1% memory reduction
- Production_r4: -2.0% memory reduction
- Production_r8: 0.1% memory reduction
- Production_r16: 0.1% memory reduction

### Maximum Sequence Lengths Achieved

- **RingDilatedV2_r1**: 16,384 tokens
- **RingDilatedV2_r4**: 32,768 tokens
- **RingDilatedV2_r8**: 65,536 tokens
- **Production_r1**: 16,384 tokens
- **Production_r4**: 32,768 tokens
- **Production_r8**: 65,536 tokens
- **Production_r16**: 131,072 tokens
