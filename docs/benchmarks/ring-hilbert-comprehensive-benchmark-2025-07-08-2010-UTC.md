# Ring Dilated Attention with Hilbert - Comprehensive Benchmark Report

Generated: 2025-07-08 20:10:52 UTC

## System Information

- Device: cuda
- Data Type: torch.float32
- GPU: NVIDIA GeForce GTX 1080
- CUDA Version: 12.6
- PyTorch Version: 2.7.1+cu126

## Performance Summary

### Sequence Length: 1,024 tokens

| Module                |   Batch |   Heads | Fwd (ms)    | Bwd (ms)      |   Total (ms) |   Peak Mem (MB) |   Throughput (tok/s) |   Mem/Token (KB) |
|-----------------------|---------|---------|-------------|---------------|--------------|-----------------|----------------------|------------------|
| Standard MHA          |       1 |       8 | 2.65±0.11   | 22.19±0.78    |        24.84 |            55.9 |                41231 |            55.93 |
| Ring+Hilbert (size=1) |       1 |       8 | 1.66±0.31   | 3.86±0.28     |         5.52 |            75.8 |               185601 |            75.75 |
| Standard MHA          |       1 |      16 | 3.51±0.92   | 19.11±2.11    |        22.62 |            56   |                45277 |            55.96 |
| Ring+Hilbert (size=1) |       1 |      16 | 2.74±0.31   | 17.95±23.74   |        20.68 |           122.8 |                49504 |           122.75 |
| Standard MHA          |       2 |       8 | 6.96±4.05   | 163.35±164.21 |       170.31 |            89   |                12025 |            44.48 |
| Ring+Hilbert (size=1) |       2 |       8 | 7.80±5.74   | 30.17±29.66   |        37.97 |           146.8 |                53938 |            73.38 |
| Standard MHA          |       2 |      16 | 5.54±0.59   | 178.91±210.92 |       184.45 |            89   |                11103 |            44.51 |
| Ring+Hilbert (size=1) |       2 |      16 | 61.19±40.85 | 25.14±24.56   |        86.33 |           242.3 |                23724 |           121.13 |

### Sequence Length: 2,048 tokens

| Module                |   Batch |   Heads | Fwd (ms)      | Bwd (ms)      |   Total (ms) |   Peak Mem (MB) |   Throughput (tok/s) |   Mem/Token (KB) |
|-----------------------|---------|---------|---------------|---------------|--------------|-----------------|----------------------|------------------|
| Standard MHA          |       1 |       8 | 40.96±58.59   | 495.00±483.33 |       535.96 |            77   |                 3821 |            38.48 |
| Dilated Attention     |       1 |       8 | 6.72±1.08     | 324.92±461.40 |       331.63 |            99.1 |                 6175 |            49.57 |
| Ring+Hilbert (size=1) |       1 |       8 | 80.47±16.39   | 57.14±64.11   |       137.61 |           165.3 |                14882 |            82.63 |
| Standard MHA          |       1 |      16 | 9.37±0.48     | 338.64±477.21 |       348.01 |            77   |                 5885 |            38.51 |
| Dilated Attention     |       1 |      16 | 90.56±81.30   | 199.88±303.68 |       290.44 |            99.2 |                 7051 |            49.59 |
| Ring+Hilbert (size=1) |       1 |      16 | 48.21±67.57   | 72.66±105.82  |       120.87 |           293.3 |                16944 |           146.63 |
| Standard MHA          |       2 |       8 | 219.75±193.75 | 530.29±438.94 |       750.03 |           143   |                 5461 |            35.76 |
| Dilated Attention     |       2 |       8 | 29.60±34.91   | 260.51±369.96 |       290.11 |           155.7 |                14119 |            38.93 |
| Ring+Hilbert (size=1) |       2 |       8 | 152.98±19.76  | 93.32±119.94  |       246.3  |           338.3 |                16630 |            84.56 |
| Standard MHA          |       2 |      16 | 241.83±224.02 | 513.12±467.21 |       754.95 |           143.1 |                 5426 |            35.79 |
| Dilated Attention     |       2 |      16 | 241.70±91.58  | 128.93±113.55 |       370.64 |           155.8 |                11051 |            38.96 |
| Ring+Hilbert (size=1) |       2 |      16 | 129.94±87.72  | 248.81±213.46 |       378.75 |           594.3 |                10815 |           148.56 |

### Sequence Length: 4,096 tokens

| Module                |   Batch |   Heads | Fwd (ms)      | Bwd (ms)       |   Total (ms) |   Peak Mem (MB) |   Throughput (tok/s) |   Mem/Token (KB) |
|-----------------------|---------|---------|---------------|----------------|--------------|-----------------|----------------------|------------------|
| Standard MHA          |       1 |       8 | 276.19±294.47 | 1203.45±853.06 |      1479.64 |           119   |                 2768 |            29.76 |
| Dilated Attention     |       1 |       8 | 22.19±19.59   | 316.32±260.92  |       338.5  |           155.7 |                12100 |            38.93 |
| Ring+Hilbert (size=1) |       1 |       8 | 62.42±32.37   | 111.61±104.39  |       174.03 |           351.3 |                23536 |            87.83 |
| Standard MHA          |       1 |      16 | 275.28±301.00 | 1481.35±756.27 |      1756.62 |           119.1 |                 2332 |            29.79 |
| Dilated Attention     |       1 |      16 | 204.49±114.93 | 140.15±152.64  |       344.64 |           155.8 |                11885 |            38.96 |
| Ring+Hilbert (size=1) |       1 |      16 | 243.16±33.91  | 309.86±210.59  |       553.02 |           606.8 |                 7407 |           151.71 |
| Standard MHA          |       2 |       8 | 628.60±481.68 | 2853.59±660.62 |      3482.19 |           251.1 |                 2353 |            31.39 |
| Dilated Attention     |       2 |       8 | 357.99±151.30 | 697.53±688.73  |      1055.52 |           275.9 |                 7761 |            34.49 |
| Ring+Hilbert (size=1) |       2 |       8 | 84.60±82.72   | 493.45±350.58  |       578.04 |           660.3 |                14172 |            82.54 |
| Standard MHA          |       2 |      16 | 968.35±529.64 | 2584.66±836.73 |      3553.01 |           251.4 |                 2306 |            31.42 |
| Dilated Attention     |       2 |      16 | 518.16±59.49  | 1001.10±640.54 |      1519.26 |           276.1 |                 5392 |            34.51 |
| Ring+Hilbert (size=1) |       2 |      16 | 428.06±45.35  | 616.81±479.97  |      1044.87 |          1173.3 |                 7840 |           146.67 |

## Scaling Analysis

### Sequence Length Scaling

#### Dilated Attention

- Batch=1, Heads=8:
  - Sequence lengths: [2048, 4096]
  - Total times (ms): ['331.6', '338.5']
  - Peak memory (MB): ['99.1', '155.7']
  - Time scaling: 1.02x for 2x sequence length
  - Scaling efficiency: 1.96

- Batch=1, Heads=16:
  - Sequence lengths: [2048, 4096]
  - Total times (ms): ['290.4', '344.6']
  - Peak memory (MB): ['99.2', '155.8']
  - Time scaling: 1.19x for 2x sequence length
  - Scaling efficiency: 1.69

- Batch=2, Heads=8:
  - Sequence lengths: [2048, 4096]
  - Total times (ms): ['290.1', '1055.5']
  - Peak memory (MB): ['155.7', '275.9']
  - Time scaling: 3.64x for 2x sequence length
  - Scaling efficiency: 0.55

- Batch=2, Heads=16:
  - Sequence lengths: [2048, 4096]
  - Total times (ms): ['370.6', '1519.3']
  - Peak memory (MB): ['155.8', '276.1']
  - Time scaling: 4.10x for 2x sequence length
  - Scaling efficiency: 0.49

#### Ring+Hilbert (size=1)

- Batch=1, Heads=8:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['5.5', '137.6', '174.0']
  - Peak memory (MB): ['75.8', '165.3', '351.3']
  - Time scaling: 31.54x for 4x sequence length
  - Scaling efficiency: 0.13

- Batch=1, Heads=16:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['20.7', '120.9', '553.0']
  - Peak memory (MB): ['122.8', '293.3', '606.8']
  - Time scaling: 26.74x for 4x sequence length
  - Scaling efficiency: 0.15

- Batch=2, Heads=8:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['38.0', '246.3', '578.0']
  - Peak memory (MB): ['146.8', '338.3', '660.3']
  - Time scaling: 15.22x for 4x sequence length
  - Scaling efficiency: 0.26

- Batch=2, Heads=16:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['86.3', '378.7', '1044.9']
  - Peak memory (MB): ['242.3', '594.3', '1173.3']
  - Time scaling: 12.10x for 4x sequence length
  - Scaling efficiency: 0.33

#### Standard MHA

- Batch=1, Heads=8:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['24.8', '536.0', '1479.6']
  - Peak memory (MB): ['55.9', '77.0', '119.0']
  - Time scaling: 59.58x for 4x sequence length
  - Scaling efficiency: 0.07

- Batch=1, Heads=16:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['22.6', '348.0', '1756.6']
  - Peak memory (MB): ['56.0', '77.0', '119.1']
  - Time scaling: 77.67x for 4x sequence length
  - Scaling efficiency: 0.05

- Batch=2, Heads=8:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['170.3', '750.0', '3482.2']
  - Peak memory (MB): ['89.0', '143.0', '251.1']
  - Time scaling: 20.45x for 4x sequence length
  - Scaling efficiency: 0.20

- Batch=2, Heads=16:
  - Sequence lengths: [1024, 2048, 4096]
  - Total times (ms): ['184.5', '754.9', '3553.0']
  - Peak memory (MB): ['89.0', '143.1', '251.4']
  - Time scaling: 19.26x for 4x sequence length
  - Scaling efficiency: 0.21

## Hilbert Ordering Impact

## Recommendations

### When to Use Hilbert Ordering

Based on the benchmark results:

- Hilbert ordering shows minimal performance benefits in these tests

### Optimal Configurations

- **1,024 tokens:**
  - Best throughput: Ring+Hilbert (size=1) (185601 tok/s)
  - Best memory efficiency: Standard MHA (44.48 KB/token)
- **2,048 tokens:**
  - Best throughput: Ring+Hilbert (size=1) (16944 tok/s)
  - Best memory efficiency: Standard MHA (35.76 KB/token)
- **4,096 tokens:**
  - Best throughput: Ring+Hilbert (size=1) (23536 tok/s)
  - Best memory efficiency: Standard MHA (29.76 KB/token)

### General Guidelines

1. **For sequences < 4K tokens**: Standard attention may be sufficient
2. **For sequences 4K-16K tokens**: Dilated attention provides good balance
3. **For sequences > 16K tokens**: Ring attention becomes necessary
4. **Hilbert ordering**: Most beneficial for very long sequences with specific access patterns
5. **Multi-GPU**: Ring attention scales well with increased ring size

## Raw Benchmark Data

Full results are saved in the accompanying JSON file.
