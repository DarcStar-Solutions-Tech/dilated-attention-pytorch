# Memory Scaling Analysis

**World Size**: 1 GPU(s)

## Scaling Analysis


### Ring Block Sparse

- Sequence length increased: 8.0x
- Memory increased: 50.1x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Standard

- Sequence length increased: 8.0x
- Memory increased: 28.0x
- Observed scaling: ~O(n)
- Expected scaling: O(n/1)

### Ring Hilbert

- Sequence length increased: 8.0x
- Memory increased: 48.7x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

## Memory Efficiency (KB per Token)

| Sequence Length | Ring Block Sparse | Ring Standard | Ring Hilbert |
|-----------------|---------------------------------------------------------------------------|
|             512 |                  22.531 |                  38.281 |                  22.031 |
|           1,024 |                  39.031 |                  38.031 |                  38.031 |
|           2,048 |                  73.016 |                  70.031 |                  70.031 |
|           4,096 |                 141.016 |                 134.031 |                 134.031 |
