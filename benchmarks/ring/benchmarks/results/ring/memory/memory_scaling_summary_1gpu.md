# Memory Scaling Analysis

**World Size**: 1 GPU(s)

## Scaling Analysis


### Ring Hilbert

- Sequence length increased: 8.0x
- Memory increased: 50.6x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Block Sparse

- Sequence length increased: 8.0x
- Memory increased: 51.1x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Standard

- Sequence length increased: 8.0x
- Memory increased: 36.5x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

## Memory Efficiency (KB per Token)

| Sequence Length | Ring Hilbert | Ring Block Sparse | Ring Standard |
|-----------------|---------------------------------------------------------------------------|
|             512 |                  42.062 |                  42.562 |                  58.312 |
|           1,024 |                  74.062 |                  75.062 |                  74.062 |
|           2,048 |                 138.062 |                 140.062 |                 138.062 |
|           4,096 |                 266.062 |                 272.031 |                 266.062 |
