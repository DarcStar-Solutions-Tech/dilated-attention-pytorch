# Memory Scaling Analysis

**World Size**: 1 GPU(s)

## Scaling Analysis


### Ring Hilbert

- Sequence length increased: 4.0x
- Memory increased: 14.4x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Block Sparse

- Sequence length increased: 4.0x
- Memory increased: 14.5x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Standard

- Sequence length increased: 4.0x
- Memory increased: 12.9x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

## Memory Efficiency (KB per Token)

| Sequence Length | Ring Hilbert | Ring Block Sparse | Ring Standard |
|-----------------|---------------------------------------------------------------------------|
|           1,024 |                  74.062 |                  75.062 |                  82.188 |
|           2,048 |                 138.062 |                 140.062 |                 138.062 |
|           4,096 |                 266.062 |                 272.031 |                 266.062 |
