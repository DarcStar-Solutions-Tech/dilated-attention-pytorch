# Memory Scaling Analysis

**World Size**: 1 GPU(s)

## Scaling Analysis


### Ring Block Sparse

- Sequence length increased: 4.0x
- Memory increased: 14.7x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Hilbert

- Sequence length increased: 4.0x
- Memory increased: 14.4x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

### Ring Standard

- Sequence length increased: 4.0x
- Memory increased: 11.8x
- Observed scaling: ~O(n²)
- Expected scaling: O(n/1)

## Memory Efficiency (KB per Token)

| Sequence Length | Ring Block Sparse | Ring Hilbert | Ring Standard |
|-----------------|---------------------------------------------------------------------------|
|           1,024 |                  38.031 |                  37.031 |                  45.156 |
|           2,048 |                  72.016 |                  69.031 |                  69.031 |
|           4,096 |                 140.016 |                 133.031 |                 133.031 |
