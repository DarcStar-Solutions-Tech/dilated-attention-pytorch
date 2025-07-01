# Ring Attention Examples

This directory contains educational and reference implementations of Ring Attention to help understand the algorithm.

## Files

### reference_implementation.py (TrueRingDilatedAttention)
A reference implementation that demonstrates the correct Ring Attention algorithm:
- Shows how queries are NOT divided (each device has full Q tensor)
- Demonstrates K/V chunking and rotation through the ring
- Achieves O(n/ring_size) memory complexity for K/V
- Useful for understanding the mathematical correctness

### single_gpu_simulation.py (SimulatedRingDilatedAttention)
A single-GPU simulation that demonstrates Ring Attention benefits:
- Processes K/V chunks sequentially on one GPU
- Only keeps one K/V chunk in memory at a time
- Shows memory savings without requiring distributed setup
- Great for testing and understanding the algorithm

## Usage

These implementations are for educational purposes. For production use, please use the main implementations:
- `RingDilatedAttention` (alias for RingDilatedAttentionV2Collective)
- `RingMultiheadDilatedAttention` for multihead attention
- `RingDilatedAttentionProduction` for production deployments

## Key Concepts

Ring Attention achieves O(n) memory complexity by:
1. **Query Replication**: Each device maintains the full query tensor
2. **K/V Chunking**: Keys and values are split across devices
3. **Ring Communication**: K/V chunks rotate through devices
4. **Local Computation**: Each device computes attention for its K/V chunk
5. **Result Accumulation**: Final output combines results from all chunks

This is fundamentally different from naive data parallelism where sequences are split, which doesn't reduce memory complexity.