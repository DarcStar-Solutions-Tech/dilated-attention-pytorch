# Examples

This directory contains example scripts demonstrating how to use the dilated attention implementations.

## Distributed Ring Attention

Ring Attention is a distributed algorithm that enables training with extremely long sequences by distributing the key-value pairs across multiple GPUs. Unlike standard attention which requires O(n²) memory on each device, Ring Attention achieves O(n²/P) memory complexity where P is the number of devices.

### How Ring Attention Works

1. **Query Distribution**: Each device keeps the full query tensor
2. **K/V Sharding**: Key and value tensors are sharded across devices
3. **Ring Communication**: K/V chunks rotate through devices using collective communication
4. **Local Computation**: Each device computes attention for its queries against each K/V chunk
5. **Accumulation**: Results are accumulated as K/V chunks rotate through the ring

### Running the Example

The `distributed_ring_attention.py` example demonstrates:
- Memory savings compared to standard attention
- How to use Ring Attention in a training loop
- Proper distributed setup with PyTorch

**Requirements:**
- Multiple GPUs (at least 2)
- PyTorch with NCCL support

**Single Node (Multiple GPUs):**
```bash
torchrun --nproc_per_node=4 examples/distributed_ring_attention.py
```

**Multiple Nodes:**
```bash
# On each node:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/distributed_ring_attention.py
```

### Key Benefits

1. **Memory Efficiency**: Train with sequences that would OOM with standard attention
2. **Linear Scaling**: Memory per device decreases linearly with more GPUs
3. **Exact Computation**: Mathematically equivalent to standard attention

### Implementation Details

Our Ring Attention implementations:
- `RingDilatedAttention` (alias for `RingDilatedAttentionV2Collective`): Uses all_gather for robust collective communication
- `RingMultiheadDilatedAttention`: Multihead wrapper with MAGNETO improvements
- Automatic ring size detection based on world size
- Support for FP16/BF16 training
- Gradient checkpointing compatible

### Common Issues

1. **NCCL Errors**: Ensure all GPUs can communicate (check with `nvidia-smi topo -m`)
2. **OOM on Small GPUs**: Reduce batch size or sequence length
3. **Hanging**: Check that all ranks are initialized properly

For more details, see the [Ring Attention Guide](../docs/guides/ring-attention-guide.md).