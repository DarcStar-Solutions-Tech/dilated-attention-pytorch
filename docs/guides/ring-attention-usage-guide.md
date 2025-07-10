# Ring Attention Usage Guide

This guide covers the usage of Ring Attention implementations in the Dilated Attention PyTorch library. Ring Attention enables processing of extremely long sequences with O(n/k) memory complexity where k is the number of devices.

## Table of Contents
1. [Overview](#overview)
2. [Available Implementations](#available-implementations)
3. [Quick Start](#quick-start)
4. [Advanced Usage](#advanced-usage)
5. [Migration Guide](#migration-guide)
6. [Performance Tips](#performance-tips)
7. [Troubleshooting](#troubleshooting)

## Overview

Ring Attention is a distributed attention mechanism that splits sequences across multiple devices in a ring topology. Each device processes a local chunk while communicating with neighbors to compute the full attention result.

### Key Benefits
- **O(n/k) Memory Complexity**: Linear memory scaling with number of devices
- **Billion-Token Sequences**: Process sequences up to 1B+ tokens
- **Efficient Communication**: Uses isend/irecv for optimal ring communication
- **Production Ready**: Includes error recovery and monitoring

### When to Use Ring Attention
- Sequences longer than 32K tokens
- Multi-GPU environments
- Memory-constrained scenarios
- Training/inference on very long documents

## Available Implementations

### 1. StandardRingAttention
The basic ring attention implementation with clean, efficient code.

```python
from dilated_attention_pytorch.ring import StandardRingAttention, RingAttentionConfig

config = RingAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)

attention = StandardRingAttention(config)
```

### 2. HilbertRingAttention
Ring attention with Hilbert curve optimization for improved cache locality.

```python
from dilated_attention_pytorch.ring import HilbertRingAttention, RingAttentionConfig

config = RingAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    use_hilbert=True,
    hilbert_curve_level=8
)

attention = HilbertRingAttention(config)
```

### 3. DistributedRingAttention
Enterprise-grade ring attention with advanced features.

```python
from dilated_attention_pytorch.ring import DistributedRingAttention, RingAttentionConfig

config = RingAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    use_checkpoint_accumulation=True,
    enable_mixed_precision=True,
    gradient_compression_ratio=0.1
)

attention = DistributedRingAttention(config)
```

### 4. BlockSparseRingAttention
Ring attention with block-sparse patterns for additional speedup.

```python
from dilated_attention_pytorch.ring import BlockSparseRingAttention, RingAttentionConfig

config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    block_size=64,
    sparsity_ratio=0.1  # 90% sparse
)

attention = BlockSparseRingAttention(config)
```

## Quick Start

### Single GPU Usage
Ring attention works on single GPU but provides no memory benefits:

```python
import torch
from dilated_attention_pytorch.ring import create_ring_attention

# Create attention module
attention = create_ring_attention(
    "standard",
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)

# Forward pass
batch_size = 2
seq_len = 8192
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn(batch_size, seq_len, num_heads, head_dim)
v = torch.randn(batch_size, seq_len, num_heads, head_dim)

output = attention(q, k, v, is_causal=True)
```

### Multi-GPU Usage
Ring attention shines with multiple GPUs:

```python
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring import create_ring_attention

# Initialize distributed (done by your training framework)
dist.init_process_group(backend="nccl")

# Create attention module
attention = create_ring_attention(
    "standard",
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    ring_size=dist.get_world_size()
)

# Each rank processes its local chunk
local_batch_size = 2
local_seq_len = 8192 // dist.get_world_size()  # Split across GPUs
num_heads = 8
head_dim = 64

# Local tensors - already split!
q_local = torch.randn(local_batch_size, local_seq_len, num_heads, head_dim).cuda()
k_local = torch.randn(local_batch_size, local_seq_len, num_heads, head_dim).cuda()
v_local = torch.randn(local_batch_size, local_seq_len, num_heads, head_dim).cuda()

# Ring attention handles communication
output = attention(q_local, k_local, v_local, already_split=True)
```

### With PyTorch Lightning
```python
import pytorch_lightning as pl
from dilated_attention_pytorch.ring import create_ring_attention

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.attention = create_ring_attention(
            "standard",
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2]
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # Split into q, k, v and add head dimension
        # ... your preprocessing ...
        
        output = self.attention(q, k, v)
        return output
```

## Advanced Usage

### Custom Configuration
```python
from dilated_attention_pytorch.ring import RingAttentionConfig, create_ring_attention

# Detailed configuration
config = RingAttentionConfig(
    # Dilated attention settings
    segment_lengths=[1024, 2048, 4096, 8192],
    dilation_rates=[1, 2, 4, 8],
    dropout=0.1,
    
    # Ring communication settings
    ring_size=None,  # Auto-detect from distributed
    enable_profiling=True,
    log_communication_stats=True,
    validate_gradients=True,
    
    # Optimization settings
    use_checkpoint_accumulation=True,
    checkpoint_every_n_layers=2,
    enable_mixed_precision=True,
    
    # Memory settings
    max_buffer_size_mb=1024,
    enable_memory_pool=True,
    
    # Hilbert optimization
    use_hilbert=True,
    hilbert_curve_level=8
)

attention = create_ring_attention("hilbert", config=config)
```

### Gradient Compression
For bandwidth-limited clusters:

```python
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    gradient_compression_ratio=0.1,  # Compress to 10%
    use_error_feedback=True,  # Error feedback for accuracy
    compression_method="topk"  # or "randomk"
)

attention = DistributedRingAttention(config)
```

### Memory Profiling
```python
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    enable_profiling=True,
    profile_memory=True,
    profile_communication=True
)

attention = create_ring_attention("standard", config=config)

# After forward pass
stats = attention.get_profiling_stats()
print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
print(f"Communication time: {stats['comm_time_ms']:.2f} ms")
```

### Custom Ring Topology
```python
# Custom ring with specific device ordering
import os
os.environ["RING_DEVICE_ORDER"] = "0,2,1,3"  # Custom ring order

config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    custom_ring_order=[0, 2, 1, 3]
)
```

## Migration Guide

### From Old Ring Implementations

```python
# Old (deprecated)
from dilated_attention_pytorch.ring_dilated_attention_production import RingDilatedAttentionProduction
attention = RingDilatedAttentionProduction(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)

# New (recommended)
from dilated_attention_pytorch.ring import StandardRingAttention, RingAttentionConfig
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
attention = StandardRingAttention(config)
```

### From Non-Ring to Ring

```python
# Before: Standard attention (limited to ~32K tokens)
from dilated_attention_pytorch import MultiheadDilatedAttention
attention = MultiheadDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)

# After: Ring attention (scales to billions of tokens)
from dilated_attention_pytorch.ring import create_ring_attention
attention = create_ring_attention(
    "standard",
    embed_dim=768,  # Only for multihead variants
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

## Performance Tips

### 1. Sequence Length Requirements
- Must be divisible by `world_size * max(segment_lengths)`
- Pad sequences if necessary:
```python
def pad_sequence_for_ring(seq, world_size, max_segment_len):
    required_multiple = world_size * max_segment_len
    current_len = seq.shape[1]
    
    if current_len % required_multiple != 0:
        pad_len = required_multiple - (current_len % required_multiple)
        seq = F.pad(seq, (0, 0, 0, pad_len))  # Pad seq dimension
    
    return seq
```

### 2. Optimal Settings by GPU Count
```python
def get_optimal_config(num_gpus, target_seq_len):
    if num_gpus <= 2:
        return RingAttentionConfig(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2]
        )
    elif num_gpus <= 4:
        return RingAttentionConfig(
            segment_lengths=[2048, 4096, 8192],
            dilation_rates=[1, 2, 4]
        )
    else:  # 8+ GPUs
        return RingAttentionConfig(
            segment_lengths=[4096, 8192, 16384],
            dilation_rates=[1, 2, 4],
            use_checkpoint_accumulation=True
        )
```

### 3. Network Optimization
```bash
# For InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# For Ethernet
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

# General optimizations
export NCCL_TREE_THRESHOLD=0
```

### 4. Memory Optimization
```python
# Enable memory pooling
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    enable_memory_pool=True,
    max_buffer_size_mb=2048,  # 2GB pool
    aggressive_memory_cleanup=True
)

# Manual memory management
attention.clear_memory_pool()
torch.cuda.empty_cache()
```

## Troubleshooting

### Common Issues

#### 1. "Sequence length must be divisible by world_size"
```python
# Fix: Ensure proper padding
seq_len = 10000
world_size = 4
max_segment = 2048

required_len = world_size * max_segment  # 8192
if seq_len % required_len != 0:
    pad_len = required_len - (seq_len % required_len)
    # Pad your sequences
```

#### 2. "Ring setup validation failed"
```python
# Check distributed initialization
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

# Verify all ranks have same config
config_tensor = torch.tensor([len(segment_lengths), len(dilation_rates)])
dist.broadcast(config_tensor, src=0)
```

#### 3. Out of Memory
```python
# Use gradient checkpointing
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    use_checkpoint_accumulation=True,
    checkpoint_every_n_layers=1  # More aggressive
)

# Or reduce precision
config.enable_mixed_precision = True
config.attention_dtype = torch.float16
```

#### 4. Communication Timeouts
```python
# Increase timeout
os.environ["NCCL_TIMEOUT"] = "600"  # 10 minutes

# Enable async communication
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    async_ring_communication=True
)
```

### Performance Debugging

```python
# Enable detailed profiling
config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    enable_profiling=True,
    profile_memory=True,
    profile_communication=True,
    log_communication_stats=True
)

attention = create_ring_attention("standard", config=config)

# Run forward pass
output = attention(q, k, v)

# Get detailed stats
if dist.get_rank() == 0:
    stats = attention.get_profiling_stats()
    print("Ring Attention Performance Report:")
    print(f"  Total time: {stats['total_time_ms']:.2f} ms")
    print(f"  Computation: {stats['compute_time_ms']:.2f} ms")
    print(f"  Communication: {stats['comm_time_ms']:.2f} ms")
    print(f"  Memory peak: {stats['peak_memory_mb']:.2f} MB")
    print(f"  Comm volume: {stats['comm_volume_mb']:.2f} MB")
```

## Best Practices

1. **Always validate sequence lengths** before passing to ring attention
2. **Use factory functions** (`create_ring_attention`) for future compatibility
3. **Profile first** to ensure ring attention provides benefits for your use case
4. **Start with StandardRingAttention** and optimize later if needed
5. **Monitor communication stats** in production for performance insights

## Example: End-to-End Training Script

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dilated_attention_pytorch.ring import create_ring_attention

def train_with_ring_attention():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create model with ring attention
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = create_ring_attention(
                "standard",
                segment_lengths=[2048, 4096, 8192],
                dilation_rates=[1, 2, 4],
                dropout=0.1
            )
            
        def forward(self, q, k, v):
            return self.attention(q, k, v, already_split=True)
    
    # Setup model
    model = Model().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Assume batch is already split across ranks
            q, k, v = batch  # Each shape: (B, L/world_size, H, D)
            
            output = model(q, k, v)
            loss = compute_loss(output)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train_with_ring_attention()
```

## Multi-GPU Communication Issues

**Important**: Some ring attention implementations have known issues with multi-GPU communication that can cause CUDA errors or hangs. See the [Ring Attention Multi-GPU Fixes Guide](ring-attention-multi-gpu-fixes.md) for:

- Common error patterns and their causes
- Critical fixes from lucidrains/ring-attention-pytorch
- Working examples with patched communication
- Implementation status for each variant

Key fixes include:
1. Ensuring tensor contiguity with `.contiguous()` before communication
2. Using `dist.barrier()` after P2P operations
3. Using `dist.batch_isend_irecv()` instead of separate operations

## Conclusion

Ring Attention is a powerful technique for scaling attention to extremely long sequences. With this guide, you should be able to:

- Choose the right ring attention implementation for your needs
- Configure it properly for optimal performance  
- Debug common issues
- Migrate from older implementations
- Fix multi-GPU communication issues

For more examples, see the `examples/ring_attention/` directory in the repository.