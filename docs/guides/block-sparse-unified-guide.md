# Unified Block-Sparse Attention Guide

**Last Updated**: 2025-07-07

This guide provides a comprehensive overview of all block-sparse attention implementations and how to use them effectively.

## 🚀 Quick Start

```python
from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention

# Auto-select best implementation
attention = create_block_sparse_attention("auto",
    sparsity_ratio=0.1,  # 90% sparse
    block_size=64
)

# Use specific variant
attention = create_block_sparse_attention("hierarchical")

# Use preset configuration
attention = get_block_sparse_preset("ultra_sparse")
```

## 📊 Available Implementations

### 1. Base Block-Sparse (`BlockSparseRingDilatedAttention`)

**When to use**: General-purpose sparse attention with fixed patterns

```python
from dilated_attention_pytorch import create_block_sparse_attention

# Create with specific pattern
attention = create_block_sparse_attention("base",
    sparse_config=SparsePatternConfig(
        pattern_type="dilated_sparse",  # or "local_window", "global_local"
        sparsity_ratio=0.1,             # 90% sparse
        block_size=64,
        dilation_rates=[1, 2, 4, 8]    # For dilated pattern
    )
)

# Simple creation
attention = create_block_sparse_attention("base",
    sparsity_ratio=0.05  # 95% sparse
)
```

**Pattern Types**:
- `local_window`: Each position attends to nearby positions
- `dilated_sparse`: Multi-scale attention with dilation
- `global_local`: Combination of global tokens and local windows

### 2. Hierarchical Block-Sparse (`BlockSparseHierarchical`)

**When to use**: Need multi-scale attention patterns

```python
# Use preset
attention = create_block_sparse_attention("hierarchical")

# Custom hierarchy
from dilated_attention_pytorch import HierarchicalConfig

config = HierarchicalConfig(
    level_configs=[
        {"stride": 1, "window_size": 128, "block_size": 32},    # Fine
        {"stride": 4, "window_size": 512, "block_size": 64},    # Medium
        {"stride": 16, "window_size": -1, "block_size": 128},   # Coarse (-1 = global)
    ]
)
attention = create_block_sparse_attention("hierarchical",
    hierarchical_config=config
)

# Available presets
attention = get_block_sparse_preset("hierarchical_long")  # Long-range dependencies
attention = get_block_sparse_preset("hierarchical_fine")  # Fine-grained patterns
```

### 3. Adaptive Block-Sparse (`BlockSparseAdaptive`)

**When to use**: Want patterns to adapt to your data

```python
from dilated_attention_pytorch import AdaptiveConfig

# Basic adaptive attention
attention = create_block_sparse_attention("adaptive")

# Custom configuration
config = AdaptiveConfig(
    base_sparsity=0.9,          # Start with 90% sparsity
    temperature=1.0,            # Gumbel-softmax temperature
    learnable_temperature=True,  # Anneal during training
    hidden_dim=128,             # Importance network size
    share_across_heads=False    # Per-head patterns
)
attention = create_block_sparse_attention("adaptive",
    adaptive_config=config
)

# Training utilities
from dilated_attention_pytorch import AdaptiveSparsityTrainer

trainer = AdaptiveSparsityTrainer(
    model=attention,
    initial_temperature=1.0,
    final_temperature=0.1,
    annealing_steps=10000
)

# In training loop
for batch in dataloader:
    output = attention(q, k, v)
    loss = compute_loss(output)
    loss.backward()
    optimizer.step()
    trainer.step()  # Anneal temperature
```

### 4. Multihead Block-Sparse (`BlockSparseRingMultiheadDilatedAttention`)

**When to use**: Drop-in replacement for `nn.MultiheadAttention`

```python
# Direct replacement
attention = create_block_sparse_attention("multihead",
    embed_dim=768,
    num_heads=12,
    sparsity_ratio=0.1,
    dropout=0.1,
    batch_first=True
)

# Use like nn.MultiheadAttention
output = attention(query, key, value, is_causal=True)

# With attention weights
output, attn_weights = attention(
    query, key, value, 
    need_weights=True,
    key_padding_mask=padding_mask
)
```

### 5. Distributed Block-Sparse (`BlockSparseRingDistributedDilatedAttention`)

**When to use**: Multi-GPU/multi-node training

```python
from dilated_attention_pytorch import DistributedSparseConfig

config = DistributedSparseConfig(
    enable_gradient_compression=True,
    compression_ratio=0.1,
    pattern_update_interval=100,
    load_balance_interval=1000
)

attention = create_block_sparse_attention("distributed",
    distributed_config=config,
    enable_deepspeed_integration=True
)
```

## 🎯 Choosing the Right Implementation

### Decision Tree

```python
def choose_block_sparse(seq_len, use_case, num_gpus=1):
    """Help choose the right implementation."""
    
    if use_case == "research":
        # Adaptive learns optimal patterns
        return "adaptive"
    
    elif use_case == "production":
        if num_gpus > 1:
            return "distributed"
        elif seq_len > 50000:
            return "hierarchical"  # Better for very long sequences
        else:
            return "base"
    
    elif use_case == "drop_in_replacement":
        return "multihead"
    
    elif use_case == "extreme_length":
        # Use hierarchical with custom config
        return "hierarchical"
    
    else:
        return "base"  # Good default
```

### Performance Characteristics

| Implementation | Memory | Speed | Best Sequence Length | Key Feature |
|----------------|--------|-------|---------------------|-------------|
| Base | O(n×s) | Fast | 1K-50K | Simple, efficient |
| Hierarchical | O(n×s) | Fast | 10K-1M | Multi-scale patterns |
| Adaptive | O(n×s) | Medium | Any | Learns from data |
| Multihead | O(n×s) | Fast | 1K-50K | PyTorch compatible |
| Distributed | O(n×s/p) | Fast* | 50K-10M | Multi-GPU scaling |

*Speed depends on communication overhead

## 📋 Common Patterns

### 1. Extreme Sparsity (99%+)

```python
# For very long sequences
attention = get_block_sparse_preset("ultra_sparse")

# Custom ultra-sparse
attention = create_block_sparse_attention("base",
    sparsity_ratio=0.001,  # 99.9% sparse
    block_size=256,        # Larger blocks for efficiency
    segment_lengths=[8192, 16384, 32768],
    dilation_rates=[1, 8, 64]
)
```

### 2. Local + Global Attention

```python
# Preset
attention = get_block_sparse_preset("global_local")

# Custom
from dilated_attention_pytorch import SparsePatternConfig

config = SparsePatternConfig(
    pattern_type="global_local",
    global_tokens=128,    # First 128 tokens are global
    window_size=512,      # Local window size
    sparsity_ratio=0.05
)
attention = create_block_sparse_attention("base", sparse_config=config)
```

### 3. Progressive Sparsity

```python
# Start dense, gradually increase sparsity
class ProgressiveSparsityScheduler:
    def __init__(self, model, initial_sparsity=0.5, final_sparsity=0.95):
        self.model = model
        self.initial = initial_sparsity
        self.final = final_sparsity
        self.current_step = 0
        self.total_steps = 10000
    
    def step(self):
        progress = min(self.current_step / self.total_steps, 1.0)
        current_sparsity = self.initial + (self.final - self.initial) * progress
        
        # Update model's sparsity
        self.model.sparse_config.sparsity_ratio = current_sparsity
        self.current_step += 1
```

## 🔧 Advanced Usage

### Custom Pattern Generation

```python
class CustomPatternBlockSparse(BlockSparseRingDilatedAttention):
    def _get_sparse_block_indices(self, num_blocks, num_heads, device):
        # Your custom pattern logic
        row_indices = []
        col_indices = []
        
        # Example: Butterfly pattern
        for i in range(num_blocks):
            # Local connections
            row_indices.append(i)
            col_indices.append(i)
            
            # Butterfly connections
            stride = 1
            while stride < num_blocks:
                if i + stride < num_blocks:
                    row_indices.append(i)
                    col_indices.append(i + stride)
                stride *= 2
        
        return torch.tensor(row_indices), torch.tensor(col_indices)
```

### Combining Patterns

```python
# Use different patterns for different heads
class MultiPatternAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        
        # Different patterns for different head groups
        self.local_heads = create_block_sparse_attention("base",
            sparse_config=SparsePatternConfig(pattern_type="local_window")
        )
        self.global_heads = create_block_sparse_attention("base",
            sparse_config=SparsePatternConfig(pattern_type="dilated_sparse")
        )
        
    def forward(self, q, k, v):
        # Split heads
        local_output = self.local_heads(q[..., :6, :], k[..., :6, :], v[..., :6, :])
        global_output = self.global_heads(q[..., 6:, :], k[..., 6:, :], v[..., 6:, :])
        
        # Combine
        return torch.cat([local_output, global_output], dim=-2)
```

## 🐛 Troubleshooting

### Common Issues

1. **Device Mismatch**
```python
# Ensure all components on same device
attention = create_block_sparse_attention("multihead", ...).to(device)
```

2. **Sequence Length Requirements**
```python
# Sequence length must be divisible by block_size
seq_len = ((original_seq_len + block_size - 1) // block_size) * block_size
padded_input = F.pad(input, (0, 0, 0, seq_len - original_seq_len))
```

3. **Memory Issues**
```python
# Reduce memory usage
attention = create_block_sparse_attention("base",
    sparsity_ratio=0.01,      # More sparse
    block_size=128,           # Larger blocks
    use_memory_pool=True,     # Enable memory pooling
    mixed_precision=True      # Use fp16
)
```

## 📈 Performance Tips

1. **Batch Operations**: Process multiple sequences together
2. **Pattern Caching**: Reuse patterns across forward passes
3. **Block Size**: Larger blocks = better GPU utilization
4. **Sparsity Ratio**: Start with 90% (0.1), increase if needed
5. **Mixed Precision**: Use fp16/bf16 for better performance

## 🔗 Integration Examples

### With Transformers

```python
from transformers import BertModel

class SparseBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        
        # Replace self-attention layers
        for layer in self.bert.encoder.layer:
            layer.attention.self = create_block_sparse_attention(
                "multihead",
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                sparsity_ratio=0.1
            )
```

### With PyTorch Lightning

```python
class SparseTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.attention = create_block_sparse_attention(
            "adaptive" if self.training else "base"
        )
    
    def training_step(self, batch, batch_idx):
        # Adaptive patterns during training
        output = self.attention(batch['q'], batch['k'], batch['v'])
        loss = self.compute_loss(output, batch['target'])
        
        # Log sparsity
        if hasattr(self.attention, 'get_pattern_stats'):
            stats = self.attention.get_pattern_stats(batch['q'].shape[1])
            self.log('sparsity', stats['sparsity'])
        
        return loss
```

## 📚 References

- [Original LongNet Paper](https://arxiv.org/abs/2307.02486)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Ring Attention](https://arxiv.org/abs/2310.01889)

## 🎓 Next Steps

1. Run benchmarks to compare implementations
2. Start with base implementation
3. Experiment with different patterns
4. Consider adaptive for research
5. Use distributed for scale

For more details, see:
- [Performance Analysis](../reports/block-sparse-analysis-2025-07-07-0142-UTC.md)
- [API Reference](../api/block-sparse.md)
- [Benchmarking Guide](benchmarking-guide.md)