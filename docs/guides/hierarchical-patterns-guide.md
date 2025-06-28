# Hierarchical Attention Patterns Guide

## Overview

Hierarchical attention patterns provide multi-scale coverage of the attention space by combining different levels of granularity. This approach captures both local details and global context efficiently while maintaining sparsity.

## Key Concepts

### Multi-Scale Attention

Hierarchical patterns combine multiple attention scales:

1. **Fine-grained Level**: High-resolution attention for capturing local dependencies
2. **Medium-grained Level**: Regional attention for medium-range dependencies  
3. **Coarse-grained Level**: Global attention for long-range dependencies

Each level operates at different strides and window sizes, creating a hierarchical structure that efficiently covers the entire sequence.

### Pattern Structure

```python
from dilated_attention_pytorch import BlockSparseHierarchical, HierarchicalConfig

# Default 3-level hierarchy
config = HierarchicalConfig(
    level_configs=[
        {"stride": 64, "window_size": 256, "block_size": 64},     # Fine
        {"stride": 256, "window_size": 1024, "block_size": 128},  # Medium
        {"stride": 1024, "window_size": -1, "block_size": 256},   # Coarse
    ]
)
```

### Parameters Explained

- **stride**: How often positions participate at this level (in tokens)
- **window_size**: Size of attention window (in tokens, -1 for global)
- **block_size**: Block size for sparse computation

## Usage Examples

### Basic Usage

```python
from dilated_attention_pytorch import create_hierarchical_attention

# Create hierarchical attention module
attention = create_hierarchical_attention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
)

# Use in forward pass
output = attention(query, key, value)
```

### Using Presets

```python
from dilated_attention_pytorch import get_hierarchical_presets

# Get available presets
presets = get_hierarchical_presets()

# Available presets:
# - "standard": Balanced 3-level hierarchy
# - "fine_grained": 4-level with more fine detail
# - "long_range": 3-level optimized for long sequences
# - "ultra_sparse": Extremely sparse for maximum efficiency

# Use a preset
from dilated_attention_pytorch import BlockSparseHierarchical

attention = BlockSparseHierarchical(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    hierarchical_config=presets["long_range"],
)
```

### Custom Configuration

```python
# Create custom hierarchy for specific needs
custom_config = HierarchicalConfig(
    level_configs=[
        # Very fine local attention
        {"stride": 32, "window_size": 128, "block_size": 32},
        # Regional attention  
        {"stride": 128, "window_size": 512, "block_size": 64},
        # Semi-global attention
        {"stride": 512, "window_size": 2048, "block_size": 128},
        # Full global attention
        {"stride": 2048, "window_size": -1, "block_size": 256},
    ]
)

attention = BlockSparseHierarchical(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    hierarchical_config=custom_config,
)
```

## Pattern Visualization

Hierarchical patterns can be visualized to understand their coverage:

```python
# Visualize the attention pattern
model = BlockSparseHierarchical(
    segment_lengths=[2048],
    dilation_rates=[1],
)

print(model.visualize_pattern(seq_len=512))
```

Output:
```
Hierarchical Attention Pattern:
Legend: ▪=Fine, ▫=Medium, ◦=Coarse
──────────
│▪▪▪▫▫▫▫▫│
│▪▪▪▪▫▫▫▫│
│▪▪▪▪▪▫▫▫│
│▫▪▪▪▪▪▫▫│
│▫▫▪▪▪▪▪▫│
│▫▫▫▪▪▪▪▪│
│▫▫▫▫▪▪▪▪│
│▫▫▫▫▫▪▪▪│
──────────
Sparsity: 87.5%
Active blocks: 32/256
```

## Pattern Statistics

Get detailed statistics about a hierarchical pattern:

```python
stats = model.get_pattern_stats(seq_len=4096)
print(f"Sparsity: {stats['sparsity']:.1%}")
print(f"Active blocks: {stats['active_blocks']}/{stats['total_blocks']}")
print("\nLevel breakdown:")
for level in stats['levels']:
    print(f"  Level {level['level']}: {level['active_positions']} positions, "
          f"{level['coverage']:.1%} coverage")
```

## Performance Characteristics

### Memory Efficiency

Hierarchical patterns achieve high sparsity while maintaining good coverage:
- Typical sparsity: 85-95%
- Memory reduction: 80-90% vs dense attention
- Adjustable based on configuration

### Computational Efficiency

- Fewer active blocks than dense patterns
- Better cache locality than random sparse patterns
- Scales well with sequence length

### Quality Trade-offs

- Fine level captures local dependencies
- Medium level captures regional context
- Coarse level maintains global coherence
- Combined levels provide comprehensive coverage

## Best Practices

### 1. Choosing Stride Values

- Strides should increase geometrically (e.g., 64, 256, 1024)
- Align with natural boundaries in your data
- Consider computational block sizes

### 2. Window Size Selection

- Fine level: 2-4x the typical local dependency range
- Medium level: Cover typical paragraph/section size
- Coarse level: Often global (-1) or very large

### 3. Number of Levels

- 3 levels work well for most sequences
- Add more levels for very long sequences (>100K tokens)
- Each level adds computational cost

### 4. Block Size Optimization

- Match GPU architecture (64 or 128 for most GPUs)
- Larger blocks = better efficiency but coarser granularity
- Can vary by level for optimal performance

## Integration with Dilated Attention

Hierarchical patterns work seamlessly with dilated attention:

```python
attention = BlockSparseHierarchical(
    segment_lengths=[2048, 4096, 8192, 16384],
    dilation_rates=[1, 2, 4, 8],
    hierarchical_config=custom_config,
)
```

This combines:
- Multi-scale hierarchical coverage
- Dilated attention's efficiency
- Adaptive segment processing

## Comparison with Other Patterns

| Pattern Type | Sparsity | Local Coverage | Global Coverage | Best For |
|-------------|----------|----------------|-----------------|-----------|
| Local Window | High | Excellent | Poor | Short sequences |
| Dilated Sparse | Medium | Good | Good | Medium sequences |
| Hierarchical | High | Good | Good | Long sequences |
| Global-Local | Medium | Good | Excellent | Mixed workloads |

## Advanced Usage

### Dynamic Level Selection

```python
def create_adaptive_hierarchy(seq_len):
    """Create hierarchy adapted to sequence length."""
    if seq_len < 1024:
        # Single level for short sequences
        return HierarchicalConfig(
            level_configs=[
                {"stride": 1, "window_size": seq_len, "block_size": 64},
            ]
        )
    elif seq_len < 8192:
        # Two levels for medium sequences
        return HierarchicalConfig(
            level_configs=[
                {"stride": 64, "window_size": 256, "block_size": 64},
                {"stride": 512, "window_size": -1, "block_size": 128},
            ]
        )
    else:
        # Full hierarchy for long sequences
        return get_hierarchical_presets()["long_range"]
```

### Combining with Ring Attention

For extreme sequence lengths, combine with Ring Attention:

```python
attention = BlockSparseHierarchical(
    segment_lengths=[8192, 16384, 32768],
    dilation_rates=[1, 2, 4],
    hierarchical_config=config,
    ring_size=8,  # Enable ring attention
)
```

## Troubleshooting

### High Memory Usage

If memory usage is too high:
1. Increase stride values
2. Reduce window sizes
3. Use fewer levels
4. Enable gradient checkpointing

### Poor Performance

If performance is suboptimal:
1. Check sparsity level (aim for >80%)
2. Align block sizes with GPU architecture
3. Use fp16/bf16 precision
4. Enable pattern caching

### Insufficient Coverage

If missing important dependencies:
1. Add intermediate levels
2. Increase window sizes
3. Reduce strides for critical levels
4. Consider adaptive patterns

## Future Developments

Upcoming features for hierarchical patterns:
- Content-adaptive level selection
- Learned hierarchy parameters
- Hardware-specific optimizations
- Integration with Flash Attention 3

## Conclusion

Hierarchical attention patterns provide an effective way to balance computational efficiency with attention coverage. By combining multiple scales of attention, they capture both local and global dependencies while maintaining high sparsity. The flexible configuration system allows adaptation to specific use cases and sequence lengths.