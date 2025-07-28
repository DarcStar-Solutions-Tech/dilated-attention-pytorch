# Hierarchical Block-Sparse Variant Removal

## Date: 2025-07-07

### What was removed:
- `BlockSparseHierarchical` class from `block_sparse_hierarchical.py`
- `HierarchicalConfig` and related presets
- `create_hierarchical_block_sparse()` factory function
- Test file `test_block_sparse_hierarchical.py`
- Example file `hierarchical_dilated_attention_example.py`

### Why it was removed:
The hierarchical variant had fundamental design flaws that made it perform worse than simple sparse patterns:

1. **Poor Memory Efficiency**: Despite claiming 91% sparsity, it used 8.9x more memory than 99% sparse patterns
2. **Overlapping Coverage**: Multiple hierarchical levels covered the same positions, creating redundancy
3. **Bad Default Configurations**: Some presets achieved 100% density (no sparsity at all!)
4. **Low Maximum Sequence Length**: Only achieved 16K tokens vs 131K for simple sparse patterns

### Migration Guide:
If you were using the hierarchical variant, replace it with the base variant using dilated_sparse pattern:

```python
# OLD (removed):
from dilated_attention_pytorch import BlockSparseHierarchical, HierarchicalConfig

model = BlockSparseHierarchical(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    hierarchical_config=HierarchicalConfig()
)

# NEW (recommended):
from dilated_attention_pytorch import create_block_sparse_attention

model = create_block_sparse_attention(
    variant="base",
    segment_lengths=[2048],
    dilation_rates=[1],
    sparse_config=SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.01,  # 99% sparse
        dilation_rates=[1, 2, 4, 8]  # Multi-scale coverage
    )
)
```

The dilated_sparse pattern provides better multi-scale coverage with 8-10x less memory usage.

### Error Handling:
Attempting to use `variant="hierarchical"` will now raise a helpful error:
```
ValueError: Hierarchical variant has been removed due to poor memory efficiency. 
Use 'base' variant with pattern_type='dilated_sparse' and sparsity_ratio=0.01-0.05 
(95-99% sparse) for better multi-scale coverage.
```