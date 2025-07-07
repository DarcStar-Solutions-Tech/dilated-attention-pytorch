# Hierarchical Block-Sparse Variant: Performance Challenges

**Date**: 2025-07-07 11:55 UTC  
**Subject**: Why hierarchical achieves only 16K tokens vs 131K for simple sparse

## Executive Summary

The hierarchical variant achieves only **16,384 tokens** maximum sequence length compared to **131,072 tokens** for simple sparse patterns. The root cause: **hierarchical patterns are much denser than they appear**, using 8.9x more memory than 99% sparse patterns.

## Key Problems Identified

### 1. Misleading Sparsity Calculations

The hierarchical pattern claims "91.1% sparsity" at 16K tokens, but actually uses:
- **5,858 active blocks** (hierarchical)
- **655 active blocks** (99% sparse)
- **8.9x more blocks** than equivalent simple sparse!

### 2. Overlapping Coverage

The hierarchical approach creates overlapping attention patterns:

```
Level 0: Every token attends to 256-token window (stride=64)
Level 1: Every 256th token attends to 1024-token window  
Level 2: Every 1024th token attends globally
```

Problem: Many positions get covered multiple times by different levels, creating redundancy.

### 3. Poor Configurations

#### Default Configuration Issues:
- **stride=64** for Level 0 means every 64th position creates a 256-token window
- This alone creates 256 active positions × 4 blocks each = 1,024 blocks minimum
- Add Levels 1 & 2, and density explodes

#### "Fine-grained" Configuration Disaster:
- **stride=1** means EVERY position attends to its window
- Results in **100% density** (no sparsity at all!)
- Uses same memory as full attention

#### "Long-range" Configuration:
- Also achieves **100% density** due to stride=1
- Completely defeats the purpose of sparse attention

### 4. Memory Usage Comparison

For 16,384 tokens:
| Pattern Type | Active Blocks | Memory | Max Sequence |
|--------------|---------------|---------|--------------|
| 99% Sparse | 655 | 40MB | 131K tokens |
| 95% Sparse | 3,276 | 204MB | 131K tokens |
| Hierarchical Default | 5,858 | 366MB | 16K tokens |
| Hierarchical Fine | 65,536 | 4,096MB | <4K tokens |

### 5. Inefficient Pattern Generation

The hierarchical pattern generation:
1. Iterates through all levels
2. For each level, iterates through active positions
3. For each position, adds all blocks in its window
4. No deduplication of overlapping blocks

This creates a union of all level patterns, not an efficient hierarchical structure.

## Why It Fails at 32K Tokens

At 32,768 tokens:
- Total possible blocks: 262,144
- Hierarchical uses: **19,922 blocks** (7.6% density)
- Memory needed: **1,245 MB** just for attention
- This exceeds available GPU memory after other allocations

Compare to 99% sparse at 131K tokens:
- Uses only **1,720 blocks** (0.01% density)  
- Memory needed: **107 MB**
- 11.6x more efficient!

## Root Cause Analysis

### Conceptual Flaw
The implementation treats hierarchical attention as **additive** (union of all levels) rather than **selective** (different heads/positions use different levels).

### Implementation Issues
1. **No mutual exclusivity**: Positions can be active at multiple levels
2. **Poor default strides**: Too dense at each level
3. **Global attention too frequent**: Level 2 has many global positions
4. **Block size mismatches**: Different block sizes per level cause alignment issues

## Recommendations

### 1. Fix the Pattern Generation
Instead of union, use **disjoint** assignment:
- Assign specific heads to specific levels
- Or assign specific positions exclusively to one level
- Prevent overlap between levels

### 2. Better Default Configuration
```python
HierarchicalConfig(
    level_configs=[
        {"stride": 512, "window_size": 512, "block_size": 64},    # 0.2% positions
        {"stride": 2048, "window_size": 2048, "block_size": 64},  # 0.05% positions  
        {"stride": 8192, "window_size": -1, "block_size": 64},    # 0.01% positions
    ]
)
```

### 3. Alternative Approach
Consider the "dilated_sparse" pattern from simple sparse:
- Achieves multi-scale coverage
- Much more memory efficient
- Already works well

## Conclusion

The hierarchical variant fails because it's not actually sparse - it's a **dense pattern masquerading as sparse**. The overlapping windows and poor default configuration make it use more memory than even 90% sparse patterns, explaining why it achieves only 16K tokens.

**Recommendation**: Use simple sparse patterns with 95-99% sparsity instead of the current hierarchical implementation. They achieve better coverage with 8-10x less memory usage.