# Naming Conventions Guide

This guide establishes consistent naming conventions for the Dilated Attention PyTorch project.

## File Naming

### Module Files

1. **Concept Separation**: Use underscores to separate distinct concepts
   - ✅ `block_sparse_attention.py`
   - ❌ `blockSparseAttention.py` or `blocksparseattention.py`

2. **Concept Ordering**: Follow hierarchical ordering from most specific to least
   - ✅ `block_sparse_ring_distributed_attention.py` (block-sparse → ring → distributed)
   - ❌ `distributed_ring_block_sparse_attention.py`

3. **Standard Prefixes**:
   - `block_sparse_` - For block-sparse implementations
   - `ring_` - For ring attention implementations
   - `distributed_` - For distributed training features
   - `utils/` - For utility modules
   - `core/` - For core shared components

### Configuration Files

- Use `_config.py` suffix for configuration modules
- Example: `distributed_sparse_config.py`

### Test Files

- Mirror the source file name with `test_` prefix
- Example: `test_block_sparse_attention.py` for `block_sparse_attention.py`

## Class Naming

1. **PascalCase**: Use PascalCase for all class names
   - ✅ `BlockSparseAttention`
   - ❌ `block_sparse_attention` or `blockSparseAttention`

2. **Descriptive Names**: Include all relevant concepts
   - ✅ `BlockSparseRingDistributedDilatedAttention`
   - ❌ `BSRDAttention` (too abbreviated)

3. **Base Classes**: Use `Base` prefix for abstract classes
   - ✅ `BaseDilatedAttention`
   - ❌ `DilatedAttentionBase` or `AbstractDilatedAttention`

## Function and Method Naming

1. **snake_case**: Use snake_case for all functions and methods
   - ✅ `create_dilated_attention_pattern()`
   - ❌ `createDilatedAttentionPattern()` or `CreateDilatedAttentionPattern()`

2. **Verb Prefixes**: Start with action verbs
   - `create_` - For factory functions
   - `compute_` - For calculation functions
   - `apply_` - For transformation functions
   - `validate_` - For validation functions
   - `_internal_` - For private implementation details

## Variable Naming

1. **snake_case**: Use snake_case for all variables
   - ✅ `attention_weights`
   - ❌ `attentionWeights` or `AttentionWeights`

2. **Descriptive Names**: Avoid single letters except in mathematical contexts
   - ✅ `batch_size`, `sequence_length`, `hidden_dim`
   - ❌ `bs`, `sl`, `hd` (except in tight loops)
   - ✓ `q`, `k`, `v` (acceptable for query, key, value in attention)

3. **Private Variables**: Use leading underscore
   - ✅ `self._internal_buffer`
   - ❌ `self.__internal_buffer` (double underscore)

## Constants

1. **UPPER_SNAKE_CASE**: Use for module-level constants
   - ✅ `DEFAULT_SEGMENT_LENGTH = 2048`
   - ❌ `defaultSegmentLength` or `default_segment_length`

2. **Grouped Constants**: Use classes or enums for related constants
   ```python
   class AttentionBackend(Enum):
       FLASH = "flash"
       XFORMERS = "xformers"
       PYTORCH = "pytorch"
   ```

## Type Annotations

1. **Always Use Type Hints**: For function signatures and class attributes
   ```python
   def create_attention_pattern(
       seq_len: int,
       num_heads: int,
       pattern_type: str = "dilated"
   ) -> torch.Tensor:
   ```

2. **Import Types**: Use appropriate imports
   ```python
   from typing import Optional, List, Dict, Tuple, Union
   from collections.abc import Sequence
   ```

## Module Organization

1. **Import Order**:
   ```python
   # Standard library
   import os
   import sys
   
   # Third-party
   import torch
   import numpy as np
   
   # Local imports
   from .core import BaseDilatedAttention
   from .utils import create_pattern
   ```

2. **Module Structure**:
   ```python
   """Module docstring."""
   
   # Imports
   # Constants
   # Helper functions
   # Main classes
   # Factory functions
   # Module exports (__all__)
   ```

## Examples of Correct Naming

### File Names
- `block_sparse_attention.py`
- `ring_dilated_attention.py`
- `distributed_memory_optimization.py`
- `sparse_pattern_generator.py`

### Class Names
- `BlockSparseAttention`
- `RingDilatedAttention`
- `DistributedMemoryOptimizer`
- `HierarchicalSparsePatternGenerator`

### Function Names
- `create_dilated_mask()`
- `compute_attention_scores()`
- `apply_ring_communication()`
- `validate_sequence_length()`

## Migration Plan

To standardize existing code:

1. **New Code**: Follow these conventions immediately
2. **Existing Code**: Gradually refactor when making other changes
3. **Public API**: Maintain backward compatibility with deprecation warnings
4. **Internal Code**: Can be renamed more aggressively

## Enforcement

1. Use `ruff` and `mypy` for automatic checking
2. Configure pre-commit hooks to enforce conventions
3. Document exceptions in code comments when necessary