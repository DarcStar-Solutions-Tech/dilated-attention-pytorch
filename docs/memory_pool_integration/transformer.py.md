
Memory Pool Integration Guide for transformer.py
============================================================

Module: DilatedTransformerEncoderLayer, DilatedTransformerDecoderLayer
Priority: MEDIUM

Large tensors that would benefit from pooling:
  - self-attention output
  - cross-attention output
  - feedforward intermediate

Integration Steps:
1. Add memory pool imports at the top of the file
2. Add enable_memory_pool parameter to __init__ (default=False)
3. Initialize self._memory_pool in __init__ when enabled
4. Add _allocate_tensor and _deallocate_tensor methods
5. Replace large tensor allocations with _allocate_tensor calls
6. Add corresponding _deallocate_tensor calls in cleanup/del methods

Example locations to update:

  - In forward(): Use for attention outputs
  - In feedforward layers: Use for intermediate activations
  - Consider reusing buffers across layers

Found 0 potential tensor allocations:
