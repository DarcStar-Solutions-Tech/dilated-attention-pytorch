# Ring Dilated Attention with Hilbert Integration Guide

This guide explains the proper integration of Ring Attention, Dilated Attention, and Hilbert Space-Filling Curves (SFC) in the `RingDilatedAttentionHilbertProper` implementation.

## Overview

The implementation correctly combines three techniques:

1. **Ring Attention**: O(n) memory complexity through distributed communication
2. **Dilated Attention**: Multi-scale attention with different dilation rates
3. **Hilbert SFC**: Spatial locality preservation through space-filling curves

## Key Design Principles

### 1. Per-Segment Processing

The most critical aspect is that each technique is applied **per-segment**, not globally:

```python
# Process each segment independently
for seg_idx, (seg_len, dil_rate) in enumerate(zip(segment_lengths, dilation_rates)):
    # Extract segment
    segment = sequence[position:position+seg_len]
    
    # Apply Hilbert ordering to THIS segment only
    if use_hilbert:
        segment = apply_hilbert_ordering(segment)
    
    # Apply dilated attention within THIS segment
    output = compute_dilated_attention(segment, dilation_rate)
    
    # Reverse Hilbert ordering
    if use_hilbert:
        output = apply_inverse_hilbert_ordering(output)
```

### 2. Proper Ring Communication

Ring attention uses **isend/irecv** operations, NOT all_gather:

```python
# Correct ring communication pattern
for step in range(ring_size):
    # Compute attention for current chunk
    attn_out, attn_lse = compute_attention(q_local, k_chunk, v_chunk)
    
    # Accumulate with numerical stability
    accumulator.update(attn_out, attn_lse)
    
    # Pass K,V to next device in ring
    if step < ring_size - 1:
        k_chunk = all_ring_pass(ring_size, k_chunk)  # Uses isend/irecv
        v_chunk = all_ring_pass(ring_size, v_chunk)
```

### 3. Numerical Stability with LSE

The implementation uses Log-Sum-Exp (LSE) for numerically stable accumulation:

```python
class StableAttentionAccumulator:
    def update(self, new_output, new_lse):
        # Stable accumulation using LSE trick
        max_lse = torch.maximum(self.lse, new_lse)
        self.output = (
            self.output * torch.exp(self.lse - max_lse) + 
            new_output * torch.exp(new_lse - max_lse)
        )
        self.lse = torch.log(
            torch.exp(self.lse - max_lse) + 
            torch.exp(new_lse - max_lse)
        ) + max_lse
```

## Architecture Details

### Input/Output Format

- Input: `[batch_size, seq_len, embed_dim]`
- Output: `[batch_size, seq_len, embed_dim]`
- Internal: `[batch_size, num_heads, seq_len, head_dim]`

### Segment Configuration

Segments are processed in order with their corresponding dilation rates:

```python
segment_lengths = [2048, 4096, 8192]
dilation_rates = [1, 2, 4]
```

This creates a hierarchical attention pattern:
- First 2048 tokens: Full attention (dilation=1)
- Next 4096 tokens: Attend every 2nd position (dilation=2)  
- Next 8192 tokens: Attend every 4th position (dilation=4)

### Hilbert Ordering

Hilbert curves preserve spatial locality by mapping 2D space to 1D:

```python
# For each segment independently:
1. Map linear positions to 2D grid
2. Apply Hilbert curve traversal
3. Reorder segment based on Hilbert indices
4. Compute attention
5. Reverse reordering for output
```

## Usage Example

```python
from dilated_attention_pytorch.ring_dilated_attention_hilbert_proper import (
    RingDilatedAttentionHilbertProper
)

# Create model
model = RingDilatedAttentionHilbertProper(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1,
    use_hilbert=True,
    ring_size=8,  # For 8 GPUs
)

# Forward pass
x = torch.randn(batch_size, seq_len, embed_dim)
output = model(x, is_causal=True)
```

## Distributed Training

For multi-GPU ring attention:

```python
import torch.distributed as dist

# Initialize distributed environment
dist.init_process_group("nccl")

# Model automatically detects distributed setup
model = RingDilatedAttentionHilbertProper(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
)

# Each GPU processes its chunk of the sequence
output = model(x)
```

## Performance Characteristics

### Memory Complexity
- Standard Attention: O(n²)
- Ring Dilated Attention: O(n)
- Per-GPU memory: O(n/ring_size)

### Computation Complexity
- Standard Attention: O(n²)
- Dilated Attention: O(n² / average_dilation_rate)
- Ring overhead: O(ring_size) communication steps

### Benefits of Hilbert Ordering
- Better cache locality (up to 2x speedup)
- Preserves spatial relationships in data
- Applied per-segment to maintain local structure

## Common Pitfalls to Avoid

### 1. Global Hilbert Ordering
❌ **Wrong**: Applying Hilbert curve to entire sequence
```python
# This destroys the segment structure!
full_sequence = apply_hilbert_to_entire_sequence(x)
```

✅ **Correct**: Apply Hilbert per-segment
```python
for segment in segments:
    segment = apply_hilbert_to_segment(segment)
```

### 2. Using all_gather
❌ **Wrong**: Gathering all outputs at the end
```python
# This breaks O(n) memory complexity!
all_outputs = all_gather(local_output)
```

✅ **Correct**: Each GPU keeps its local output
```python
# Ring attention maintains distributed output
local_output = ring_attention(local_input)
```

### 3. Ignoring Numerical Stability
❌ **Wrong**: Direct accumulation
```python
output += new_attention_output
```

✅ **Correct**: LSE-based accumulation
```python
accumulator.update(new_output, new_lse)
```

## Testing and Validation

The implementation includes comprehensive tests:

```bash
# Run unit tests
pytest tests/test_ring_dilated_attention_hilbert_proper.py -v

# Run integration tests
python scripts/test_ring_hilbert_proper.py
```

Key test coverage:
- ✅ Gradient flow verification
- ✅ Numerical stability with extreme values
- ✅ Memory efficiency validation
- ✅ Distributed communication patterns
- ✅ Hilbert ordering effects
- ✅ Dilated attention patterns

## Conclusion

The `RingDilatedAttentionHilbertProper` implementation correctly combines:
- Ring attention for O(n) memory scaling
- Dilated attention for efficient long-range modeling
- Per-segment Hilbert ordering for spatial locality

This enables processing of extremely long sequences (millions of tokens) while maintaining high quality attention patterns and numerical stability.