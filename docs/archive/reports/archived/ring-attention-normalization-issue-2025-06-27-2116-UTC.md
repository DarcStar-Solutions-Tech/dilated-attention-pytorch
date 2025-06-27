# Ring Attention Normalization Issue Report

**Date**: 2025-06-27 21:16 UTC  
**Issue**: Incorrect softmax normalization in Ring Attention implementations

## Executive Summary

Multiple Ring Attention implementations in the codebase have a fundamental flaw: they apply softmax to each K/V chunk independently, which breaks the normalization constraint that attention weights should sum to 1.0 across all positions. This causes incorrect outputs when `ring_size > 1`.

## Issue Details

### Root Cause

When computing attention with chunked K/V:
1. Each chunk computes `softmax(Q @ K_chunk^T)`
2. Since softmax normalizes to sum=1.0 for each chunk independently
3. The total attention weights across all chunks sum to `ring_size` instead of 1.0
4. This causes the output to be scaled incorrectly by a factor of `ring_size`

### Affected Files

1. `ring_dilated_attention_v2.py` - Delegates to RingAttentionCorrect which has the issue
2. `ring_attention_correct.py` - Applies softmax per chunk (line 116)
3. Potentially other ring attention implementations in the codebase

### Test Results

```
Ring size 1: max_diff=0.00e+00  ✓ (correct - no chunking)
Ring size 2: max_diff=2.97e-01  ✗ (incorrect - ~2x scaling)
Ring size 4: max_diff=3.88e-01  ✗ (incorrect - ~4x scaling)
```

The output sum roughly scales with ring_size:
- Expected: 88.09
- Ring size 2: 178.68 (~2x)
- Ring size 4: ~352 (would be ~4x)

## Correct Algorithm

The correct Ring Attention algorithm uses **online softmax** with running statistics:

```python
# Initialize running max and sum
m = -inf  # running max
l = 0     # running sum

# Process each chunk
for chunk in chunks:
    scores_chunk = Q @ K_chunk^T
    
    # Update running statistics
    m_new = max(m, max(scores_chunk))
    l_new = exp(m - m_new) * l + sum(exp(scores_chunk - m_new))
    
    # Update output with proper normalization
    output = output * (exp(m - m_new) * l / l_new) + 
             (exp(scores_chunk - m_new) / l_new) @ V_chunk
    
    m = m_new
    l = l_new
```

## Performance Impact

The correct algorithm:
- Maintains O(n/ring_size) memory complexity for K/V
- Requires 2 passes through chunks or online computation
- Adds minimal computational overhead
- Produces mathematically correct results

## Recommendations

1. **Immediate**: Fix `RingAttentionCorrect` to use online softmax
2. **Short-term**: Update all ring attention implementations 
3. **Long-term**: Add numerical tests to verify attention weight normalization
4. **Consider**: Using FlashAttention's ring attention implementation as reference

## Test Case

A simple test to verify correct normalization:
```python
# Standard attention
output_std = softmax(Q @ K^T) @ V

# Ring attention with chunks
output_ring = ring_attention(Q, K, V, ring_size=4)

# Should match within numerical precision
assert torch.allclose(output_std, output_ring, atol=1e-5)
```

## References

1. FlashAttention paper: Online softmax algorithm (Algorithm 1)
2. Ring Attention paper: Proper handling of normalization across devices
3. Test script: `test_correct_ring_attention.py` demonstrates the fix