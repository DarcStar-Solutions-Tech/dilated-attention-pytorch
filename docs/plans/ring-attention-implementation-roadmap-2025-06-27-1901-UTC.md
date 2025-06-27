# Ring Attention Implementation Roadmap

**Date**: 2025-06-27 19:01 UTC  
**Priority**: High - Core functionality broken

## Quick Start: Minimum Viable Fix

### Step 1: Create Working Single-GPU Demo (Day 1)

```python
# File: dilated_attention_pytorch/ring_attention_correct.py

class RingAttentionCorrect(nn.Module):
    """Minimal correct Ring Attention implementation."""
    
    def forward(self, q, k, v, ring_size=4):
        b, n, h, d = q.shape
        chunk_size = n // ring_size
        
        # Keep full Q (this is the KEY difference!)
        output = torch.zeros_like(q)
        
        # Process K/V chunks sequentially
        for i in range(ring_size):
            # Get chunk
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            
            # ALL queries attend to this chunk
            scores = torch.matmul(q, k_chunk.transpose(-2, -1)) / math.sqrt(d)
            attn = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn, v_chunk)
            
            # Accumulate
            output += chunk_output
            
            # FREE MEMORY (this gives the benefit!)
            del k_chunk, v_chunk, scores, attn, chunk_output
            if q.is_cuda:
                torch.cuda.empty_cache()
        
        return output
```

### Step 2: Prove Memory Savings (Day 1)

```python
# File: benchmarks/prove_ring_attention_memory.py

def measure_memory_usage():
    """Prove that Ring Attention saves memory."""
    
    seq_lengths = [1024, 2048, 4096, 8192, 16384]
    ring_sizes = [1, 2, 4, 8, 16]
    
    results = []
    for seq_len in seq_lengths:
        for ring_size in ring_sizes:
            torch.cuda.reset_peak_memory_stats()
            
            # Create minimal tensors
            q = torch.randn(1, seq_len, 8, 64, device='cuda', dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            
            # Run Ring Attention
            module = RingAttentionCorrect()
            output = module(q, k, v, ring_size)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            results.append({
                'seq_len': seq_len,
                'ring_size': ring_size,
                'memory_gb': peak_memory,
                'memory_per_token_mb': peak_memory * 1024 / seq_len
            })
            
            print(f"Seq {seq_len}, Ring {ring_size}: {peak_memory:.3f}GB")
    
    # Plot results showing memory reduction
    plot_memory_scaling(results)
```

### Step 3: Fix Multi-GPU Implementation (Day 2-3)

```python
# File: dilated_attention_pytorch/ring_dilated_attention_v2.py

class RingDilatedAttentionV2(nn.Module):
    """Fixed Ring Attention with proper architecture."""
    
    def __init__(self, segment_lengths, dilation_rates, ring_size=None):
        super().__init__()
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.ring_size = ring_size or dist.get_world_size() if dist.is_initialized() else 1
    
    def forward(self, q, k, v, is_causal=False):
        if not dist.is_initialized() or self.ring_size == 1:
            return self._single_gpu_forward(q, k, v, is_causal)
        else:
            return self._multi_gpu_forward(q, k, v, is_causal)
    
    def _multi_gpu_forward(self, q, k, v, is_causal):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        b, n, h, d = q.shape
        
        # CRITICAL: Each GPU keeps FULL query!
        q_local = q  # No slicing!
        
        # Chunk K/V
        chunk_size = n // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size if rank < world_size - 1 else n
        
        # Initial K/V chunk for this GPU
        k_local = k[:, start_idx:end_idx].contiguous()
        v_local = v[:, start_idx:end_idx].contiguous()
        
        # Pad to uniform size
        if end_idx - start_idx < chunk_size:
            pad_size = chunk_size - (end_idx - start_idx)
            k_local = F.pad(k_local, (0, 0, 0, 0, 0, pad_size))
            v_local = F.pad(v_local, (0, 0, 0, 0, 0, pad_size))
        
        # Output accumulator
        output = torch.zeros_like(q_local)
        
        # Ring iterations
        for step in range(world_size):
            # Which chunk are we processing?
            source_rank = (rank - step) % world_size
            chunk_start = source_rank * chunk_size
            
            # Compute attention: ALL Q vs current K/V chunk
            chunk_out = self._compute_attention_chunk(
                q_local, k_local, v_local, 
                chunk_offset=chunk_start,
                is_causal=is_causal
            )
            output += chunk_out
            
            # Rotate K/V for next iteration
            if step < world_size - 1:
                k_local = self._ring_sendrecv(k_local)
                v_local = self._ring_sendrecv(v_local)
        
        return output
    
    def _ring_sendrecv(self, tensor):
        """Rotate tensor through ring."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size
        
        return torch.distributed.sendrecv(tensor, recv_rank, send_rank)
```

## Testing Strategy

### 1. Unit Tests (Day 3)

```python
# File: tests/test_ring_attention_v2.py

def test_ring_attention_correctness():
    """Verify output matches standard attention."""
    seq_len = 1024
    ring_sizes = [1, 2, 4, 8]
    
    # Standard attention reference
    q = torch.randn(2, seq_len, 8, 64)
    k, v = torch.randn_like(q), torch.randn_like(q)
    
    standard = F.scaled_dot_product_attention(q, k, v)
    
    for ring_size in ring_sizes:
        ring_module = RingAttentionCorrect()
        ring_output = ring_module(q, k, v, ring_size)
        
        assert torch.allclose(ring_output, standard, rtol=1e-3, atol=1e-4)
        print(f"✓ Ring size {ring_size} matches standard attention")

def test_memory_scaling():
    """Verify memory scales as O(n/ring_size)."""
    # Implementation here
```

### 2. Integration Tests (Day 4)

```python
def test_with_dilated_attention():
    """Test Ring Attention with dilated patterns."""
    # Implementation here

def test_billion_token_capability():
    """Demonstrate billion tokens with simulation."""
    # Implementation here
```

## Migration Plan

### Phase 1: Add New Implementation (Week 1)
1. Add `RingAttentionCorrect` as new class
2. Add comprehensive tests
3. Add benchmarks proving memory savings
4. Document usage

### Phase 2: Deprecate Old Implementation (Week 2)
1. Add deprecation warnings to old `RingDilatedAttention`
2. Update factory functions to use new implementation
3. Update examples and documentation
4. Release as minor version bump

### Phase 3: Remove Old Implementation (Month 2)
1. Remove deprecated code
2. Rename new implementation to `RingDilatedAttention`
3. Major version bump

## Success Metrics

1. **Memory Reduction**: 
   - Ring size 2: 40% reduction
   - Ring size 4: 70% reduction
   - Ring size 8: 85% reduction

2. **Performance**:
   - Single GPU: < 1.5x overhead
   - Multi-GPU: Near-linear scaling

3. **Max Sequence Length**:
   - Single GPU: 500K+ tokens
   - 8 GPUs: 4M+ tokens
   - 64 GPUs: 32M+ tokens

## Immediate Actions

1. **Today**: Implement `RingAttentionCorrect` (2 hours)
2. **Today**: Create memory scaling proof (1 hour)
3. **Tomorrow**: Fix multi-GPU version (4 hours)
4. **Day 3**: Complete test suite (3 hours)
5. **Day 4**: Integration and benchmarks (4 hours)

## Code Review Checklist

- [ ] Q tensor is NEVER sliced/divided
- [ ] K/V chunks are properly sized (n/ring_size)
- [ ] Memory is freed after each chunk
- [ ] Output accumulation is correct
- [ ] Causal masking accounts for chunk offset
- [ ] Gradients flow correctly
- [ ] Memory profiling shows O(n/ring_size) scaling
- [ ] Tests pass with various ring sizes

## Conclusion

This roadmap provides a clear path to fix Ring Attention. The key insight is that **queries must never be divided** - they must be replicated on all devices while only K/V are chunked. By following this plan, we can achieve true billion-token capability within a week.