very good, but shouldn't # OOM Error Analysis - Multi-GPU Verification

## Root Cause

The OOM error occurred due to **massive temporary tensor allocations** during the softmax computation in ring attention.

## Specific Issue

In `_compute_chunk_attention_with_online_softmax` at line 924:
```python
running_sum.add_(torch.exp(scores - new_max).sum(dim=-1, keepdim=True))
```

This single line creates **3 large temporary tensors**:
1. `scores - new_max`: 1024 MB
2. `torch.exp(...)`: 1024 MB  
3. Result of sum: Smaller, but still allocated

## Why Multi-GPU Failed but Single-GPU Worked

### 1. **Larger Sequence Length**
- Multi-GPU test: `seq_length = 8192`
- Single-GPU test: `seq_length = 4096`
- Quadratic memory scaling: 8192² = 4x more memory than 4096²

### 2. **Attention Matrix Size**
For `[batch=1, heads=8, seq_len=8192, seq_len=8192]`:
- FP16: 1 × 8 × 8192 × 8192 × 2 bytes = **1024 MB per tensor**
- Multiple temporaries = **3+ GB just for intermediates**

### 3. **Process Overhead**
The GPU was shared by multiple processes:
- Process 3093175: 16.45 MB
- Process 4050264: 274.00 MB  
- Current process: 3.38 GB
- Only 592 MB free out of 7.88 GB total

### 4. **Memory Fragmentation**
- PyTorch allocated: 2.15 GB
- PyTorch reserved but unused: 1.04 GB
- Fragmentation prevents allocating large contiguous blocks

## Pascal GPU FP16 Complications

On Pascal GPUs, FP16 can actually **increase** memory pressure:

1. **Alignment requirements**: FP16 ops may need specific memory alignment
2. **Internal conversions**: Some ops convert to FP32 internally for stability
3. **Inefficient kernels**: Pascal's limited FP16 support may use more temporaries
4. **Numerical instability**: `exp()` in FP16 can overflow, requiring FP32 fallbacks

## Code Analysis

The problematic code pattern:
```python
# Creates massive temporary tensors
scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(d)  # [b,h,8192,8192] = 1GB

# Line 924 - THE KILLER: Creates 3 temporaries!
running_sum.add_(torch.exp(scores - new_max).sum(dim=-1, keepdim=True))
#                          ^---- temp1 (1GB) 
#                     ^--------- temp2 (1GB)
#                                      ^------- temp3 

# Line 930 - Another 1GB allocation
exp_scores = torch.exp(scores - new_max)  
```

## Solutions

### 1. **Use In-Place Operations**
```python
# Instead of: torch.exp(scores - new_max)
scores.sub_(new_max)  # In-place subtraction
scores.exp_()         # In-place exponential
```

### 2. **Chunked Softmax**
Process attention in smaller chunks:
```python
chunk_size = 2048  # Process 2K at a time instead of 8K
for i in range(0, seq_len, chunk_size):
    scores_chunk = scores[:, :, :, i:i+chunk_size]
    # Process chunk...
```

### 3. **Memory Pool Pre-allocation**
Pre-allocate buffers for temporaries and reuse them.

### 4. **FP32 for Attention Scores**
Use FP32 for attention computation even when model is FP16:
```python
scores = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1))
```

### 5. **Flash Attention**
Use Flash Attention which computes attention without materializing the full attention matrix.

## Immediate Workaround

For testing, reduce sequence length or batch size:
```python
# Instead of seq_length = 8192
seq_length = 4096  # or even 2048 for multi-GPU tests
```

## Key Insight

The **Pascal FP16 fix is even more critical** because:
1. FP16 doesn't save enough memory to offset the temporary allocations
2. FP16 may actually use MORE memory due to conversions
3. FP32 is both faster AND more memory-efficient on Pascal

This OOM reinforces that **FP32 should be the default for Pascal GPUs**.
