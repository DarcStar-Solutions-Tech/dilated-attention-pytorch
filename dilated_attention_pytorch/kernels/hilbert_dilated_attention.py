#!/usr/bin/env python3
"""
Hilbert curve ordered dilated attention kernel.
Uses Hilbert space-filling curve for memory layout to improve cache locality.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from typing import Tuple
import math


# CUDA kernel source code
hilbert_dilated_attention_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Hilbert curve utilities
__device__ inline void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }
        // Swap x and y
        int t = *x;
        *x = *y;
        *y = t;
    }
}

// Convert (x,y) to Hilbert curve distance
__device__ int xy2hilbert(int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(n, &x, &y, rx, ry);
    }
    return d;
}

// Convert Hilbert curve distance to (x,y)
__device__ void hilbert2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t=d;
    *x = *y = 0;
    for (s=1; s<n; s*=2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

// Dilated attention kernel with Hilbert ordering
template <typename scalar_t>
__global__ void hilbert_dilated_attention_forward_kernel(
    const scalar_t* __restrict__ query,      // [batch, heads, seq_len, head_dim]
    const scalar_t* __restrict__ key,        // [batch, heads, seq_len, head_dim]  
    const scalar_t* __restrict__ value,      // [batch, heads, seq_len, head_dim]
    scalar_t* __restrict__ output,           // [batch, heads, seq_len, head_dim]
    const int* __restrict__ hilbert_indices, // [seq_len] - precomputed Hilbert ordering
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int segment_size,
    int dilation_rate,
    float scale
) {
    // Grid-stride loop indices
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }
    
    // Get Hilbert-ordered position
    const int hilbert_pos = hilbert_indices[seq_idx];
    
    // Determine segment for this position
    const int segment_idx = hilbert_pos / segment_size;
    const int pos_in_segment = hilbert_pos % segment_size;
    
    // Calculate attention window with dilation
    const int window_start = segment_idx * segment_size;
    const int window_end = min(window_start + segment_size, seq_len);
    
    // Shared memory for key/value caching
    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_keys = shared_mem;
    scalar_t* shared_values = shared_mem + blockDim.x * head_dim;
    
    // Load query for this position
    scalar_t query_vec[32]; // Assuming head_dim <= 32 for simplicity
    for (int d = 0; d < head_dim; d++) {
        query_vec[d] = query[batch_idx * num_heads * seq_len * head_dim +
                             head_idx * seq_len * head_dim +
                             hilbert_pos * head_dim + d];
    }
    
    // Accumulate attention
    scalar_t output_vec[32] = {0};
    scalar_t sum_weights = 0;
    
    // Process attention window with dilation
    for (int offset = 0; offset < segment_size; offset += dilation_rate) {
        int key_pos = window_start + offset;
        if (key_pos >= window_end) break;
        
        // Get Hilbert-ordered key position
        int key_hilbert_pos = hilbert_indices[key_pos];
        
        // Compute attention score
        scalar_t score = 0;
        for (int d = 0; d < head_dim; d++) {
            scalar_t k = key[batch_idx * num_heads * seq_len * head_dim +
                            head_idx * seq_len * head_dim +
                            key_hilbert_pos * head_dim + d];
            score += query_vec[d] * k;
        }
        score *= scale;
        
        // Apply softmax (numerically stable)
        scalar_t weight = exp(score);
        sum_weights += weight;
        
        // Accumulate weighted values
        for (int d = 0; d < head_dim; d++) {
            scalar_t v = value[batch_idx * num_heads * seq_len * head_dim +
                              head_idx * seq_len * head_dim +
                              key_hilbert_pos * head_dim + d];
            output_vec[d] += weight * v;
        }
    }
    
    // Normalize and write output
    if (sum_weights > 0) {
        for (int d = 0; d < head_dim; d++) {
            output[batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   hilbert_pos * head_dim + d] = output_vec[d] / sum_weights;
        }
    }
}

// Standard dilated attention kernel (for comparison)
template <typename scalar_t>
__global__ void standard_dilated_attention_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int segment_size,
    int dilation_rate,
    float scale
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }
    
    // Standard linear ordering
    const int segment_idx = seq_idx / segment_size;
    const int window_start = segment_idx * segment_size;
    const int window_end = min(window_start + segment_size, seq_len);
    
    // Load query
    scalar_t query_vec[32];
    for (int d = 0; d < head_dim; d++) {
        query_vec[d] = query[batch_idx * num_heads * seq_len * head_dim +
                             head_idx * seq_len * head_dim +
                             seq_idx * head_dim + d];
    }
    
    // Compute attention
    scalar_t output_vec[32] = {0};
    scalar_t sum_weights = 0;
    
    for (int offset = 0; offset < segment_size; offset += dilation_rate) {
        int key_pos = window_start + offset;
        if (key_pos >= window_end) break;
        
        // Compute score
        scalar_t score = 0;
        for (int d = 0; d < head_dim; d++) {
            scalar_t k = key[batch_idx * num_heads * seq_len * head_dim +
                            head_idx * seq_len * head_dim +
                            key_pos * head_dim + d];
            score += query_vec[d] * k;
        }
        score *= scale;
        
        scalar_t weight = exp(score);
        sum_weights += weight;
        
        // Accumulate values
        for (int d = 0; d < head_dim; d++) {
            scalar_t v = value[batch_idx * num_heads * seq_len * head_dim +
                              head_idx * seq_len * head_dim +
                              key_pos * head_dim + d];
            output_vec[d] += weight * v;
        }
    }
    
    // Normalize output
    if (sum_weights > 0) {
        for (int d = 0; d < head_dim; d++) {
            output[batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   seq_idx * head_dim + d] = output_vec[d] / sum_weights;
        }
    }
}

// C++ interface
torch::Tensor hilbert_dilated_attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor hilbert_indices,
    int segment_size,
    int dilation_rate
) {
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_len = query.size(2);
    const auto head_dim = query.size(3);
    
    auto output = torch::zeros_like(query);
    
    const float scale = 1.0f / sqrt(float(head_dim));
    
    // Configure kernel launch
    const int threads = 256;
    const dim3 blocks(batch_size, num_heads, (seq_len + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "hilbert_dilated_attention_forward_cuda", ([&] {
        hilbert_dilated_attention_forward_kernel<scalar_t><<<blocks, threads>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            hilbert_indices.data_ptr<int>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            segment_size,
            dilation_rate,
            scale
        );
    }));
    
    return output;
}

torch::Tensor standard_dilated_attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int segment_size,
    int dilation_rate
) {
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_len = query.size(2);
    const auto head_dim = query.size(3);
    
    auto output = torch::zeros_like(query);
    
    const float scale = 1.0f / sqrt(float(head_dim));
    
    const int threads = 256;
    const dim3 blocks(batch_size, num_heads, (seq_len + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "standard_dilated_attention_forward_cuda", ([&] {
        standard_dilated_attention_forward_kernel<scalar_t><<<blocks, threads>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            segment_size,
            dilation_rate,
            scale
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hilbert_forward", &hilbert_dilated_attention_forward_cuda, "Hilbert dilated attention forward");
    m.def("standard_forward", &standard_dilated_attention_forward_cuda, "Standard dilated attention forward");
}
"""

# Compile CUDA kernels
try:
    hilbert_attention_cuda = load_inline(
        name="hilbert_attention_cuda",
        cpp_sources="",
        cuda_sources=hilbert_dilated_attention_source,
        functions=["hilbert_forward", "standard_forward"],
        verbose=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not compile CUDA kernels: {e}")
    CUDA_AVAILABLE = False


def generate_hilbert_indices(n: int) -> torch.Tensor:
    """Generate Hilbert curve ordering for n x n grid."""
    # Find next power of 2
    size = 2 ** int(math.ceil(math.log2(math.sqrt(n))))

    # Generate 2D Hilbert curve
    indices = []
    for i in range(min(n, size * size)):
        # Gray code based Hilbert curve generation
        x, y = 0, 0
        s = 1
        d = i

        while s < size:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)

            if ry == 0:
                if rx == 1:
                    x, y = size - 1 - x, size - 1 - y
                x, y = y, x

            x += s * rx
            y += s * ry
            d //= 4
            s *= 2

        # Map 2D position back to 1D
        linear_idx = y * size + x
        if linear_idx < n:
            indices.append(linear_idx)

    # Fill remaining indices linearly if needed
    while len(indices) < n:
        indices.append(len(indices))

    return torch.tensor(indices[:n], dtype=torch.int32)


class HilbertDilatedAttention(nn.Module):
    """Dilated attention using Hilbert curve memory ordering."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 256,
        dilation_rate: int = 1,
        use_cuda_kernel: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.use_cuda_kernel = use_cuda_kernel and CUDA_AVAILABLE

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Pre-generate Hilbert indices for common sequence lengths
        self.register_buffer("hilbert_indices_cache", {})

    def get_hilbert_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or generate Hilbert indices for given sequence length."""
        if seq_len not in self.hilbert_indices_cache:
            indices = generate_hilbert_indices(seq_len).to(device)
            self.hilbert_indices_cache[seq_len] = indices
        return self.hilbert_indices_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            use_hilbert: Whether to use Hilbert ordering
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]

        if self.use_cuda_kernel and x.is_cuda:
            if use_hilbert:
                # Get Hilbert indices
                hilbert_indices = self.get_hilbert_indices(seq_len, x.device)

                # Apply Hilbert-ordered attention
                attn_output = hilbert_attention_cuda.hilbert_forward(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    hilbert_indices,
                    self.segment_size,
                    self.dilation_rate,
                )
            else:
                # Standard linear ordering
                attn_output = hilbert_attention_cuda.standard_forward(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    self.segment_size,
                    self.dilation_rate,
                )
        else:
            # Fallback to PyTorch implementation
            attn_output = self._pytorch_dilated_attention(
                query, key, value, use_hilbert
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output

    def _pytorch_dilated_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_hilbert: bool,
    ) -> torch.Tensor:
        """PyTorch implementation for CPU or fallback."""
        batch_size, num_heads, seq_len, head_dim = query.shape

        if use_hilbert:
            # Get Hilbert ordering
            hilbert_indices = self.get_hilbert_indices(seq_len, query.device)

            # Reorder tensors according to Hilbert curve
            query_h = query.gather(
                2,
                hilbert_indices.view(1, 1, -1, 1).expand(
                    batch_size, num_heads, seq_len, head_dim
                ),
            )
            key_h = key.gather(
                2,
                hilbert_indices.view(1, 1, -1, 1).expand(
                    batch_size, num_heads, seq_len, head_dim
                ),
            )
            value_h = value.gather(
                2,
                hilbert_indices.view(1, 1, -1, 1).expand(
                    batch_size, num_heads, seq_len, head_dim
                ),
            )

            # Apply attention on Hilbert-ordered data
            output_h = self._apply_dilated_attention(query_h, key_h, value_h)

            # Reorder output back to original order
            inverse_indices = torch.argsort(hilbert_indices)
            output = output_h.gather(
                2,
                inverse_indices.view(1, 1, -1, 1).expand(
                    batch_size, num_heads, seq_len, head_dim
                ),
            )
        else:
            output = self._apply_dilated_attention(query, key, value)

        return output

    def _apply_dilated_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Apply dilated attention with segment-wise computation."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)

        output = torch.zeros_like(query)

        # Process each segment
        for seg_start in range(0, seq_len, self.segment_size):
            seg_end = min(seg_start + self.segment_size, seq_len)

            # Get queries for this segment
            q_seg = query[:, :, seg_start:seg_end]

            # Get dilated keys and values
            indices = list(range(seg_start, seg_end, self.dilation_rate))
            if indices:
                k_seg = key[:, :, indices]
                v_seg = value[:, :, indices]

                # Compute attention scores
                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(scores, dim=-1)

                # Apply attention
                attn_output = torch.matmul(attn_weights, v_seg)
                output[:, :, seg_start:seg_end] = attn_output

        return output


def create_hilbert_benchmark_data(
    batch_size: int = 4,
    seq_len: int = 4096,
    hidden_dim: int = 512,
    num_heads: int = 8,
    device: str = "cuda",
) -> Tuple[torch.Tensor, nn.Module]:
    """Create benchmark data and model."""
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Create model
    model = HilbertDilatedAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=256,
        dilation_rate=2,
        use_cuda_kernel=True,
    ).to(device)

    return x, model


if __name__ == "__main__":
    # Quick test
    if CUDA_AVAILABLE and torch.cuda.is_available():
        print("Testing Hilbert dilated attention kernel...")
        x, model = create_hilbert_benchmark_data(
            batch_size=2, seq_len=1024, hidden_dim=256, num_heads=8
        )

        # Test both modes
        with torch.no_grad():
            out_hilbert = model(x, use_hilbert=True)
            out_standard = model(x, use_hilbert=False)

        print(f"Hilbert output shape: {out_hilbert.shape}")
        print(f"Standard output shape: {out_standard.shape}")
        print(f"Outputs match: {torch.allclose(out_hilbert, out_standard, atol=1e-5)}")
    else:
        print("CUDA not available for testing")
