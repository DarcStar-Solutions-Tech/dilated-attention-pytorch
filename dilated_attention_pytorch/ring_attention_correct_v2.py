"""
Ring Attention Correct V2 - With proper online softmax normalization.

This implementation fixes the normalization issue in the original RingAttentionCorrect
by using online softmax to maintain correct attention weight normalization across chunks.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class RingAttentionCorrectV2:
    """
    Correct Ring Attention implementation with proper normalization.

    Key improvement: Uses online softmax to ensure attention weights
    sum to 1.0 across all positions, not per chunk.
    """

    def __init__(
        self,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.dropout = dropout
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ring_size: int = 4,
        is_causal: bool = False,
        return_memory_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Apply Ring Attention with correct normalization.

        Args:
            query: [batch, seq_len, num_heads, head_dim]
            key: [batch, seq_len, num_heads, head_dim]
            value: [batch, seq_len, num_heads, head_dim]
            ring_size: Number of chunks to split K/V into
            is_causal: Apply causal masking
            return_memory_stats: Return memory statistics

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
            Optional memory statistics
        """
        b, n, h, d = query.shape

        # Validate inputs
        assert key.shape == value.shape == query.shape
        assert n % ring_size == 0, (
            f"Sequence length {n} must be divisible by ring_size {ring_size}"
        )

        # Move to correct device/dtype
        query = query.to(device=self.device, dtype=self.dtype)
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)

        # Chunk size
        chunk_size = n // ring_size

        # Track memory if requested
        if return_memory_stats and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / (1024**2)

        # Initialize output and running statistics
        # Output will be accumulated in [b, h, n, d] format for efficiency
        output = torch.zeros(b, h, n, d, device=self.device, dtype=self.dtype)

        # Running max and sum for online softmax
        # Shape: [b, h, n, 1] to broadcast correctly
        running_max = torch.full(
            (b, h, n, 1), float("-inf"), device=self.device, dtype=self.dtype
        )
        running_sum = torch.zeros((b, h, n, 1), device=self.device, dtype=self.dtype)

        # Process each chunk
        for chunk_idx in range(ring_size):
            start_idx = chunk_idx * chunk_size
            end_idx = (chunk_idx + 1) * chunk_size

            # Get K/V chunk
            k_chunk = key[:, start_idx:end_idx].contiguous()
            v_chunk = value[:, start_idx:end_idx].contiguous()

            # Compute attention scores for ALL queries vs this K chunk
            # [b, h, n, d] @ [b, h, d, chunk_size] -> [b, h, n, chunk_size]
            scores = torch.matmul(
                query.transpose(1, 2),  # [b, h, n, d]
                k_chunk.transpose(1, 2).transpose(-2, -1),  # [b, h, d, chunk_size]
            ) / math.sqrt(d)

            # Apply causal mask if needed
            if is_causal:
                causal_mask = torch.ones(
                    n, chunk_size, device=self.device, dtype=torch.bool
                )
                for i in range(n):
                    for j in range(chunk_size):
                        if i < start_idx + j:
                            causal_mask[i, j] = False
                scores.masked_fill_(
                    ~causal_mask.unsqueeze(0).unsqueeze(1), float("-inf")
                )

            # Online softmax update
            # 1. Find max across this chunk
            chunk_max = scores.amax(dim=-1, keepdim=True)  # [b, h, n, 1]

            # 2. Update running max
            new_max = torch.maximum(running_max, chunk_max)

            # 3. Rescale existing output if max changed
            if chunk_idx > 0:
                output = output * torch.exp(running_max - new_max)

            # 4. Update running sum with proper scaling
            # Scale previous sum by exp(old_max - new_max)
            running_sum = running_sum * torch.exp(running_max - new_max)
            # Add this chunk's contribution
            running_sum = running_sum + torch.exp(scores - new_max).sum(
                dim=-1, keepdim=True
            )

            # 5. Update running max
            running_max = new_max

            # Accumulate weighted values
            exp_scores = torch.exp(scores - running_max)  # [b, h, n, chunk_size]

            # Apply values
            # [b, h, n, chunk_size] @ [b, h, chunk_size, d] -> [b, h, n, d]
            chunk_output = torch.matmul(exp_scores, v_chunk.transpose(1, 2))

            # Add to output (already in [b, h, n, d] format)
            output = output + chunk_output

            # Free memory (this is what gives Ring Attention its efficiency!)
            del k_chunk, v_chunk, scores, exp_scores, chunk_output
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Final normalization: divide by running sum
        # output is already [b, h, n, d], running_sum is [b, h, n, 1]
        output = output / running_sum

        # Apply dropout if training
        if self.dropout > 0 and hasattr(self, "training") and self.training:
            output = F.dropout(output, p=self.dropout)

        # Transpose back to [b, n, h, d]
        output = output.transpose(1, 2)

        if return_memory_stats and self.device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            current_memory = torch.cuda.memory_allocated() / (1024**2)

            stats = {
                "peak_memory_mb": peak_memory,
                "current_memory_mb": current_memory,
                "memory_saved_mb": start_memory - current_memory,
                "ring_size": ring_size,
                "chunk_size": chunk_size,
            }
            return output, stats

        return output


def test_correctness():
    """Test that the corrected implementation matches standard attention."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 for precision

    # Test parameters
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64

    # Create test tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Standard attention
    scores = torch.matmul(
        q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)
    ) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    expected = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2)

    # Test with different ring sizes
    ring_attention = RingAttentionCorrectV2(device=device, dtype=dtype)

    for ring_size in [1, 2, 4, 8]:
        if seq_len % ring_size != 0:
            continue

        output = ring_attention(q, k, v, ring_size=ring_size)

        # Check values
        max_diff = torch.max(torch.abs(output - expected)).item()
        mean_diff = torch.mean(torch.abs(output - expected)).item()

        # Check sums (should be roughly equal)
        expected_sum = expected.sum().item()
        actual_sum = output.sum().item()
        sum_ratio = actual_sum / expected_sum if expected_sum != 0 else 0

        print(f"Ring size {ring_size}:")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Sum ratio: {sum_ratio:.4f} (should be ~1.0)")

        assert max_diff < 1e-5, f"Output mismatch for ring_size={ring_size}"
        assert abs(sum_ratio - 1.0) < 0.01, (
            f"Incorrect normalization for ring_size={ring_size}"
        )

    print("\n✓ All correctness tests passed!")


if __name__ == "__main__":
    test_correctness()
