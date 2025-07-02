"""
Bucketed processing utilities for Ring Attention.

Bucketing allows processing attention in smaller chunks to reduce memory usage
and improve efficiency, especially for very long sequences.
"""

import math
from typing import Tuple, Optional, List, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from .ring_attention_lse import compute_attention_with_lse, StableRingAccumulator


@dataclass
class BucketConfig:
    """Configuration for bucketed processing."""

    bucket_size: int = 1024
    num_buckets: Optional[int] = None
    grad_checkpoint: bool = False
    use_flash_attn: bool = True

    def get_num_buckets(self, seq_len: int) -> int:
        """Calculate number of buckets for given sequence length."""
        if self.num_buckets is not None:
            return self.num_buckets
        return math.ceil(seq_len / self.bucket_size)


def create_buckets(
    tensor: Tensor,
    bucket_size: int,
    dim: int = 1,
) -> List[Tensor]:
    """
    Split tensor into buckets along specified dimension.

    Args:
        tensor: Input tensor to bucket
        bucket_size: Size of each bucket
        dim: Dimension to split along

    Returns:
        List of tensor buckets
    """
    seq_len = tensor.shape[dim]
    buckets = []

    for i in range(0, seq_len, bucket_size):
        end = min(i + bucket_size, seq_len)
        # Use narrow for efficiency (creates view, not copy)
        bucket = tensor.narrow(dim, i, end - i)
        buckets.append(bucket)

    return buckets


def merge_buckets(
    buckets: List[Tensor],
    dim: int = 1,
) -> Tensor:
    """
    Merge buckets back into single tensor.

    Args:
        buckets: List of tensor buckets
        dim: Dimension to concatenate along

    Returns:
        Merged tensor
    """
    if len(buckets) == 1:
        return buckets[0]
    return torch.cat(buckets, dim=dim)


class BucketedAttentionProcessor:
    """
    Process attention in buckets for memory efficiency.

    This processor handles the computation of attention in smaller chunks,
    accumulating results using log-sum-exp for numerical stability.
    """

    def __init__(
        self,
        bucket_config: BucketConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = bucket_config
        self.device = device
        self.dtype = dtype

    def process_attention(
        self,
        q: Tensor,  # (batch, heads, seq_q, dim)
        k: Tensor,  # (batch, heads, seq_k, dim)
        v: Tensor,  # (batch, heads, seq_k, dim)
        scale: float,
        mask: Optional[Tensor] = None,
        is_causal: bool = False,
        dropout: float = 0.0,
        training: bool = False,
        q_offset: int = 0,  # Offset for query positions
        kv_offset: int = 0,  # Offset for key/value positions
    ) -> Tuple[Tensor, Tensor]:
        """
        Process attention in buckets with LSE accumulation.

        Args:
            q, k, v: Query, key, value tensors
            scale: Attention scaling factor
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            dropout: Dropout probability
            training: Whether in training mode

        Returns:
            (output, lse) tuple
        """
        b, h, seq_q, d = q.shape
        b, h, seq_k, d = k.shape

        # Create buckets for queries
        q_buckets = create_buckets(q, self.config.bucket_size, dim=2)
        _ = len(q_buckets)

        # Create buckets for keys and values
        k_buckets = create_buckets(k, self.config.bucket_size, dim=2)
        v_buckets = create_buckets(v, self.config.bucket_size, dim=2)
        _ = len(k_buckets)

        # Initialize accumulator
        accumulator = StableRingAccumulator(
            output_shape=(b, h, seq_q, d),
            device=self.device,
            dtype=self.dtype,
        )

        # Process each query bucket
        for q_idx, q_bucket in enumerate(q_buckets):
            q_start = q_idx * self.config.bucket_size
            q_end = min(q_start + self.config.bucket_size, seq_q)
            q_len = q_end - q_start

            # Adjust for global offset
            global_q_start = q_start + q_offset
            global_q_end = q_end + q_offset

            # Initialize bucket accumulator
            bucket_accumulator = StableRingAccumulator(
                output_shape=(b, h, q_len, d),
                device=self.device,
                dtype=self.dtype,
            )

            # Process against each key/value bucket
            for kv_idx, (k_bucket, v_bucket) in enumerate(zip(k_buckets, v_buckets)):
                kv_start = kv_idx * self.config.bucket_size
                kv_end = min(kv_start + self.config.bucket_size, seq_k)

                # Adjust for global offset
                global_kv_start = kv_start + kv_offset
                global_kv_end = kv_end + kv_offset

                # Create bucket mask if needed
                bucket_mask = None
                if is_causal:
                    bucket_mask = self._create_causal_bucket_mask(
                        global_q_start,
                        global_q_end,
                        global_kv_start,
                        global_kv_end,
                        self.device,
                    )
                    bucket_mask = bucket_mask.unsqueeze(0).unsqueeze(0)
                elif mask is not None:
                    # Extract relevant portion of mask
                    bucket_mask = mask[:, :, q_start:q_end, kv_start:kv_end]

                # Compute attention for this bucket pair
                if self.config.grad_checkpoint and training:
                    # Use gradient checkpointing to save memory
                    bucket_output, bucket_lse = torch.utils.checkpoint.checkpoint(
                        compute_attention_with_lse,
                        q_bucket,
                        k_bucket,
                        v_bucket,
                        scale,
                        bucket_mask,
                        dropout,
                        training,
                        use_reentrant=False,
                    )
                else:
                    bucket_output, bucket_lse = compute_attention_with_lse(
                        q_bucket,
                        k_bucket,
                        v_bucket,
                        scale,
                        bucket_mask,
                        dropout,
                        training,
                    )

                # Accumulate bucket results
                bucket_accumulator.update(bucket_output, bucket_lse)

            # Get output for this query bucket
            q_bucket_output = bucket_accumulator.get_output()

            # Update main accumulator
            # We need to expand this bucket's output to full sequence length
            full_output = torch.zeros(
                b, h, seq_q, d, device=self.device, dtype=self.dtype
            )
            full_output[:, :, q_start:q_end] = q_bucket_output

            # Create corresponding LSE
            full_lse = torch.full(
                (b, h, seq_q), float("-inf"), device=self.device, dtype=self.dtype
            )
            full_lse[:, :, q_start:q_end] = bucket_accumulator.lse

            accumulator.update(full_output, full_lse)

        return accumulator.get_output(), accumulator.lse

    def _create_causal_bucket_mask(
        self,
        q_start: int,
        q_end: int,
        kv_start: int,
        kv_end: int,
        device: torch.device,
    ) -> Tensor:
        """Create causal mask for bucket pair."""
        _ = q_end - q_start
        _ = kv_end - kv_start

        # Create position indices
        q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(1)
        kv_pos = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)

        # Causal mask: q can only attend to kv with position <= q
        mask = q_pos >= kv_pos
        return mask


def bucketed_ring_pass(
    q_buckets: List[Tensor],
    kv_local: Tensor,
    ring_info,
    bucket_config: BucketConfig,
    scale: float,
    is_causal: bool = False,
    dropout: float = 0.0,
    training: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Process one ring pass with bucketed computation.

    This function handles the attention computation for one position
    in the ring, processing queries in buckets against the current
    K,V chunk.

    Args:
        q_buckets: List of query buckets
        kv_local: Current K,V chunk from ring
        ring_info: Information about current ring position
        bucket_config: Bucketing configuration
        scale: Attention scaling factor
        is_causal: Whether to apply causal masking
        dropout: Dropout probability
        training: Whether in training mode

    Returns:
        (output, lse) tuple for this ring pass
    """
    k_local, v_local = kv_local
    device = q_buckets[0].device
    dtype = q_buckets[0].dtype

    # Create processor
    processor = BucketedAttentionProcessor(bucket_config, device, dtype)

    # Merge query buckets temporarily for processing
    # In practice, we'd process bucket-by-bucket, but for simplicity...
    q_merged = merge_buckets(q_buckets, dim=2)

    # Process attention
    output, lse = processor.process_attention(
        q_merged,
        k_local,
        v_local,
        scale=scale,
        is_causal=is_causal,
        dropout=dropout,
        training=training,
    )

    return output, lse


class BucketIterator:
    """
    Iterator for processing sequences in buckets.

    Yields bucket indices and slices for efficient processing.
    """

    def __init__(self, seq_len: int, bucket_size: int):
        self.seq_len = seq_len
        self.bucket_size = bucket_size
        self.num_buckets = math.ceil(seq_len / bucket_size)

    def __iter__(self) -> Iterator[Tuple[int, slice]]:
        """Yield (bucket_idx, slice) tuples."""
        for i in range(self.num_buckets):
            start = i * self.bucket_size
            end = min(start + self.bucket_size, self.seq_len)
            yield i, slice(start, end)

    def get_bucket_positions(self, bucket_idx: int) -> Tuple[int, int]:
        """Get start and end positions for a bucket."""
        start = bucket_idx * self.bucket_size
        end = min(start + self.bucket_size, self.seq_len)
        return start, end


def create_bucket_attention_mask(
    seq_len: int,
    bucket_size: int,
    is_causal: bool = True,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Create attention mask for bucketed processing.

    This creates a block-diagonal mask pattern that allows
    attention within buckets while respecting causality.

    Args:
        seq_len: Total sequence length
        bucket_size: Size of each bucket
        is_causal: Whether to apply causal masking
        device: Device to create mask on

    Returns:
        Attention mask tensor
    """
    if device is None:
        device = torch.device("cpu")

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Create bucket iterator
    iterator = BucketIterator(seq_len, bucket_size)

    # Fill in attention pattern
    for q_idx, q_slice in iterator:
        for kv_idx, kv_slice in iterator:
            if is_causal:
                # For causal, only attend to previous and current buckets
                if kv_idx <= q_idx:
                    # Create causal pattern within valid buckets
                    q_start, q_end = iterator.get_bucket_positions(q_idx)
                    kv_start, kv_end = iterator.get_bucket_positions(kv_idx)

                    for i in range(q_start, q_end):
                        for j in range(kv_start, kv_end):
                            if i >= j:  # Causal constraint
                                mask[i, j] = True
            else:
                # For non-causal, simple bucket pattern
                mask[q_slice, kv_slice] = True

    return mask
