"""
Ring Dilated Attention V2 with fixed distributed communication.

This version uses isend/irecv instead of sendrecv for compatibility
with older PyTorch versions.
"""

import torch
import torch.distributed as dist
from torch import Tensor

from .ring_dilated_attention_v2 import RingDilatedAttentionV2


class RingDilatedAttentionV2Fixed(RingDilatedAttentionV2):
    """
    Ring Dilated Attention V2 with fixed distributed communication.

    This version overrides the _ring_sendrecv method to use isend/irecv
    instead of the newer sendrecv API, making it compatible with more
    PyTorch versions.
    """

    def _ring_sendrecv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate K/V chunks through the ring using isend/irecv."""
        if not dist.is_initialized():
            # Fallback for non-distributed mode
            return k, v

        # Pack K and V
        k_size = k.numel()
        v_size = v.numel()
        total_size = k_size + v_size

        # Ensure buffers are allocated
        if self._kv_send_buffer is None or self._kv_send_buffer.numel() < total_size:
            self._allocate_comm_buffers(k, v)

        # Copy data to send buffer
        self._kv_send_buffer[:k_size].copy_(k.flatten())
        self._kv_send_buffer[k_size:total_size].copy_(v.flatten())

        # Ring communication using isend/irecv
        send_rank = (self.rank + 1) % self.ring_size
        recv_rank = (self.rank - 1) % self.ring_size

        # Use isend and irecv for non-blocking communication
        send_op = dist.isend(
            self._kv_send_buffer[:total_size],
            dst=send_rank,
        )

        recv_op = dist.irecv(
            self._kv_recv_buffer[:total_size],
            src=recv_rank,
        )

        # Wait for both operations to complete
        send_op.wait()
        recv_op.wait()

        # Unpack received data
        k_new = self._kv_recv_buffer[:k_size].reshape_as(k)
        v_new = self._kv_recv_buffer[k_size:total_size].reshape_as(v)

        return k_new, v_new

    def _distributed_ring_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """True distributed ring attention across multiple GPUs."""
        b, n, h, d = q.shape

        # Split sequence across GPUs
        assert n % self.ring_size == 0, (
            f"Sequence length {n} must be divisible by ring size {self.ring_size}"
        )
        chunk_size = n // self.ring_size

        # Each GPU gets its local chunk
        local_start = self.rank * chunk_size
        local_end = (self.rank + 1) * chunk_size

        # Get local Q, K, V chunks
        q_local = q[:, local_start:local_end].contiguous()
        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Apply dilated attention pattern to local chunks
        q_local = self._apply_dilated_attention_pattern(
            q_local, k_local, v_local, is_causal=False
        )

        # Initialize output and running statistics
        output = torch.zeros_like(q_local)
        running_max = torch.full(
            (b, h, q_local.size(1), 1),
            fill_value=-float("inf"),
            device=q.device,
            dtype=q.dtype,
        )
        running_sum = torch.zeros_like(running_max)

        # Allocate communication buffers
        self._allocate_comm_buffers(k_local, v_local)

        # Ring iterations
        k_chunk = k_local.clone()
        v_chunk = v_local.clone()

        for step in range(self.ring_size):
            # Calculate which chunk we're processing
            source_rank = (self.rank - step) % self.ring_size
            chunk_start = source_rank * chunk_size

            # Compute attention scores with online softmax
            scores, new_max, new_sum = self._compute_attention_chunk_online(
                q_local,
                k_chunk,
                v_chunk,
                chunk_start,
                is_causal,
                running_max,
                running_sum,
                output,
                step,
            )

            # Update running statistics
            running_max = new_max
            running_sum = new_sum

            # Rotate K/V for next iteration (except last)
            if step < self.ring_size - 1:
                k_chunk, v_chunk = self._ring_sendrecv(k_chunk, v_chunk)

        # Final normalization
        output = output / running_sum

        # Gather outputs from all GPUs
        # For now, we'll use all_gather to collect results
        output_list = [torch.zeros_like(output) for _ in range(self.ring_size)]
        dist.all_gather(output_list, output)

        # Concatenate to get full output
        full_output = torch.cat(output_list, dim=1)

        return full_output
