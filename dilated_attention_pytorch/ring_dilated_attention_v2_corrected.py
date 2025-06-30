"""
Corrected Ring Dilated Attention V2 - Using proper PyTorch distributed APIs.

This fixes the main issue in the original V2: using non-existent dist.sendrecv.
Instead, we use the correct isend/irecv APIs that actually exist in PyTorch.
"""

import torch.distributed as dist
from torch import Tensor

# Import the base class and utilities from V2
from .ring_dilated_attention_v2 import RingDilatedAttentionV2


class RingDilatedAttentionV2Corrected(RingDilatedAttentionV2):
    """
    Corrected Ring Dilated Attention that uses proper PyTorch distributed APIs.

    The only change from V2 is fixing the _ring_sendrecv method to use
    isend/irecv instead of the non-existent sendrecv function.
    """

    def _ring_sendrecv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Rotate K/V chunks through the ring using CORRECT PyTorch APIs.

        This is the key fix - using isend/irecv which actually exist in PyTorch
        instead of the non-existent sendrecv function.

        Args:
            k: Key tensor to send/receive
            v: Value tensor to send/receive

        Returns:
            Tuple of (k_new, v_new) received from neighbor
        """
        if self.mode != "distributed":
            # For non-distributed mode, just return the same tensors
            return k, v

        # Allocate communication buffers if needed
        self._allocate_comm_buffers(k, v)

        # Pack K and V into send buffer
        k_size = k.numel()
        v_size = v.numel()
        total_size = k_size + v_size

        self._kv_send_buffer[:k_size].copy_(k.flatten())
        self._kv_send_buffer[k_size:total_size].copy_(v.flatten())

        # Determine communication partners
        send_rank = (self.rank + 1) % self.ring_size
        recv_rank = (self.rank - 1 + self.ring_size) % self.ring_size

        # CORRECT IMPLEMENTATION: Use isend/irecv instead of sendrecv
        # Option 1: Using separate isend/irecv operations
        send_req = dist.isend(self._kv_send_buffer[:total_size], dst=send_rank)
        recv_req = dist.irecv(self._kv_recv_buffer[:total_size], src=recv_rank)

        # Wait for both operations to complete
        send_req.wait()
        recv_req.wait()

        # Alternative Option 2: Using batch_isend_irecv (more efficient)
        # This is available in PyTorch 1.9+ and batches multiple P2P operations
        """
        ops = []
        ops.append(dist.P2POp(dist.isend, self._kv_send_buffer[:total_size], send_rank))
        ops.append(dist.P2POp(dist.irecv, self._kv_recv_buffer[:total_size], recv_rank))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        """

        # Unpack received data
        k_new = self._kv_recv_buffer[:k_size].view_as(k)
        v_new = self._kv_recv_buffer[k_size:total_size].view_as(v)

        return k_new.contiguous(), v_new.contiguous()

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass with corrected distributed communication.

        Everything else remains the same as V2, only the communication is fixed.
        """
        # Use parent's forward method which calls our corrected _ring_sendrecv
        return super().forward(q, k, v, is_causal)


# For convenience, also provide a factory function
def create_ring_dilated_attention_v2(
    segment_lengths: list[int], dilation_rates: list[int], **kwargs
) -> RingDilatedAttentionV2Corrected:
    """
    Create a corrected Ring Dilated Attention V2 instance.

    This is the version that should be used in production as it properly
    implements distributed communication using existing PyTorch APIs.

    Args:
        segment_lengths: List of segment lengths
        dilation_rates: List of dilation rates
        **kwargs: Additional arguments passed to constructor

    Returns:
        Corrected Ring Dilated Attention V2 instance
    """
    return RingDilatedAttentionV2Corrected(
        segment_lengths=segment_lengths, dilation_rates=dilation_rates, **kwargs
    )


# Also provide a drop-in replacement class name
RingDilatedAttention = RingDilatedAttentionV2Corrected
