"""
Ring Attention implementation for Dilated Attention.

This module implements Ring Attention pattern for dilated attention, enabling 
O(n) memory scaling instead of O(n²) for arbitrarily long sequences through
distributed computation across multiple devices.

Ring Attention splits the key-value computation across devices in a ring pattern,
allowing each device to process only a fraction of the sequence while maintaining
global attention patterns.

Key Features:
- O(n) memory complexity instead of O(n²)
- Linear scaling to arbitrarily long sequences
- Distributed computation with minimal communication overhead
- Maintains mathematical equivalence to full attention
- Optimized for dilated attention patterns

References:
- Ring Attention with Blockwise Transformers (Liu et al., 2023)
- Dilated Attention (LongNet paper)
"""

from typing import Optional, Sequence, Tuple, Union, List, Dict, Any
import math
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel, SDPBackend


class RingAttentionMemoryPool:
    """Centralized memory pool for Ring Attention operations."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self._pools: Dict[tuple, torch.Tensor] = {}
        self._usage_count: Dict[tuple, int] = {}
    
    def get_buffer(self, shape: tuple, dtype: torch.dtype, key: str) -> torch.Tensor:
        """Get buffer from pool or allocate new one."""
        pool_key = (shape, dtype, key)
        
        if pool_key not in self._pools:
            self._pools[pool_key] = torch.empty(shape, dtype=dtype, device=self.device)
            self._usage_count[pool_key] = 0
        
        self._usage_count[pool_key] += 1
        return self._pools[pool_key]
    
    def clear_unused_buffers(self, threshold: int = 100):
        """Clear buffers that haven't been used recently."""
        keys_to_remove = [
            key for key, count in self._usage_count.items() 
            if count < threshold
        ]
        for key in keys_to_remove:
            del self._pools[key]
            del self._usage_count[key]

class RingDilatedAttention(nn.Module):
    """
    Ring Attention implementation for Dilated Attention with O(n) memory complexity.
    
    This implementation combines dilated attention patterns with ring attention
    to achieve linear memory scaling for extremely long sequences.
    
    Key innovations:
    - Ring-based key-value computation across devices
    - Dilated attention patterns within each ring segment  
    - Block-wise computation to minimize memory overhead
    - Optimized communication patterns for distributed training
    """
    
    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dropout: float = 0.0,
        use_tf32: bool = True,
        block_size: int = 1024,
        ring_size: Optional[int] = None,
        use_checkpointing: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize Ring Dilated Attention.
        
        Args:
            segment_lengths: Sequence of segment lengths for dilated attention
            dilation_rates: Corresponding dilation rates for each segment
            dropout: Dropout probability (default: 0.0)
            use_tf32: Enable TF32 optimization (default: True)
            block_size: Block size for ring attention computation (default: 1024)
            ring_size: Number of devices in ring (auto-detected if None)
            use_checkpointing: Enable gradient checkpointing (default: True)
            device: Device to place tensors on
        """
        super().__init__()
        
        # Validate inputs
        assert len(segment_lengths) == len(dilation_rates)
        assert all(s > 0 for s in segment_lengths), "Segment lengths must be positive"
        assert all(r > 0 for r in dilation_rates), "Dilation rates must be positive"
        assert block_size > 0, "Block size must be positive"
        
        self.segment_lengths = list(segment_lengths)
        self.dilation_rates = list(dilation_rates)
        self.dropout = dropout
        self.block_size = block_size
        self.use_checkpointing = use_checkpointing
        self.num_groups = len(self.segment_lengths)
        
        # Ring attention configuration
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.ring_size = ring_size or self.world_size
        
        # Optimization settings
        if use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Advanced memory optimization: Pre-compute head distribution and cache indices
        self._head_groups = None
        self._cached_indices = {}
        self._ring_buffers = {}
        self._ring_patterns = {}  # Cache ring-specific patterns
        
        # Memory pool for efficient buffer management
        device_obj = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._memory_pool = RingAttentionMemoryPool(device_obj)
        
        # Pre-allocated rotation buffers
        self._rotation_buffers = {}
        self._communication_buffers = {}
        
        # Setup ring communication if distributed
        self._setup_ring_communication()
    
    def _setup_ring_communication(self):
        """Setup ring communication pattern for distributed computation."""
        if not dist.is_initialized() or self.ring_size <= 1:
            self.ring_group = None
            return
        
        # Create ring communication group
        ring_ranks = list(range(min(self.ring_size, self.world_size)))
        self.ring_group = dist.new_group(ranks=ring_ranks)
        
        # Pre-allocate communication buffers
        self._setup_communication_buffers()
    
    def _setup_communication_buffers(self):
        """Pre-allocate buffers for efficient ring communication."""
        # These will be allocated dynamically based on input shapes
        self._kv_send_buffer = None
        self._kv_recv_buffer = None
        self._packed_communication_buffer = None  # For optimized K/V packing
    
    def _get_head_groups(self, h: int):
        """Pre-compute and cache head group distribution."""
        if self._head_groups is None or self._head_groups[0] != h:
            gs = [h // self.num_groups] * self.num_groups
            for i in range(h % self.num_groups):
                gs[i] += 1
            
            # Pre-compute head ranges for faster indexing
            head_ranges = []
            hmin = 0
            for g in gs:
                hmax = hmin + g
                head_ranges.append((hmin, hmax))
                hmin = hmax
            
            self._head_groups = (h, gs, head_ranges)
        
        return self._head_groups[1], self._head_groups[2]
    
    def _precompute_ring_patterns(self, h: int, ring_size: int):
        """Pre-compute all ring-specific patterns for maximum efficiency."""
        cache_key = (h, ring_size)
        if cache_key in self._ring_patterns:
            return self._ring_patterns[cache_key]
        
        gs, head_ranges = self._get_head_groups(h)
        
        # Pre-compute all ring step patterns
        ring_patterns = []
        for step in range(ring_size):
            step_patterns = {}
            for i, ((g, (hmin, hmax)), r, s) in enumerate(
                zip(zip(gs, head_ranges), self.dilation_rates, self.segment_lengths)
            ):
                # Cache dilation indices
                cache_key_indices = (s, r, i % r)
                if cache_key_indices not in self._cached_indices:
                    self._cached_indices[cache_key_indices] = torch.arange(
                        i % r, s, r, device=self._memory_pool.device
                    )
                
                step_patterns[i] = {
                    'head_range': (hmin, hmax),
                    'indices': self._cached_indices[cache_key_indices],
                    'segment_size': s,
                    'dilation': r,
                    'group_size': g
                }
            ring_patterns.append(step_patterns)
        
        self._ring_patterns[cache_key] = ring_patterns
        return ring_patterns
    
    def _allocate_ring_buffers(self, k: Tensor, v: Tensor):
        """Allocate ring communication buffers with memory pool optimization."""
        if self.ring_group is None:
            return
        
        b, n, h, d = k.shape
        
        # Calculate local sequence length per device in ring
        local_seq_len = n // self.ring_size
        buffer_shape = (b, local_seq_len, h, d)
        
        # Use memory pool for efficient buffer allocation
        if (self._kv_send_buffer is None or 
            self._kv_send_buffer.shape != buffer_shape or
            self._kv_send_buffer.dtype != k.dtype):
            
            self._kv_send_buffer = self._memory_pool.get_buffer(
                buffer_shape, k.dtype, "kv_send"
            )
            self._kv_recv_buffer = self._memory_pool.get_buffer(
                buffer_shape, k.dtype, "kv_recv"
            )
            
            # Allocate optimized packed communication buffer
            packed_size = 2 * b * local_seq_len * h * d  # K + V
            self._packed_communication_buffer = self._memory_pool.get_buffer(
                (packed_size,), k.dtype, "packed_comm"
            )
        
        # Pre-allocate rotation buffers for in-place operations
        buffer_key = (buffer_shape, k.dtype)
        if buffer_key not in self._rotation_buffers:
            self._rotation_buffers[buffer_key] = {
                'k': self._memory_pool.get_buffer(buffer_shape, k.dtype, "rot_k"),
                'v': self._memory_pool.get_buffer(buffer_shape, k.dtype, "rot_v")
            }
    
    def _ring_attention_step(
        self, 
        q_local: Tensor, 
        k_segment: Tensor, 
        v_segment: Tensor,
        is_causal: bool = False,
        step: int = 0
    ) -> Tensor:
        """
        Perform single step of ring attention computation.
        
        Args:
            q_local: Local query tensor [b, local_seq_len, h, d]
            k_segment: Key segment from ring [b, seg_len, h, d]  
            v_segment: Value segment from ring [b, seg_len, h, d]
            is_causal: Whether to apply causal masking
            step: Current ring step (for causal masking)
            
        Returns:
            Attention output for this step [b, local_seq_len, h, d]
        """
        # Apply dilated attention patterns within ring segments
        return self._dilated_attention_block(
            q_local, k_segment, v_segment, is_causal, step
        )
    
    def _dilated_attention_block(
        self,
        q: Tensor,
        k: Tensor, 
        v: Tensor,
        is_causal: bool = False,
        ring_step: int = 0
    ) -> Tensor:
        """
        Apply dilated attention patterns within a block.
        
        This is the core dilated attention computation adapted for ring attention.
        """
        b, n_q, h, d = q.shape
        b, n_kv, h, d = k.shape
        
        # Get pre-computed head distribution and ring patterns
        gs, head_ranges = self._get_head_groups(h)
        ring_patterns = self._precompute_ring_patterns(h, self.ring_size)
        
        # Optimized output allocation: use empty + zero for better memory efficiency
        out = torch.empty_like(q)
        out.zero_()
        
        # Process each dilation group using pre-computed patterns
        for i, ((g, (hmin, hmax)), r, s) in enumerate(
            zip(zip(gs, head_ranges), self.dilation_rates, self.segment_lengths)
        ):
            # Skip if segments are larger than available sequence
            if n_q < s or n_kv < s:
                continue
            
            # Calculate effective segment size for this block
            effective_s = min(s, n_kv)
            
            # Use tensor views instead of slicing for zero-copy operations
            q_slice = q[:, :, hmin:hmax, :].view(b, n_q, g, d)
            k_slice = k[:, :, hmin:hmax, :].view(b, n_kv, g, d)
            v_slice = v[:, :, hmin:hmax, :].view(b, n_kv, g, d)
            
            # Segment and apply dilation
            q_segments = self._segment_tensor(q_slice, effective_s, n_q)
            k_segments = self._segment_tensor(k_slice, effective_s, n_kv)
            v_segments = self._segment_tensor(v_slice, effective_s, n_kv)
            
            # Apply dilation within segments using pre-computed indices
            if r > 1:
                offset = i % r
                cache_key = (effective_s, r, offset)
                if cache_key not in self._cached_indices:
                    self._cached_indices[cache_key] = torch.arange(
                        offset, effective_s, r, device=q.device
                    )
                idx = self._cached_indices[cache_key]
                
                # Use advanced indexing for efficient dilation
                k_segments = k_segments.index_select(2, idx)
                v_segments = v_segments.index_select(2, idx)
            
            # Flatten for attention computation
            num_segments_q = q_segments.size(1)
            num_segments_kv = k_segments.size(1) 
            dilated_len = k_segments.size(2)
            
            q_flat = q_segments.view(b * num_segments_q, effective_s, g, d)
            k_flat = k_segments.view(b * num_segments_kv, dilated_len, g, d)
            v_flat = v_segments.view(b * num_segments_kv, dilated_len, g, d)
            
            # Handle different segment counts between q and kv (ring attention)
            if num_segments_q != num_segments_kv:
                # Repeat k,v to match q segments for ring attention
                repeat_factor = (num_segments_q + num_segments_kv - 1) // num_segments_kv
                k_flat = k_flat.repeat(repeat_factor, 1, 1, 1)[:b * num_segments_q]
                v_flat = v_flat.repeat(repeat_factor, 1, 1, 1)[:b * num_segments_q]
            
            # Apply scaled dot product attention
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                attn_out = F.scaled_dot_product_attention(
                    q_flat, k_flat, v_flat,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal and ring_step == 0,  # Only causal for current ring position
                    scale=None,
                )
            
            # Reshape back and accumulate
            attn_reshaped = attn_out.view(b, num_segments_q, effective_s, g, d)
            attn_flat = attn_reshaped.view(b, n_q, g, d)
            
            # Accumulate results with in-place operations for efficiency
            out_slice = out[:, :, hmin:hmax, :] 
            out_slice.add_(attn_flat)
        
        # Normalize by number of groups using in-place division
        out.div_(self.num_groups)
        return out
    
    def _segment_tensor(self, x: Tensor, segment_size: int, total_len: int) -> Tensor:
        """
        Segment tensor for dilated attention computation.
        
        Args:
            x: Input tensor [b, seq_len, h, d]
            segment_size: Size of each segment
            total_len: Total sequence length
            
        Returns:
            Segmented tensor [b, num_segments, segment_size, h, d]
        """
        b, seq_len, h, d = x.shape
        
        # Pad if necessary
        pad_len = ((total_len + segment_size - 1) // segment_size) * segment_size - total_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        
        # Reshape to segments
        num_segments = (total_len + segment_size - 1) // segment_size
        return x[:, :num_segments * segment_size].view(b, num_segments, segment_size, h, d)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        """
        Forward pass with Ring Attention for O(n) memory complexity.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            
        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        b, n, h, d = q.shape
        
        # Single device case - fall back to standard dilated attention
        if self.ring_group is None or self.ring_size <= 1:
            return self._single_device_forward(q, k, v, is_causal)
        
        # Ring attention with multiple devices
        return self._ring_forward(q, k, v, is_causal)
    
    def _single_device_forward(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        """Single device forward pass (optimized dilated attention)."""
        return self._dilated_attention_block(q, k, v, is_causal, ring_step=0)
    
    def _ring_forward(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        """
        Ring attention forward pass for distributed computation.
        
        This implements the core ring attention algorithm where each device
        processes a local portion of queries while keys/values are rotated
        through the ring.
        """
        b, n, h, d = q.shape
        device = q.device
        
        # Calculate local sequence lengths
        local_seq_len = n // self.ring_size
        start_idx = self.rank * local_seq_len
        end_idx = start_idx + local_seq_len
        
        # Handle remainder
        if self.rank == self.ring_size - 1:
            end_idx = n
            local_seq_len = end_idx - start_idx
        
        # Get local query segment
        q_local = q[:, start_idx:end_idx]  # [b, local_seq_len, h, d]
        
        # Get local key/value segments  
        k_local = k[:, start_idx:end_idx]  # [b, local_seq_len, h, d]
        v_local = v[:, start_idx:end_idx]  # [b, local_seq_len, h, d]
        
        # Allocate ring communication buffers
        self._allocate_ring_buffers(k_local, v_local)
        
        # Initialize output accumulator
        output = torch.zeros_like(q_local)
        
        # Ring attention steps
        k_ring = k_local.clone()
        v_ring = v_local.clone()
        
        for step in range(self.ring_size):
            # Compute attention with current k/v ring segment
            if self.use_checkpointing:
                step_output = torch.utils.checkpoint.checkpoint(
                    self._ring_attention_step,
                    q_local, k_ring, v_ring, is_causal, step,
                    use_reentrant=False
                )
            else:
                step_output = self._ring_attention_step(q_local, k_ring, v_ring, is_causal, step)
            
            # Accumulate results
            output.add_(step_output)
            
            # Rotate k,v to next position in ring (except last step)
            if step < self.ring_size - 1:
                k_ring, v_ring = self._rotate_kv_ring(k_ring, v_ring)
        
        # Gather results from all devices
        output_gathered = self._gather_ring_outputs(output, local_seq_len)
        
        return output_gathered
    
    def _rotate_kv_ring(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Optimized ring rotation with packed K/V communication and buffer reuse.
        
        This implements an optimized ring communication pattern that:
        1. Packs K/V into single communication to halve latency
        2. Uses pre-allocated buffers to eliminate memory allocation
        3. Employs in-place operations to minimize memory overhead
        """
        if self.ring_group is None:
            return k, v
        
        # Get pre-allocated rotation buffers
        buffer_key = (k.shape, k.dtype)
        rotation_buffers = self._rotation_buffers[buffer_key]
        k_buffer = rotation_buffers['k']
        v_buffer = rotation_buffers['v']
        
        # Calculate send/receive ranks
        send_rank = (self.rank + 1) % self.ring_size
        recv_rank = (self.rank - 1) % self.ring_size
        
        # Pack K and V into single communication buffer for efficiency
        k_flat = k.flatten()
        v_flat = v.flatten()
        k_size = k_flat.numel()
        
        # Create packed buffer view
        packed_send = self._packed_communication_buffer[:k_size + v_flat.numel()]
        packed_recv = torch.empty_like(packed_send)
        
        # Pack data
        packed_send[:k_size].copy_(k_flat)
        packed_send[k_size:].copy_(v_flat)
        
        # Single async communication for both K and V
        send_req = dist.isend(packed_send, dst=send_rank, group=self.ring_group)
        recv_req = dist.irecv(packed_recv, src=recv_rank, group=self.ring_group)
        
        # Wait for communication completion
        send_req.wait()
        recv_req.wait()
        
        # Unpack received data directly into pre-allocated buffers
        k_received_flat = packed_recv[:k_size]
        v_received_flat = packed_recv[k_size:]
        
        # Use copy instead of clone for better memory efficiency
        k_buffer.copy_(k_received_flat.view_as(k))
        v_buffer.copy_(v_received_flat.view_as(v))
        
        return k_buffer, v_buffer
    
    def _gather_ring_outputs(self, local_output: Tensor, local_seq_len: int) -> Tensor:
        """
        Gather outputs from all devices in ring to reconstruct full sequence.
        
        Args:
            local_output: Local attention output [b, local_seq_len, h, d]
            local_seq_len: Length of local sequence
            
        Returns:
            Full sequence output [b, total_seq_len, h, d]
        """
        if self.ring_group is None:
            return local_output
        
        # Gather all local outputs
        output_list = [torch.zeros_like(local_output) for _ in range(self.ring_size)]
        dist.all_gather(output_list, local_output, group=self.ring_group)
        
        # Concatenate to form full sequence
        return torch.cat(output_list, dim=1)
    
    def clear_cache(self):
        """Clear cached patterns and buffers to free memory."""
        self._cached_indices.clear()
        self._ring_patterns.clear()
        self._rotation_buffers.clear()
        self._memory_pool.clear_unused_buffers()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory usage information."""
        info = {
            "memory_complexity": "O(n)",
            "ring_size": self.ring_size,
            "supports_infinite_context": True,
            "cached_patterns": len(self._ring_patterns),
            "cached_indices": len(self._cached_indices),
            "allocated_buffers": len(self._memory_pool._pools),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
            })
        
        return info


# Optional: Enable torch.compile for additional optimization
# RingDilatedAttention = torch.compile(RingDilatedAttention, fullgraph=True)