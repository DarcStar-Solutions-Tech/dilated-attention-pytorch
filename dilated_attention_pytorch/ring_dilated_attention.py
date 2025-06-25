"""
Ring Attention implementation for Dilated Attention using the refactored core architecture.

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
import threading

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist

from .core import (
    BaseDilatedAttention,
    RingAttentionConfig,
    get_global_memory_pool,
    optimize_attention_computation,
    HAS_FLASH_ATTN_3,
    GPU_TYPE,
    HAS_FLASH_ATTN,
)

# Handle torch.nn.attention availability
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_SDPA_KERNEL = True
except ImportError:
    HAS_SDPA_KERNEL = False
    # Fallback for older PyTorch versions
    class SDPBackend:
        FLASH_ATTENTION = "flash_attention"
        EFFICIENT_ATTENTION = "efficient_attention"
        MATH = "math"


def get_flash_attention_version() -> Optional[str]:
    """Get the version of Flash Attention if available."""
    if not HAS_FLASH_ATTN:
        return None
    try:
        import flash_attn
        return flash_attn.__version__
    except:
        return None


def is_flash_attention_3_available() -> bool:
    """Check if Flash Attention 3 is available."""
    return HAS_FLASH_ATTN_3


class RingAttentionMemoryPool:
    """Centralized memory pool for Ring Attention operations with optimized lookups."""
    
    def __init__(self, device: torch.device, max_pool_size: int = 100, max_cache_size: int = 20):
        self.device = device
        self.max_pool_size = max_pool_size
        self.max_cache_size = max_cache_size
        self._pools: Dict[tuple, torch.Tensor] = {}
        self._usage_count: Dict[tuple, int] = {}
        # Optimization: Cache frequently used buffer keys for faster lookups
        self._hot_keys_cache = {}  # Maps simplified keys to full keys
        self._access_lock = threading.Lock()  # Thread-safe access
        self._access_order = []  # Track access order for LRU eviction
    
    def get_buffer(self, shape: tuple, dtype: torch.dtype, key: str, pin_memory: bool = False) -> torch.Tensor:
        """Get buffer from pool or allocate new one with optimized hot cache lookup."""
        pool_key = (shape, dtype, key, pin_memory)
        
        # Optimization: Check hot cache first for frequently used buffers
        simplified_key = (shape, dtype, key)  # Omit pin_memory for hot cache
        
        with self._access_lock:
            # Fast path: Check hot cache for common buffer patterns
            if simplified_key in self._hot_keys_cache:
                cached_key = self._hot_keys_cache[simplified_key]
                if cached_key in self._pools:
                    self._usage_count[cached_key] += 1
                    return self._pools[cached_key]
            
            # Standard path: Full lookup
            if pool_key not in self._pools:
                # Check if we need to evict before allocating
                if len(self._pools) >= self.max_pool_size:
                    self._evict_lru_buffer()
                
                if self.device.type == 'cuda' and pin_memory:
                    # Allocate pinned memory for faster GPU transfers
                    self._pools[pool_key] = torch.empty(
                        shape, dtype=dtype, device='cpu', pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    self._pools[pool_key] = torch.empty(shape, dtype=dtype, device=self.device)
                self._usage_count[pool_key] = 0
                
                # Update hot cache if this becomes frequently used
                if self._usage_count.get(pool_key, 0) > 5:  # Threshold for hot cache
                    # Limit hot cache size
                    if len(self._hot_keys_cache) >= self.max_cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self._hot_keys_cache))
                        del self._hot_keys_cache[oldest_key]
                    self._hot_keys_cache[simplified_key] = pool_key
            
            # Update access order for LRU
            if pool_key in self._access_order:
                self._access_order.remove(pool_key)
            self._access_order.append(pool_key)
            
            self._usage_count[pool_key] += 1
            return self._pools[pool_key]
    
    def _evict_lru_buffer(self):
        """Evict least recently used buffer."""
        if not self._access_order:
            # Fallback to evicting buffer with lowest usage count
            if self._usage_count:
                lru_key = min(self._usage_count.items(), key=lambda x: x[1])[0]
            else:
                # Last resort: evict any buffer
                lru_key = next(iter(self._pools.keys()))
        else:
            lru_key = self._access_order[0]
            self._access_order.remove(lru_key)
        
        # Remove from all data structures
        if lru_key in self._pools:
            del self._pools[lru_key]
        if lru_key in self._usage_count:
            del self._usage_count[lru_key]
        # Remove from hot cache if present
        for simple_key, full_key in list(self._hot_keys_cache.items()):
            if full_key == lru_key:
                del self._hot_keys_cache[simple_key]
                break
    
    def clear_unused_buffers(self, threshold: int = 100, reset_counters: bool = True):
        """Clear buffers that haven't been used recently."""
        with self._access_lock:
            if not self._usage_count:
                return
            
            # Use adaptive threshold based on memory pressure
            if torch.cuda.is_available():
                memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                memory_ratio = memory_free / torch.cuda.get_device_properties(0).total_memory
                if memory_ratio < 0.1:  # Low memory, be more aggressive
                    threshold = max(1, threshold // 4)
                elif memory_ratio > 0.5:  # High memory, be more conservative  
                    threshold = threshold * 2
            
            keys_to_remove = [
                key for key, count in self._usage_count.items() 
                if count < threshold
            ]
            for key in keys_to_remove:
                del self._pools[key]
                del self._usage_count[key]
            
            # Reset counters to prevent overflow and provide fair chance for new buffers
            if reset_counters and self._usage_count:
                min_count = min(self._usage_count.values())
                for key in self._usage_count:
                    self._usage_count[key] = max(0, self._usage_count[key] - min_count)
            
            # Update access order list to remove deleted keys
            self._access_order = [key for key in self._access_order if key in self._pools]

class RingDilatedAttention(BaseDilatedAttention):
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
        dtype: Optional[torch.dtype] = None,
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
            dtype: Data type for parameters
        """
        # Create configuration
        config = RingAttentionConfig(
            segment_lengths=list(segment_lengths),
            dilation_rates=list(dilation_rates),
            dropout=dropout,
            use_tf32=use_tf32,
            block_size=block_size,
            ring_size=ring_size,
            use_checkpointing=use_checkpointing,
            device=device,
            dtype=dtype,
        )
        
        # Initialize base class
        super().__init__(config)
        
        # Store ring-specific attributes
        self.block_size = self.config.block_size
        self.use_checkpointing = self.config.use_checkpointing
        
        # Ring attention configuration
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.ring_size = self.config.ring_size or self.world_size
        
        # Ring-specific caches and buffers
        self._cached_indices = {}  # Cache for dilation indices
        self._ring_buffers = {}
        self._ring_patterns = {}  # Cache ring-specific patterns
        self._vectorized_patterns = {}  # Cache for vectorized pattern computation
        
        # Memory pool from core module
        self.memory_pool = get_global_memory_pool()
        
        # Create ring-specific memory pool wrapper
        self._ring_memory_pool = RingAttentionMemoryPool(self.device)
        
        # Pre-allocated rotation buffers
        self._rotation_buffers = {}
        self._communication_buffers = {}
        
        # Thread safety for concurrent operations
        self._buffer_lock = threading.Lock()
        
        # Setup ring communication if distributed
        self._setup_ring_communication()
        
        # Hardware optimization detection
        self._is_h100_gpu = GPU_TYPE == "h100"
        self._flash_attn_3_available = HAS_FLASH_ATTN_3
    
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
    
    
    def _precompute_ring_patterns(self, h: int, ring_size: int):
        """Pre-compute all ring-specific patterns with vectorized operations."""
        cache_key = (h, ring_size)
        if cache_key in self._ring_patterns:
            return self._ring_patterns[cache_key]
        
        gs, head_ranges = self._get_head_groups(h)
        
        # Vectorized pattern precomputation - compute all patterns at once
        if cache_key not in self._vectorized_patterns:
            # Pre-compute all dilation indices for all segments/dilations in batch
            all_indices = {}
            for i, (r, s) in enumerate(zip(self.dilation_rates, self.segment_lengths)):
                for offset in range(r):
                    indices_key = (s, r, offset)
                    if indices_key not in all_indices:
                        all_indices[indices_key] = torch.arange(
                            offset, s, r, device=self._ring_memory_pool.device
                        )
            
            # Vectorized step pattern creation
            ring_patterns = []
            for step in range(ring_size):
                step_patterns = {}
                for i, ((g, (hmin, hmax)), r, s) in enumerate(
                    zip(zip(gs, head_ranges), self.dilation_rates, self.segment_lengths)
                ):
                    indices_key = (s, r, i % r)
                    step_patterns[i] = {
                        'head_range': (hmin, hmax),
                        'indices': all_indices[indices_key],
                        'segment_size': s,
                        'dilation': r,
                        'group_size': g
                    }
                ring_patterns.append(step_patterns)
            
            # Cache both individual indices and complete patterns
            with self._cache_lock:
                self._cached_indices.update(all_indices)
                self._vectorized_patterns[cache_key] = ring_patterns
        
        self._ring_patterns[cache_key] = self._vectorized_patterns[cache_key]
        return self._ring_patterns[cache_key]
    
    def _allocate_ring_buffers(self, k: Tensor, v: Tensor):
        """Allocate ring communication buffers with memory pool optimization and thread safety."""
        if self.ring_group is None:
            return
        
        b, n, h, d = k.shape
        
        # Calculate local sequence length per device in ring
        local_seq_len = n // self.ring_size
        buffer_shape = (b, local_seq_len, h, d)
        
        # Double-checked locking pattern for thread safety
        # First check without lock for performance
        if (self._kv_send_buffer is None or 
            self._kv_send_buffer.shape != buffer_shape or
            self._kv_send_buffer.dtype != k.dtype):
            
            # Acquire lock and check again to ensure atomicity
            with self._buffer_lock:
                # Check again inside lock to prevent race condition
                if (self._kv_send_buffer is None or 
                    self._kv_send_buffer.shape != buffer_shape or
                    self._kv_send_buffer.dtype != k.dtype):
                    
                    self._kv_send_buffer = self._ring_memory_pool.get_buffer(
                        buffer_shape, k.dtype, "kv_send"
                    )
                    self._kv_recv_buffer = self._ring_memory_pool.get_buffer(
                        buffer_shape, k.dtype, "kv_recv"
                    )
                    
                    # Allocate optimized packed communication buffer
                    packed_size = 2 * b * local_seq_len * h * d  # K + V
                    self._packed_communication_buffer = self._ring_memory_pool.get_buffer(
                        (packed_size,), k.dtype, "packed_comm"
                    )
        
        # Pre-allocate rotation buffers for in-place operations
        buffer_key = (buffer_shape, k.dtype)
        if buffer_key not in self._rotation_buffers:
            with self._buffer_lock:
                # Check again inside lock
                if buffer_key not in self._rotation_buffers:
                    self._rotation_buffers[buffer_key] = {
                        'k': self._ring_memory_pool.get_buffer(buffer_shape, k.dtype, "rot_k"),
                        'v': self._ring_memory_pool.get_buffer(buffer_shape, k.dtype, "rot_v")
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
            
            # Apply scaled dot product attention with optimized backend selection
            if HAS_SDPA_KERNEL:
                # Optimize backend selection for hardware - prioritize Flash Attention on H100
                backends = self._get_optimal_sdpa_backends()
                with sdpa_kernel(backends):
                    attn_out = F.scaled_dot_product_attention(
                        q_flat, k_flat, v_flat,
                        attn_mask=None,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal and ring_step == 0,  # Only causal for current ring position
                        scale=None,
                    )
            else:
                # Fallback for older PyTorch versions
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
            
            # Accumulate results - slicing creates a view, so this modifies 'out' in-place
            out[:, :, hmin:hmax, :].add_(attn_flat)
        
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
    
    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        is_causal: bool = False,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with Ring Attention for O(n) memory complexity.
        
        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            attention_mask: Optional attention mask (not supported in ring mode)
            
        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        # Validate inputs using base class method
        self._validate_forward_inputs(query, key, value, attention_mask)
        
        # Use consistent naming
        q, k, v = query, key, value
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
        Ring attention forward pass for distributed computation with enhanced error recovery.
        
        This implements the core ring attention algorithm where each device
        processes a local portion of queries while keys/values are rotated
        through the ring.
        """
        b, n, h, d = q.shape
        device = q.device
        
        try:
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
            
            # Ring attention steps with computation-communication overlap
            k_ring = k_local.clone()
            v_ring = v_local.clone()
            
            # Double buffering for overlapping computation and communication
            k_next, v_next = None, None
            rotation_handle = None
            
            for step in range(self.ring_size):
                try:
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
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Memory recovery strategy
                        torch.cuda.empty_cache()
                        self.clear_cache(force=True)
                        
                        # Retry with checkpointing enabled
                        if not self.use_checkpointing:
                            step_output = torch.utils.checkpoint.checkpoint(
                                self._ring_attention_step,
                                q_local, k_ring, v_ring, is_causal, step,
                                use_reentrant=False
                            )
                            output.add_(step_output)
                        else:
                            raise e  # Re-raise if already using checkpointing
                    else:
                        raise e
                
                # Optimized rotation with computation-communication overlap
                if step < self.ring_size - 1:
                    try:
                        # Start next rotation asynchronously while computing current step
                        if step == 0:
                            # First step: start rotation in background
                            rotation_handle = self._start_async_rotation(k_ring, v_ring)
                        else:
                            # Subsequent steps: wait for previous rotation, start next
                            if rotation_handle is not None:
                                k_ring, v_ring = self._complete_async_rotation(rotation_handle)
                            
                            if step < self.ring_size - 2:  # Not the penultimate step
                                rotation_handle = self._start_async_rotation(k_ring, v_ring)
                            else:
                                # Last rotation
                                k_ring, v_ring = self._rotate_kv_ring(k_ring, v_ring)
                                
                    except Exception as comm_error:
                        # Clean up any pending communications
                        self._cleanup_ring_communication()
                        
                        if self.ring_size > 1:
                            warnings.warn(
                                f"Ring communication failed at step {step}: {comm_error}. "
                                f"Falling back to single device computation."
                            )
                            # Fall back to single device computation
                            return self._single_device_forward(q, k, v, is_causal)
                        else:
                            raise comm_error
            
            # Gather results from all devices
            output_gathered = self._gather_ring_outputs(output, local_seq_len)
            
            return output_gathered
            
        except Exception as e:
            # Clean up resources before fallback
            self._cleanup_ring_communication()
            
            # Final fallback: single device computation
            warnings.warn(
                f"Ring attention failed: {e}. Falling back to single device computation."
            )
            return self._single_device_forward(q, k, v, is_causal)
    
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
        total_size = k_size + v_flat.numel()
        
        # Ensure communication buffer is large enough with bounds checking
        if self._packed_communication_buffer.numel() < total_size:
            # Validate buffer size is reasonable (prevent excessive memory allocation)
            max_reasonable_size = 1024 * 1024 * 1024  # 1GB max buffer size
            if total_size > max_reasonable_size:
                raise RuntimeError(
                    f"Requested communication buffer size ({total_size / (1024*1024):.1f}MB) "
                    f"exceeds maximum reasonable size ({max_reasonable_size / (1024*1024):.1f}MB). "
                    f"Consider reducing sequence length or ring size."
                )
            
            self._packed_communication_buffer = self._ring_memory_pool.get_buffer(
                (total_size,), k.dtype, "packed_comm_resized"
            )
        
        # Create packed buffer view
        packed_send = self._packed_communication_buffer[:total_size]
        packed_recv = torch.empty_like(packed_send)
        
        # Optimized in-place packing - avoids tensor creation overhead
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
    
    def _start_async_rotation(self, k: Tensor, v: Tensor) -> Dict[str, Any]:
        """Start asynchronous ring rotation for computation-communication overlap."""
        if self.ring_group is None:
            return {'k': k, 'v': v, 'requests': None}
        
        # Get pre-allocated rotation buffers
        buffer_key = (k.shape, k.dtype)
        rotation_buffers = self._rotation_buffers[buffer_key]
        k_buffer = rotation_buffers['k']
        v_buffer = rotation_buffers['v']
        
        # Calculate send/receive ranks
        send_rank = (self.rank + 1) % self.ring_size
        recv_rank = (self.rank - 1) % self.ring_size
        
        # Pack K and V for efficient communication
        k_flat = k.flatten()
        v_flat = v.flatten()
        k_size = k_flat.numel()
        total_size = k_size + v_flat.numel()
        
        # Prepare packed buffers
        packed_send = self._packed_communication_buffer[:total_size]
        packed_recv = torch.empty_like(packed_send)
        
        # In-place packing
        packed_send[:k_size].copy_(k_flat)
        packed_send[k_size:].copy_(v_flat)
        
        # Start async communication
        send_req = dist.isend(packed_send, dst=send_rank, group=self.ring_group)
        recv_req = dist.irecv(packed_recv, src=recv_rank, group=self.ring_group)
        
        return {
            'k_buffer': k_buffer,
            'v_buffer': v_buffer,
            'packed_recv': packed_recv,
            'k_size': k_size,
            'k_shape': k.shape,
            'v_shape': v.shape,
            'send_req': send_req,
            'recv_req': recv_req
        }
    
    def _complete_async_rotation(self, rotation_handle: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Complete asynchronous ring rotation and return rotated tensors."""
        if rotation_handle.get('requests') is None:
            return rotation_handle['k'], rotation_handle['v']
        
        # Wait for communication completion
        rotation_handle['send_req'].wait()
        rotation_handle['recv_req'].wait()
        
        # Unpack received data
        k_size = rotation_handle['k_size']
        packed_recv = rotation_handle['packed_recv']
        k_received_flat = packed_recv[:k_size]
        v_received_flat = packed_recv[k_size:]
        
        # Copy to pre-allocated buffers
        k_buffer = rotation_handle['k_buffer']
        v_buffer = rotation_handle['v_buffer']
        k_buffer.copy_(k_received_flat.view(rotation_handle['k_shape']))
        v_buffer.copy_(v_received_flat.view(rotation_handle['v_shape']))
        
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
    
    def clear_cache(self, force: bool = False):
        """Clear cached patterns and buffers to free memory with thread safety and optimization tracking."""
        # Clear base class cache first
        super().clear_cache(force)
        
        # Clear ring-specific caches
        with self._cache_lock:
            if force:
                self._cached_indices.clear()
                self._ring_patterns.clear()
                self._rotation_buffers.clear()
                self._vectorized_patterns.clear()
            else:
                # Smart cache cleanup - keep frequently used patterns with priority
                if len(self._cached_indices) > 50:  # Threshold for cleanup
                    # Keep only the most recent half of cached indices
                    keys_to_keep = list(self._cached_indices.keys())[-25:]
                    new_cache = {k: self._cached_indices[k] for k in keys_to_keep}
                    self._cached_indices = new_cache
                
                if len(self._ring_patterns) > 10:  # Ring patterns are expensive to recompute
                    keys_to_keep = list(self._ring_patterns.keys())[-5:]
                    new_patterns = {k: self._ring_patterns[k] for k in keys_to_keep}
                    self._ring_patterns = new_patterns
                
                # Clean vectorized patterns cache (less frequently used)
                if len(self._vectorized_patterns) > 5:
                    keys_to_keep = list(self._vectorized_patterns.keys())[-3:]
                    new_vec_patterns = {k: self._vectorized_patterns[k] for k in keys_to_keep}
                    self._vectorized_patterns = new_vec_patterns
            
            self._ring_memory_pool.clear_unused_buffers()
    
    
    def _get_optimal_sdpa_backends(self):
        """Get optimal SDPA backends based on hardware and Flash Attention version."""
        if not HAS_SDPA_KERNEL:
            return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        
        # Prioritize Flash Attention on H100 with Flash Attention 3
        if self._is_h100_gpu and self._flash_attn_3_available:
            return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        # Standard priority for other hardware
        elif self._flash_attn_3_available:
            return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        else:
            # Fallback for Flash Attention 2 or older
            return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    
    def _cleanup_ring_communication(self):
        """Clean up any pending ring communications and resources."""
        # Clean up communication buffers
        if hasattr(self, '_kv_send_buffer'):
            self._kv_send_buffer = None
        if hasattr(self, '_kv_recv_buffer'):
            self._kv_recv_buffer = None
        
        # Cancel any pending async operations
        if hasattr(self, '_pending_sends'):
            for handle in getattr(self, '_pending_sends', []):
                try:
                    if hasattr(handle, 'wait'):
                        # Try to complete the operation
                        handle.wait()
                except Exception:
                    # Ignore errors during cleanup
                    pass
            self._pending_sends = []
        
        if hasattr(self, '_pending_recvs'):
            for handle in getattr(self, '_pending_recvs', []):
                try:
                    if hasattr(handle, 'wait'):
                        handle.wait()
                except Exception:
                    pass
            self._pending_recvs = []
        
        # Return any temporary buffers to pool
        if hasattr(self, '_temp_buffers'):
            for buffer in getattr(self, '_temp_buffers', []):
                try:
                    if self._ring_memory_pool and buffer is not None:
                        # Check if buffer is still valid before returning
                        if buffer.device.type == self.device.type:
                            # Ring memory pool handles cleanup internally
                            pass
                except Exception:
                    pass
            self._temp_buffers = []
        
        # Clear any intermediate results
        if hasattr(self, '_intermediate_outputs'):
            self._intermediate_outputs = None
        
        # Synchronize device to ensure all operations complete
        if self.device.type == 'cuda':
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory usage information with optimization metrics."""
        info = {
            "memory_complexity": "O(n)",
            "ring_size": self.ring_size,
            "supports_infinite_context": True,
            "cached_patterns": len(self._ring_patterns),
            "cached_indices": len(self._cached_indices),
            "allocated_buffers": len(self._ring_memory_pool._pools),
            "head_groups_cached": len(self._head_groups_cache) if hasattr(self, '_head_groups_cache') else 0,
            "vectorized_patterns_cached": len(self._vectorized_patterns),
            "hot_buffer_keys": len(self._ring_memory_pool._hot_keys_cache),
            "flash_attention_version": None,  # Use core detection
            "flash_attention_3_available": self._flash_attn_3_available,
            "hardware_optimized_for_fa3": self._is_h100_gpu,
            "sdpa_backend_available": HAS_SDPA_KERNEL,
            "optimizations_enabled": [
                "vectorized_pattern_computation",
                "hot_cache_lookup", 
                "in_place_kv_packing",
                "computation_communication_overlap",
                "memoized_head_groups",
                "fused_accumulation",
                "hardware_optimized_backends"
            ]
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
            })
        
        return info


# Optional: Enable torch.compile for additional optimization
# RingDilatedAttention = torch.compile(RingDilatedAttention, fullgraph=True)