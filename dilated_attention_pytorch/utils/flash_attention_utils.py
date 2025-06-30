"""
Flash Attention utilities with GPU architecture-aware fallback.

This module provides optimized attention computation that automatically selects
the best implementation based on GPU architecture and available backends.
"""

import warnings
from typing import Optional

import torch
from torch import Tensor

from .gpu_utils import get_gpu_compute_capability


def get_flash_attention_support(device: torch.device) -> dict:
    """
    Check Flash Attention support for a specific device.

    Args:
        device: PyTorch device to check

    Returns:
        Dict with support information
    """
    support = {
        "has_flash_attn": False,
        "has_flash_attn_2": False,
        "has_flash_attn_3": False,
        "has_xformers": False,
        "recommended_backend": "standard",
        "gpu_architecture": None,
        "compute_capability": None,
        "supports_fp8": False,
    }

    if device.type != "cuda":
        return support

    # Get compute capability
    capability = get_gpu_compute_capability(device)
    if capability is None:
        return support

    major, minor = capability
    support["compute_capability"] = (major, minor)

    # Determine GPU architecture
    if major < 7:
        support["gpu_architecture"] = "pascal_or_older"
    elif major == 7:
        support["gpu_architecture"] = "volta_turing"
    elif major == 8:
        support["gpu_architecture"] = "ampere"
    elif major == 9:
        support["gpu_architecture"] = "hopper"
    else:
        support["gpu_architecture"] = "future"

    # Check Flash Attention availability
    try:
        import flash_attn

        support["has_flash_attn"] = True

        # Check version
        version = getattr(flash_attn, "__version__", "0.0.0")
        major_version = int(version.split(".")[0])

        if major_version >= 2:
            support["has_flash_attn_2"] = True
        if major_version >= 3:
            support["has_flash_attn_3"] = True

    except ImportError:
        pass

    # Check xformers availability
    try:
        import xformers
        import xformers.ops

        _ = xformers.ops  # Mark as used

        support["has_xformers"] = True
    except ImportError:
        pass

    # Determine recommended backend based on architecture
    if major < 7:  # Pascal or older
        # Flash Attention requires SM 7.5+, try xformers first
        if support["has_xformers"]:
            support["recommended_backend"] = "xformers"
        else:
            support["recommended_backend"] = "sdpa"
    elif major == 7 and minor < 5:  # Volta (V100)
        # V100 doesn't support Flash Attention but works well with xformers
        if support["has_xformers"]:
            support["recommended_backend"] = "xformers"
        else:
            support["recommended_backend"] = "sdpa"
    elif major == 7 and minor >= 5:  # Turing (T4, RTX 2000)
        # Supports Flash Attention 1/2
        if support["has_flash_attn"]:
            support["recommended_backend"] = "flash_attn"
        elif support["has_xformers"]:
            support["recommended_backend"] = "xformers"
        else:
            support["recommended_backend"] = "sdpa"
    elif major == 8:  # Ampere (A100, RTX 3000)
        # Excellent Flash Attention 2 support
        if support["has_flash_attn_2"]:
            support["recommended_backend"] = "flash_attn_2"
        elif support["has_flash_attn"]:
            support["recommended_backend"] = "flash_attn"
        elif support["has_xformers"]:
            support["recommended_backend"] = "xformers"
        else:
            support["recommended_backend"] = "sdpa"
    elif major >= 9:  # Hopper (H100) and newer
        # Best with Flash Attention 3
        support["supports_fp8"] = True
        if support["has_flash_attn_3"]:
            support["recommended_backend"] = "flash_attn_3"
        elif support["has_flash_attn_2"]:
            support["recommended_backend"] = "flash_attn_2"
        elif support["has_xformers"]:
            support["recommended_backend"] = "xformers"
        else:
            support["recommended_backend"] = "sdpa"

    return support


def select_attention_backend(
    device: torch.device,
    seq_len: int,
    is_causal: bool = False,
    has_custom_mask: bool = False,
) -> str:
    """
    Select the best attention backend for given parameters.

    Args:
        device: PyTorch device
        seq_len: Sequence length
        is_causal: Whether using causal masking
        has_custom_mask: Whether using custom attention mask

    Returns:
        Backend name: "flash_attn_3", "flash_attn_2", "flash_attn", "xformers", "sdpa", or "standard"
    """
    support = get_flash_attention_support(device)

    # Override for specific conditions
    if has_custom_mask and support["recommended_backend"].startswith("flash_attn"):
        # Flash Attention has limited mask support, try xformers first
        if support["has_xformers"]:
            return "xformers"
        else:
            return "sdpa"

    # For very short sequences, standard might be faster
    if seq_len < 128 and not is_causal:
        return "standard"

    return support["recommended_backend"]


def flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend: Optional[str] = None,
) -> Tensor:
    """
    Compute attention using Flash Attention with automatic fallback.

    Args:
        q, k, v: Query, key, value tensors [batch, seq_len, num_heads, head_dim]
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        backend: Force specific backend (auto-detect if None)

    Returns:
        Attention output [batch, seq_len, num_heads, head_dim]
    """
    device = q.device
    seq_len = q.shape[1]

    # Auto-select backend if not specified
    if backend is None:
        backend = select_attention_backend(device, seq_len, is_causal)

    # Store original shape for reshaping
    original_shape = q.shape
    needs_reshape = q.dim() > 4

    if needs_reshape:
        # Flatten batch dimensions
        q = q.reshape(-1, *q.shape[-3:])
        k = k.reshape(-1, *k.shape[-3:])
        v = v.reshape(-1, *v.shape[-3:])

    try:
        if backend == "flash_attn_3":
            from flash_attn import flash_attn_func_v3

            # Flash Attention 3 specific features
            use_fp8 = (
                device.type == "cuda"
                and "h100" in torch.cuda.get_device_name(device.index).lower()
            )

            output = flash_attn_func_v3(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=is_causal,
                use_fp8=use_fp8,
                enable_async=True,
            )

        elif backend == "flash_attn_2" or backend == "flash_attn":
            from flash_attn import flash_attn_func

            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=is_causal,
            )

        elif backend == "xformers":
            # Use xformers memory efficient attention
            import xformers.ops as xops

            # xformers expects [batch * num_heads, seq_len, head_dim]
            batch_size, seq_len, num_heads, head_dim = q.shape

            # Reshape to xformers format
            q_xf = q.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
            k_xf = k.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
            v_xf = v.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

            # Create attention bias for causal masking if needed
            attn_bias = None
            if is_causal:
                # xformers LowerTriangularMask is more efficient than custom bias
                attn_bias = xops.LowerTriangularMask()

            # Use memory efficient attention
            output_xf = xops.memory_efficient_attention(
                q_xf,
                k_xf,
                v_xf,
                attn_bias=attn_bias,
                p=dropout_p if (hasattr(q, "training") and q.training) else 0.0,
            )

            # Reshape back to [batch, seq_len, num_heads, head_dim]
            output = output_xf.reshape(
                batch_size, num_heads, seq_len, head_dim
            ).transpose(1, 2)

        elif backend == "sdpa":
            # Use PyTorch's scaled_dot_product_attention
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            ):
                # SDPA expects [batch, num_heads, seq_len, head_dim]
                q_t = q.transpose(1, 2)
                k_t = k.transpose(1, 2)
                v_t = v.transpose(1, 2)

                output = torch.nn.functional.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    dropout_p=dropout_p
                    if (hasattr(q, "training") and q.training)
                    else 0.0,
                    is_causal=is_causal,
                )

                # Transpose back
                output = output.transpose(1, 2)

        else:  # standard
            # Manual implementation
            scale = 1.0 / (q.shape[-1] ** 0.5)

            # [batch, num_heads, seq_len, head_dim]
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

            # Apply causal mask
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                    diagonal=1,
                )
                scores.masked_fill_(causal_mask, float("-inf"))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # Dropout
            if dropout_p > 0 and hasattr(q, "training") and q.training:
                attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

            # Apply attention
            output = torch.matmul(attn_weights, v_t)

            # Transpose back
            output = output.transpose(1, 2)

    except Exception as e:
        # Fallback to next best implementation
        if backend == "flash_attn_3":
            warnings.warn(
                f"Flash Attention 3 failed, trying Flash Attention 2: {e}", stacklevel=2
            )
            return flash_attention_forward(
                q, k, v, dropout_p, is_causal, backend="flash_attn_2"
            )
        elif backend == "flash_attn_2":
            warnings.warn(
                f"Flash Attention 2 failed, trying Flash Attention 1: {e}", stacklevel=2
            )
            return flash_attention_forward(
                q, k, v, dropout_p, is_causal, backend="flash_attn"
            )
        elif backend == "flash_attn":
            warnings.warn(f"Flash Attention failed, trying xformers: {e}", stacklevel=2)
            return flash_attention_forward(
                q, k, v, dropout_p, is_causal, backend="xformers"
            )
        elif backend == "xformers":
            warnings.warn(f"xformers failed, trying SDPA: {e}", stacklevel=2)
            return flash_attention_forward(
                q, k, v, dropout_p, is_causal, backend="sdpa"
            )
        elif backend == "sdpa":
            warnings.warn(
                f"SDPA failed, using standard implementation: {e}", stacklevel=2
            )
            return flash_attention_forward(
                q, k, v, dropout_p, is_causal, backend="standard"
            )
        else:
            # Re-raise if standard implementation fails
            raise

    # Reshape back if needed
    if needs_reshape:
        output = output.reshape(*original_shape[:-3], *output.shape[-3:])

    return output


def chunked_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int = 2048,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """
    Compute attention in chunks to reduce memory usage.

    Useful for very long sequences or when OOM occurs.

    Args:
        q, k, v: Query, key, value tensors [batch, seq_len, num_heads, head_dim]
        chunk_size: Size of chunks to process
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking

    Returns:
        Attention output [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    device = q.device

    # Select backend
    backend = select_attention_backend(device, chunk_size, is_causal)

    # Initialize output
    output = torch.zeros_like(q)

    # Process in chunks
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)

        # For causal attention, we need all previous K/V
        if is_causal:
            k_chunk = k[:, :end_i]
            v_chunk = v[:, :end_i]
        else:
            # For non-causal, we can process independently
            k_chunk = k[:, i:end_i]
            v_chunk = v[:, i:end_i]

        q_chunk = q[:, i:end_i]

        # Compute attention for this chunk
        output_chunk = flash_attention_forward(
            q_chunk,
            k_chunk,
            v_chunk,
            dropout_p=dropout_p,
            is_causal=is_causal,
            backend=backend,
        )

        output[:, i:end_i] = output_chunk

    return output
