"""
Validation utilities for Dilated Attention implementations.

This module provides a mixin class with common validation methods that can be
used across all dilated attention implementations to ensure consistent error
checking and input validation.
"""

from collections.abc import Sequence

import torch


class ValidationMixin:
    """
    Mixin providing common validation methods for dilated attention.

    This class provides reusable validation logic that ensures:
    - Input tensors have correct shapes and dimensions
    - Configuration parameters are valid
    - Device and dtype consistency
    - Sequence length compatibility
    """

    @staticmethod
    def validate_segment_dilation_match(
        segment_lengths: Sequence[int], dilation_rates: Sequence[int]
    ) -> None:
        """
        Validate that segment lengths and dilation rates have matching lengths.

        Args:
            segment_lengths: List of segment lengths
            dilation_rates: List of dilation rates

        Raises:
            ValueError: If lengths don't match or lists are empty
        """
        if not segment_lengths:
            raise ValueError("segment_lengths cannot be empty")
        if not dilation_rates:
            raise ValueError("dilation_rates cannot be empty")
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError(
                f"segment_lengths and dilation_rates must have the same length: "
                f"{len(segment_lengths)} != {len(dilation_rates)}"
            )

    @staticmethod
    def validate_positive_values(values: Sequence[int | float], name: str) -> None:
        """
        Validate that all values in a sequence are positive.

        Args:
            values: Sequence of values to check
            name: Name of the parameter for error messages

        Raises:
            ValueError: If any value is not positive
        """
        for i, val in enumerate(values):
            if val <= 0:
                raise ValueError(f"{name}[{i}] must be positive, got {val}")

    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_dims: int, name: str) -> None:
        """
        Validate that a tensor has the expected number of dimensions.

        Args:
            tensor: Tensor to validate
            expected_dims: Expected number of dimensions
            name: Name of the tensor for error messages

        Raises:
            ValueError: If tensor dimensions don't match expected
        """
        if tensor.dim() != expected_dims:
            raise ValueError(
                f"{name} expected {expected_dims}D tensor, got {tensor.dim()}D. "
                f"Shape: {tensor.shape}"
            )

    @staticmethod
    def validate_tensor_shapes_match(
        tensors: list[torch.Tensor],
        names: list[str],
        dims_to_check: list[int] | None = None,
    ) -> None:
        """
        Validate that multiple tensors have matching shapes.

        Args:
            tensors: List of tensors to compare
            names: Names of tensors for error messages
            dims_to_check: Specific dimensions to check (None = all dims)

        Raises:
            ValueError: If shapes don't match
        """
        if not tensors:
            return

        base_shape = tensors[0].shape

        for i, (tensor, name) in enumerate(zip(tensors[1:], names[1:], strict=False), 1):
            if dims_to_check is None:
                if tensor.shape != base_shape:
                    raise ValueError(
                        f"Shape mismatch: {names[0]} has shape {base_shape}, "
                        f"{name} has shape {tensor.shape}"
                    )
            else:
                for dim in dims_to_check:
                    if tensor.shape[dim] != base_shape[dim]:
                        raise ValueError(
                            f"Shape mismatch at dimension {dim}: "
                            f"{names[0]} has size {base_shape[dim]}, "
                            f"{name} has size {tensor.shape[dim]}"
                        )

    @staticmethod
    def validate_sequence_length(seq_len: int, segment_lengths: Sequence[int]) -> None:
        """
        Validate that sequence length is compatible with segment lengths.

        Args:
            seq_len: Sequence length to validate
            segment_lengths: List of segment lengths

        Raises:
            ValueError: If sequence length is not divisible by max segment length
        """
        if not segment_lengths:
            raise ValueError("segment_lengths cannot be empty")

        max_segment = max(segment_lengths)
        if seq_len % max_segment != 0:
            raise ValueError(
                f"Sequence length ({seq_len}) must be divisible by "
                f"the largest segment length ({max_segment}). "
                f"Consider padding sequence to {((seq_len // max_segment) + 1) * max_segment}"
            )

    @staticmethod
    def validate_device_dtype_consistency(
        tensors: Sequence[torch.Tensor], names: Sequence[str]
    ) -> None:
        """
        Validate that all tensors have consistent device and dtype.

        Args:
            tensors: List of tensors to check
            names: Names of tensors for error messages

        Raises:
            ValueError: If devices or dtypes don't match
        """
        if not tensors:
            return

        base_device = tensors[0].device
        base_dtype = tensors[0].dtype

        for tensor, name in zip(tensors, names, strict=False):
            if tensor.device != base_device:
                raise ValueError(
                    f"Device mismatch: {names[0]} on {base_device}, {name} on {tensor.device}"
                )
            if tensor.dtype != base_dtype:
                raise ValueError(
                    f"Dtype mismatch: {names[0]} has {base_dtype}, {name} has {tensor.dtype}"
                )

    @staticmethod
    def validate_batch_first_format(
        tensor: torch.Tensor, name: str, expected_dims: int = 4
    ) -> None:
        """
        Validate tensor is in batch-first format.

        Args:
            tensor: Tensor to validate
            name: Name for error messages
            expected_dims: Expected number of dimensions

        Raises:
            ValueError: If tensor is not in correct format
        """
        if tensor.dim() != expected_dims:
            raise ValueError(
                f"{name} must be {expected_dims}D tensor in batch-first format, "
                f"got {tensor.dim()}D tensor with shape {tensor.shape}"
            )

    @staticmethod
    def validate_attention_mask(
        mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        name: str = "attention_mask",
    ) -> None:
        """
        Validate attention mask shape and values.

        Args:
            mask: Optional attention mask
            batch_size: Expected batch size
            seq_len: Expected sequence length
            name: Name for error messages

        Raises:
            ValueError: If mask shape or values are invalid
        """
        if mask is None:
            return

        # Check shape - allow both 2D and 4D masks
        if mask.dim() == 2:
            if mask.shape != (seq_len, seq_len):
                raise ValueError(
                    f"{name} with 2D shape must be ({seq_len}, {seq_len}), got {mask.shape}"
                )
        elif mask.dim() == 4:
            if mask.shape != (batch_size, 1, seq_len, seq_len):
                raise ValueError(
                    f"{name} with 4D shape must be ({batch_size}, 1, {seq_len}, {seq_len}), "
                    f"got {mask.shape}"
                )
        else:
            raise ValueError(f"{name} must be either 2D or 4D, got {mask.dim()}D")

        # Check for boolean or float mask
        if mask.dtype not in [torch.bool, torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(f"{name} must have boolean or float dtype, got {mask.dtype}")

    @staticmethod
    def validate_dropout_prob(dropout: float, name: str = "dropout") -> None:
        """
        Validate dropout probability is in valid range.

        Args:
            dropout: Dropout probability to validate
            name: Name for error messages

        Raises:
            ValueError: If dropout is not in [0, 1]
        """
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {dropout}")

    @staticmethod
    def validate_head_dim(embed_dim: int, num_heads: int) -> int:
        """
        Validate and compute head dimension.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads

        Returns:
            Head dimension

        Raises:
            ValueError: If dimensions are incompatible
        """
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        head_dim = embed_dim // num_heads

        # Warn about suboptimal head dimensions
        import warnings

        if head_dim % 8 != 0:
            warnings.warn(
                f"head_dim ({head_dim}) should be divisible by 8 for "
                f"optimal performance on modern GPUs"
            )

        if head_dim > 128:
            warnings.warn(
                f"head_dim ({head_dim}) > 128 may result in suboptimal "
                f"performance. Consider using more heads with smaller dimensions."
            )

        return head_dim

    @staticmethod
    def validate_num_heads(num_heads: int, num_groups: int) -> None:
        """
        Validate number of heads is compatible with number of groups.

        Args:
            num_heads: Number of attention heads
            num_groups: Number of dilated attention groups

        Raises:
            ValueError: If heads < groups
        """
        if num_heads < num_groups:
            raise ValueError(f"num_heads ({num_heads}) must be >= num_groups ({num_groups})")
