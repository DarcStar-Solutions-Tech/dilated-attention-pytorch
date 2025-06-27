"""
Utilities for standardizing return values across different attention implementations.
"""

from torch import Tensor


def standardize_attention_output(
    output: Tensor | tuple[Tensor, Tensor | None],
    need_weights: bool = False,
    force_tuple: bool = True,
) -> Tensor | tuple[Tensor, Tensor | None]:
    """
    Standardize attention output to ensure consistent return format.

    Args:
        output: The output from an attention module (either tensor or tuple)
        need_weights: Whether weights were requested
        force_tuple: If True, always return a tuple regardless of need_weights

    Returns:
        Either a tensor or tuple (output, weights) based on parameters
    """
    # If output is already a tuple, extract components
    if isinstance(output, tuple):
        attention_output = output[0]
        attention_weights = output[1] if len(output) > 1 else None
    else:
        # Output is just a tensor
        attention_output = output
        attention_weights = None

    # Return based on preferences
    if force_tuple or need_weights:
        return (attention_output, attention_weights)
    else:
        return attention_output


class MultiheadAttentionWrapper:
    """
    Wrapper to ensure consistent return format for multihead attention modules.

    This wrapper can be used to standardize the output of any multihead attention
    implementation to match the expected interface.
    """

    def __init__(self, attention_module, always_return_tuple: bool = True):
        """
        Initialize the wrapper.

        Args:
            attention_module: The attention module to wrap
            always_return_tuple: If True, always return (output, weights) tuple
        """
        self.attention_module = attention_module
        self.always_return_tuple = always_return_tuple

    def __call__(
        self, query: Tensor, key: Tensor, value: Tensor, need_weights: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """
        Forward pass with standardized return format.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            need_weights: Whether to return attention weights
            **kwargs: Additional arguments for the attention module

        Returns:
            Either tensor or tuple based on configuration
        """
        # Call the wrapped module
        output = self.attention_module(query, key, value, need_weights=need_weights, **kwargs)

        # Standardize the output
        return standardize_attention_output(
            output, need_weights=need_weights, force_tuple=self.always_return_tuple
        )

    def __getattr__(self, name):
        """Forward attribute access to the wrapped module."""
        return getattr(self.attention_module, name)
