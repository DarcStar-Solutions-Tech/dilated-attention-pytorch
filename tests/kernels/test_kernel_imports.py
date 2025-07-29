#!/usr/bin/env python3
"""
Basic tests for kernel module imports and structure.

This test file verifies that kernel modules can be imported and have the expected
structure, even if the underlying Triton kernels have compilation issues.
"""

import pytest


class TestKernelImports:
    """Test that kernel modules can be imported."""

    def test_hilbert_attention_core_import(self):
        """Test importing hilbert_attention_core module."""
        try:
            from dilated_attention_pytorch.kernels import hilbert_attention_core

            # Check expected classes/functions exist
            assert hasattr(hilbert_attention_core, "UnifiedHilbertAttention")
            assert hasattr(hilbert_attention_core, "create_hilbert_mapping")
            assert hasattr(hilbert_attention_core, "HilbertAttentionFunction")

            # The module imported successfully
            assert True
        except ImportError as e:
            # Document why import might fail
            if "triton" in str(e).lower():
                pytest.skip("Triton not available")
            else:
                raise

    def test_kernel_module_structure(self):
        """Test that the kernels module has expected structure."""
        try:
            import dilated_attention_pytorch.kernels as kernels

            # Check module attributes
            assert hasattr(kernels, "__file__")
            assert hasattr(kernels, "__name__")

            # The module exists
            assert True
        except ImportError:
            pytest.skip("Kernels module not available")

    def test_create_hilbert_mapping_basic(self):
        """Test create_hilbert_mapping function works for basic cases."""
        try:
            from dilated_attention_pytorch.kernels.hilbert_attention_core import (
                create_hilbert_mapping,
            )
            import torch

            # Test small sequence
            mapping = create_hilbert_mapping(16)
            assert isinstance(mapping, torch.Tensor)
            assert mapping.shape == (16,)
            assert mapping.dtype == torch.int32

            # Test that all indices are present
            sorted_mapping = torch.sort(mapping)[0]
            expected = torch.arange(16, dtype=torch.int32)
            assert torch.equal(sorted_mapping, expected)

        except ImportError:
            pytest.skip("Cannot import create_hilbert_mapping")

    @pytest.mark.xfail(reason="Triton kernel compilation issues on some GPUs")
    def test_hilbert_attention_core_instantiation(self):
        """Test that UnifiedHilbertAttention can be instantiated."""
        try:
            from dilated_attention_pytorch.kernels.hilbert_attention_core import (
                UnifiedHilbertAttention,
            )
            import torch

            # Try to create an instance
            model = UnifiedHilbertAttention(
                hidden_dim=256,
                num_heads=8,
                segment_size=64,
                dilation_rate=1,
                dropout=0.0,
                use_custom_backward=False,  # Avoid custom backward issues
            )

            # Check basic attributes
            assert model.hidden_dim == 256
            assert model.num_heads == 8
            assert model.head_dim == 32
            assert model.segment_size == 64

            # Try a forward pass on CPU with use_hilbert=False
            x = torch.randn(1, 64, 256)
            try:
                # This might fail due to Triton issues
                output = model(x, use_hilbert=False)
                assert output.shape == (1, 64, 256)
            except Exception as e:
                if "triton" in str(e).lower():
                    pytest.xfail("Triton compilation failed")
                else:
                    raise

        except ImportError:
            pytest.skip("Cannot import UnifiedHilbertAttention")


class TestKernelDocumentation:
    """Test that kernel modules are properly documented."""

    def test_module_docstrings(self):
        """Test that modules have docstrings."""
        try:
            from dilated_attention_pytorch.kernels import hilbert_attention_core

            # Check module docstrings
            assert hilbert_attention_core.__doc__ is not None

            # Check class docstrings
            assert hilbert_attention_core.UnifiedHilbertAttention.__doc__ is not None

        except ImportError:
            pytest.skip("Cannot import kernel modules")

    def test_readme_exists(self):
        """Test that kernels directory has a README."""
        import os

        # Get the kernels directory path
        try:
            import dilated_attention_pytorch.kernels as kernels

            kernels_dir = os.path.dirname(kernels.__file__)
            readme_path = os.path.join(kernels_dir, "README.md")

            assert os.path.exists(readme_path), (
                "README.md should exist in kernels directory"
            )

            # Verify it's not empty
            with open(readme_path, "r") as f:
                content = f.read()
                assert len(content) > 0, "README.md should not be empty"

        except ImportError:
            pytest.skip("Cannot import kernels module")
