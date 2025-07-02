"""Test Flash Attention integration with GPU architecture awareness."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from dilated_attention_pytorch.utils.flash_attention_utils import (
    get_flash_attention_support,
    select_attention_backend,
    flash_attention_forward,
)
from dilated_attention_pytorch import (
    RingDilatedAttentionV2Collective,
)


class TestFlashAttentionSupport:
    """Test Flash Attention support detection."""

    def test_cpu_device(self):
        """Test that CPU devices don't support Flash Attention."""
        device = torch.device("cpu")
        support = get_flash_attention_support(device)

        assert support["recommended_backend"] == "standard"
        assert support["gpu_architecture"] is None
        assert not support["has_flash_attn"]

    @patch("torch.cuda.get_device_properties")
    def test_pascal_gpu(self, mock_props):
        """Test Pascal GPU (no Flash Attention support)."""
        mock_device_props = MagicMock()
        mock_device_props.major = 6
        mock_device_props.minor = 1
        mock_device_props.name = "NVIDIA GeForce GTX 1080"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")
        support = get_flash_attention_support(device)

        assert support["gpu_architecture"] == "pascal_or_older"
        assert support["compute_capability"] == (6, 1)
        assert support["recommended_backend"] == "standard"

    @patch("torch.cuda.get_device_properties")
    def test_volta_gpu(self, mock_props):
        """Test Volta GPU (V100 - no Flash Attention)."""
        mock_device_props = MagicMock()
        mock_device_props.major = 7
        mock_device_props.minor = 0
        mock_device_props.name = "Tesla V100-SXM2-16GB"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")
        support = get_flash_attention_support(device)

        assert support["gpu_architecture"] == "volta_turing"
        assert support["compute_capability"] == (7, 0)
        assert support["recommended_backend"] == "sdpa"  # V100 uses SDPA

    @patch("torch.cuda.get_device_properties")
    def test_turing_gpu(self, mock_props):
        """Test Turing GPU (T4, RTX 2080)."""
        mock_device_props = MagicMock()
        mock_device_props.major = 7
        mock_device_props.minor = 5
        mock_device_props.name = "Tesla T4"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")
        support = get_flash_attention_support(device)

        assert support["gpu_architecture"] == "volta_turing"
        assert support["compute_capability"] == (7, 5)
        # Should recommend flash_attn if available, otherwise sdpa
        assert support["recommended_backend"] in ["flash_attn", "sdpa"]

    @patch("torch.cuda.get_device_properties")
    def test_ampere_gpu(self, mock_props):
        """Test Ampere GPU (A100, RTX 3090)."""
        mock_device_props = MagicMock()
        mock_device_props.major = 8
        mock_device_props.minor = 0
        mock_device_props.name = "NVIDIA A100-SXM4-40GB"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")
        support = get_flash_attention_support(device)

        assert support["gpu_architecture"] == "ampere"
        assert support["compute_capability"] == (8, 0)
        # Should recommend flash_attn_2 if available
        assert support["recommended_backend"] in ["flash_attn_2", "flash_attn", "sdpa"]

    @patch("torch.cuda.get_device_properties")
    def test_hopper_gpu(self, mock_props):
        """Test Hopper GPU (H100)."""
        mock_device_props = MagicMock()
        mock_device_props.major = 9
        mock_device_props.minor = 0
        mock_device_props.name = "NVIDIA H100 PCIe"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")
        support = get_flash_attention_support(device)

        assert support["gpu_architecture"] == "hopper"
        assert support["compute_capability"] == (9, 0)
        assert support["supports_fp8"] is True
        # Should recommend flash_attn_3 if available
        assert support["recommended_backend"] in [
            "flash_attn_3",
            "flash_attn_2",
            "sdpa",
        ]


class TestAttentionBackendSelection:
    """Test attention backend selection logic."""

    @patch(
        "dilated_attention_pytorch.utils.flash_attention_utils.get_flash_attention_support"
    )
    def test_short_sequence_fallback(self, mock_support):
        """Test that short sequences use standard attention."""
        mock_support.return_value = {
            "recommended_backend": "flash_attn_2",
        }

        device = torch.device("cuda:0")
        backend = select_attention_backend(device, seq_len=64, is_causal=False)

        assert backend == "standard"  # Short sequences use standard

    @patch(
        "dilated_attention_pytorch.utils.flash_attention_utils.get_flash_attention_support"
    )
    def test_custom_mask_fallback(self, mock_support):
        """Test that custom masks fall back to SDPA."""
        mock_support.return_value = {
            "recommended_backend": "flash_attn_2",
        }

        device = torch.device("cuda:0")
        backend = select_attention_backend(
            device, seq_len=2048, is_causal=False, has_custom_mask=True
        )

        assert backend == "sdpa"  # Custom masks use SDPA


class TestFlashAttentionForward:
    """Test Flash Attention forward pass."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_attention_shapes(self):
        """Test that Flash Attention preserves shapes."""
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        device = torch.device("cuda:0")

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Force standard backend for testing
        output = flash_attention_forward(
            q, k, v, dropout_p=0.0, is_causal=False, backend="standard"
        )

        assert output.shape == q.shape
        assert output.device == q.device
        assert output.dtype == q.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_attention_causal(self):
        """Test causal masking in Flash Attention."""
        batch_size = 1
        seq_len = 8
        num_heads = 1
        head_dim = 16

        device = torch.device("cuda:0")

        # Create simple inputs where causal masking matters
        q = torch.ones(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.ones_like(q)
        v = (
            torch.arange(seq_len, device=device)
            .float()
            .view(1, seq_len, 1, 1)
            .expand_as(q)
        )

        # Non-causal should average all values
        output_non_causal = flash_attention_forward(
            q, k, v, is_causal=False, backend="standard"
        )

        # Causal should only see previous values
        output_causal = flash_attention_forward(
            q, k, v, is_causal=True, backend="standard"
        )

        # First token in causal should only see itself
        assert torch.allclose(output_causal[0, 0, 0, 0], v[0, 0, 0, 0], atol=1e-5)

        # Last token in non-causal should see average of all
        expected_avg = v[0, :, 0, 0].mean()
        assert torch.allclose(output_non_causal[0, -1, 0, 0], expected_avg, atol=1e-5)


class TestRingDilatedAttentionFlash:
    """Test Ring Dilated Attention with Flash Attention."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_ring_attention_creation(self):
        """Test creating Ring Attention with Flash support."""
        device = torch.device("cuda:0")

        model = RingDilatedAttentionV2Collective(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            device=device,
            use_flash_attention=True,
        )

        # Check that Flash support was detected
        assert hasattr(model, "flash_support")
        assert hasattr(model, "flash_backend")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_ring_attention_forward(self):
        """Test forward pass with Flash Attention."""
        device = torch.device("cuda:0")
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        head_dim = 64

        model = RingDilatedAttentionV2Collective(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            device=device,
            dtype=torch.float32,  # Use FP32 for stability
            use_flash_attention=True,
        )

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_vs_standard_equivalence(self):
        """Test that Flash and standard attention produce similar results."""
        device = torch.device("cuda:0")
        torch.manual_seed(42)

        batch_size = 1
        seq_len = 512
        num_heads = 4
        head_dim = 32

        # Create models
        model_flash = RingDilatedAttentionV2Collective(
            segment_lengths=[256],
            dilation_rates=[1],
            device=device,
            dtype=torch.float32,
            use_flash_attention=True,
            ring_size=1,  # Single GPU for simpler comparison
        )

        model_standard = RingDilatedAttentionV2Collective(
            segment_lengths=[256],
            dilation_rates=[1],
            device=device,
            dtype=torch.float32,
            use_flash_attention=False,  # Force standard
            ring_size=1,
        )

        # Same inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward passes
        with torch.no_grad():
            output_flash = model_flash(q, k, v, is_causal=False)
            output_standard = model_standard(q, k, v, is_causal=False)

        # Should be very close (some numerical differences expected)
        assert torch.allclose(output_flash, output_standard, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
