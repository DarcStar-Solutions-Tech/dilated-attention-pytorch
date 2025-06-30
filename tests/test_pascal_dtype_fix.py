"""Test smart dtype selection for Pascal GPUs."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from dilated_attention_pytorch.utils.gpu_utils import (
    get_gpu_compute_capability,
    is_pascal_or_older,
    get_optimal_dtype,
    warn_suboptimal_dtype,
)
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


class TestGPUUtils:
    """Test GPU utility functions."""

    def test_get_gpu_compute_capability_cpu(self):
        """Test compute capability on CPU returns None."""
        device = torch.device("cpu")
        assert get_gpu_compute_capability(device) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_compute_capability_cuda(self):
        """Test compute capability on CUDA returns tuple."""
        device = torch.device("cuda:0")
        capability = get_gpu_compute_capability(device)
        assert isinstance(capability, tuple)
        assert len(capability) == 2
        assert all(isinstance(x, int) for x in capability)

    def test_is_pascal_or_older_cpu(self):
        """Test Pascal check on CPU returns False."""
        device = torch.device("cpu")
        assert not is_pascal_or_older(device)

    def test_get_optimal_dtype_cpu(self):
        """Test optimal dtype for CPU is always float32."""
        device = torch.device("cpu")
        assert get_optimal_dtype(device) == torch.float32
        assert get_optimal_dtype(device, prefer_fp16=True) == torch.float32
        assert get_optimal_dtype(device, prefer_fp16=False) == torch.float32

    @patch("torch.cuda.get_device_properties")
    def test_get_optimal_dtype_pascal(self, mock_props):
        """Test optimal dtype for Pascal GPU is float32."""
        # Mock Pascal GPU (GTX 1080)
        mock_device_props = MagicMock()
        mock_device_props.major = 6
        mock_device_props.minor = 1
        mock_device_props.name = "NVIDIA GeForce GTX 1080"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")

        # Should return float32 and warn
        with pytest.warns(RuntimeWarning, match="limited FP16 performance"):
            dtype = get_optimal_dtype(device, prefer_fp16=True, warn_pascal=True)
        assert dtype == torch.float32

        # Without warning
        dtype = get_optimal_dtype(device, prefer_fp16=True, warn_pascal=False)
        assert dtype == torch.float32

    @patch("torch.cuda.get_device_properties")
    def test_get_optimal_dtype_volta(self, mock_props):
        """Test optimal dtype for Volta GPU is float16 when preferred."""
        # Mock Volta GPU (V100)
        mock_device_props = MagicMock()
        mock_device_props.major = 7
        mock_device_props.minor = 0
        mock_device_props.name = "Tesla V100-SXM2-16GB"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")

        # Should return float16 when preferred
        dtype = get_optimal_dtype(device, prefer_fp16=True)
        assert dtype == torch.float16

        # Should return float32 when not preferred
        dtype = get_optimal_dtype(device, prefer_fp16=False)
        assert dtype == torch.float32

    @patch("torch.cuda.get_device_properties")
    def test_warn_suboptimal_dtype_pascal_fp16(self, mock_props):
        """Test warning when using FP16 on Pascal."""
        # Mock Pascal GPU
        mock_device_props = MagicMock()
        mock_device_props.major = 6
        mock_device_props.minor = 1
        mock_device_props.name = "NVIDIA GeForce GTX 1080"
        mock_props.return_value = mock_device_props

        device = torch.device("cuda:0")

        # Should warn for FP16
        with pytest.warns(RuntimeWarning, match="5-10x slower"):
            warn_suboptimal_dtype(device, torch.float16)

        # Should warn for bfloat16 too
        with pytest.warns(RuntimeWarning, match="5-10x slower"):
            warn_suboptimal_dtype(device, torch.bfloat16)

        # Should not warn for FP32
        with pytest.warns(None) as record:
            warn_suboptimal_dtype(device, torch.float32)
        assert len(record) == 0


class TestRingAttentionDtypeSelection:
    """Test dtype selection in Ring Attention."""

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available")
    def test_ring_attention_pascal_auto_dtype(self, mock_cuda_avail, mock_props):
        """Test Ring Attention auto-selects FP32 on Pascal."""
        mock_cuda_avail.return_value = True

        # Mock Pascal GPU
        mock_device_props = MagicMock()
        mock_device_props.major = 6
        mock_device_props.minor = 1
        mock_device_props.name = "NVIDIA GeForce GTX 1080"
        mock_props.return_value = mock_device_props

        # Create model without specifying dtype
        with pytest.warns(RuntimeWarning, match="limited FP16 performance"):
            model = RingDilatedAttentionV2Collective(
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                device=torch.device("cuda:0"),
                # dtype not specified - should auto-select
            )

        # Should have selected FP32
        assert model.dtype == torch.float32

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available")
    def test_ring_attention_volta_auto_dtype(self, mock_cuda_avail, mock_props):
        """Test Ring Attention auto-selects FP16 on Volta."""
        mock_cuda_avail.return_value = True

        # Mock Volta GPU
        mock_device_props = MagicMock()
        mock_device_props.major = 7
        mock_device_props.minor = 0
        mock_device_props.name = "Tesla V100-SXM2-16GB"
        mock_props.return_value = mock_device_props

        # Create model without specifying dtype
        model = RingDilatedAttentionV2Collective(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=torch.device("cuda:0"),
            # dtype not specified - should auto-select
        )

        # Should have selected FP16
        assert model.dtype == torch.float16

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available")
    def test_ring_attention_explicit_dtype_warning(self, mock_cuda_avail, mock_props):
        """Test Ring Attention warns when explicitly using FP16 on Pascal."""
        mock_cuda_avail.return_value = True

        # Mock Pascal GPU
        mock_device_props = MagicMock()
        mock_device_props.major = 6
        mock_device_props.minor = 1
        mock_device_props.name = "NVIDIA GeForce GTX 1080"
        mock_props.return_value = mock_device_props

        # Create model with explicit FP16
        with pytest.warns(RuntimeWarning, match="5-10x slower"):
            model = RingDilatedAttentionV2Collective(
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                device=torch.device("cuda:0"),
                dtype=torch.float16,  # Explicitly set
            )

        # Should respect explicit dtype
        assert model.dtype == torch.float16

    def test_ring_attention_cpu_dtype(self):
        """Test Ring Attention uses FP32 on CPU."""
        model = RingDilatedAttentionV2Collective(
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=torch.device("cpu"),
        )

        assert model.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
