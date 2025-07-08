"""
GPU utilities for device detection and optimization.

This module provides utilities for detecting GPU architectures and
selecting optimal configurations based on hardware capabilities.
"""

import warnings
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
from dataclasses import dataclass, field

import torch


def get_gpu_compute_capability(device: torch.device) -> Optional[Tuple[int, int]]:
    """
    Get the compute capability of a GPU device.

    Args:
        device: PyTorch device

    Returns:
        Tuple of (major, minor) compute capability or None if not available
    """
    if device.type != "cuda":
        return None

    try:
        # Get device properties
        props = torch.cuda.get_device_properties(device)
        return (props.major, props.minor)
    except Exception:
        return None


def is_pascal_or_older(device: torch.device) -> bool:
    """
    Check if GPU is Pascal architecture or older (compute < 7.0).

    Pascal GPUs have limited FP16 performance and no Tensor Cores.

    Args:
        device: PyTorch device

    Returns:
        True if Pascal or older, False otherwise
    """
    capability = get_gpu_compute_capability(device)
    if capability is None:
        return False

    major, minor = capability
    return major < 7


def get_optimal_dtype(
    device: torch.device, prefer_fp16: bool = True, warn_pascal: bool = True
) -> torch.dtype:
    """
    Get optimal dtype for a device based on its architecture.

    Pascal and older GPUs often perform worse with FP16 due to:
    - No Tensor Core support
    - Limited FP16 compute units
    - Overhead from FP16/FP32 conversions

    Args:
        device: PyTorch device
        prefer_fp16: Whether to prefer FP16 on capable devices
        warn_pascal: Whether to warn about Pascal FP16 performance

    Returns:
        Optimal dtype for the device (torch.float32 or torch.float16)
    """
    if device.type != "cuda":
        return torch.float32

    # Check compute capability
    capability = get_gpu_compute_capability(device)
    if capability is None:
        return torch.float32

    major, minor = capability

    # Pascal and older (compute < 7.0)
    if major < 7:
        if prefer_fp16 and warn_pascal:
            props = torch.cuda.get_device_properties(device)
            warnings.warn(
                f"GPU {props.name} (compute {major}.{minor}) has limited FP16 performance. "
                f"Using FP32 for optimal performance. Pascal GPUs can be 5-10x slower with FP16.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.float32

    # Volta and newer
    if prefer_fp16:
        # Check if bfloat16 is available (Ampere+)
        if major >= 8 and hasattr(torch, "bfloat16") and torch.cuda.is_bf16_supported():
            # Could return torch.bfloat16 here, but FP16 is more widely supported
            return torch.float16
        else:
            return torch.float16
    else:
        return torch.float32


def warn_suboptimal_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """
    Warn if using a suboptimal dtype for the device.

    Args:
        device: PyTorch device
        dtype: Current dtype being used
    """
    if device.type != "cuda":
        return

    capability = get_gpu_compute_capability(device)
    if capability is None:
        return

    major, minor = capability

    # Warn if using FP16 on Pascal
    if major < 7 and dtype in [torch.float16, torch.bfloat16]:
        props = torch.cuda.get_device_properties(device)
        warnings.warn(
            f"Using {dtype} on Pascal GPU {props.name} may result in poor performance. "
            f"Consider using torch.float32 instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Warn if using FP32 on modern GPUs when FP16 would be faster
    elif major >= 7 and dtype == torch.float32:
        props = torch.cuda.get_device_properties(device)
        warnings.warn(
            f"GPU {props.name} supports efficient FP16. "
            f"Consider using torch.float16 for better performance.",
            RuntimeWarning,
            stacklevel=2,
        )


@dataclass
class GPUInfo:
    """Comprehensive GPU information and capabilities."""

    # Basic info
    device: torch.device
    name: str = ""
    compute_capability: Tuple[int, int] = (0, 0)
    architecture: str = "unknown"

    # Memory info
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0

    # Compute capabilities
    cuda_cores: int = 0
    tensor_cores: bool = False
    fp16_performance_tflops: float = 0.0
    fp32_performance_tflops: float = 0.0
    supports_fp8: bool = False
    supports_bf16: bool = False

    # Attention backend support
    has_flash_attn: bool = False
    has_flash_attn_2: bool = False
    has_flash_attn_3: bool = False
    has_xformers: bool = False
    has_sdpa: bool = False
    recommended_backend: str = "manual"

    # Optimal settings
    optimal_dtype: torch.dtype = torch.float32
    optimal_block_size: int = 128
    optimal_num_warps: int = 4

    # Cache for backend benchmarks
    backend_benchmarks: Dict[str, float] = field(default_factory=dict)


class GPUDetector:
    """Singleton GPU detector with comprehensive capabilities."""

    _instance: Optional["GPUDetector"] = None
    _gpu_info_cache: Dict[str, GPUInfo] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @lru_cache(maxsize=8)
    def get_gpu_info(self, device: Optional[torch.device] = None) -> GPUInfo:
        """Get comprehensive GPU information."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to string for caching
        device_key = str(device)

        # Check cache
        if device_key in self._gpu_info_cache:
            return self._gpu_info_cache[device_key]

        # Create new GPUInfo
        info = GPUInfo(device=device)

        if device.type != "cuda":
            info.name = "CPU"
            info.architecture = "cpu"
            info.has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
            info.recommended_backend = "sdpa" if info.has_sdpa else "manual"
            self._gpu_info_cache[device_key] = info
            return info

        # Get device properties
        try:
            props = torch.cuda.get_device_properties(device)
            info.name = props.name
            info.compute_capability = (props.major, props.minor)
            info.total_memory_gb = props.total_memory / (1024**3)
            info.cuda_cores = props.multi_processor_count * self._get_cores_per_sm(
                props.major, props.minor
            )

            # Get available memory
            try:
                info.available_memory_gb = torch.cuda.mem_get_info(device.index)[0] / (
                    1024**3
                )
            except:
                info.available_memory_gb = info.total_memory_gb * 0.9  # Estimate

            # Determine architecture
            info.architecture = self._get_architecture_name(props.major, props.minor)

            # Set architecture-specific features
            self._set_architecture_features(info, props)

            # Check backend support
            self._check_backend_support(info)

            # Determine optimal settings
            self._determine_optimal_settings(info)

        except Exception as e:
            warnings.warn(f"Failed to get GPU properties: {e}", RuntimeWarning)

        self._gpu_info_cache[device_key] = info
        return info

    def _get_cores_per_sm(self, major: int, minor: int) -> int:
        """Get CUDA cores per streaming multiprocessor."""
        # Based on NVIDIA documentation
        if major == 3:  # Kepler
            return 192
        elif major == 5:  # Maxwell
            return 128
        elif major == 6:  # Pascal
            if minor == 0:
                return 64  # GP100
            else:
                return 128
        elif major == 7:  # Volta/Turing
            if minor == 0:
                return 64  # V100
            else:
                return 64  # Turing
        elif major == 8:  # Ampere
            if minor == 0:
                return 64  # A100
            else:
                return 128  # GA102
        elif major == 9:  # Hopper
            return 128  # H100
        else:
            return 64  # Conservative estimate

    def _get_architecture_name(self, major: int, minor: int) -> str:
        """Get architecture name from compute capability."""
        if major < 3:
            return "fermi_or_older"
        elif major == 3:
            return "kepler"
        elif major == 5:
            return "maxwell"
        elif major == 6:
            return "pascal"
        elif major == 7:
            if minor == 0:
                return "volta"
            else:
                return "turing"
        elif major == 8:
            if minor == 0:
                return "ampere_datacenter"  # A100
            elif minor == 6:
                return "ampere_consumer"  # RTX 3000
            else:
                return "ampere"
        elif major == 9:
            return "hopper"
        elif major == 10:
            return "blackwell"  # Future
        else:
            return "future"

    def _set_architecture_features(self, info: GPUInfo, props: Any):
        """Set architecture-specific features."""
        major, minor = info.compute_capability

        # Tensor Cores
        info.tensor_cores = major >= 7

        # FP8 support (Hopper+)
        info.supports_fp8 = major >= 9

        # BF16 support (Ampere+)
        info.supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()

        # Estimate performance (very rough estimates)
        sm_count = props.multi_processor_count
        clock_ghz = props.clock_rate / 1e6  # Convert to GHz

        if major >= 9:  # Hopper
            # H100 has exceptional performance
            info.fp32_performance_tflops = sm_count * clock_ghz * 0.128  # ~67 TFLOPS
            info.fp16_performance_tflops = (
                info.fp32_performance_tflops * 4
            )  # ~268 TFLOPS with tensor cores
            info.memory_bandwidth_gbps = 3350  # H100 SXM
        elif major == 8:  # Ampere
            if minor == 0:  # A100
                info.fp32_performance_tflops = (
                    sm_count * clock_ghz * 0.064
                )  # ~19.5 TFLOPS
                info.fp16_performance_tflops = (
                    info.fp32_performance_tflops * 4
                )  # ~78 TFLOPS
                info.memory_bandwidth_gbps = 1935  # A100 80GB
            else:  # Consumer Ampere
                info.fp32_performance_tflops = sm_count * clock_ghz * 0.032
                info.fp16_performance_tflops = info.fp32_performance_tflops * 2
                info.memory_bandwidth_gbps = 760  # RTX 3090
        elif major == 7:  # Volta/Turing
            if minor == 0:  # V100
                info.fp32_performance_tflops = (
                    sm_count * clock_ghz * 0.032
                )  # ~15.7 TFLOPS
                info.fp16_performance_tflops = (
                    info.fp32_performance_tflops * 2
                )  # ~31.4 TFLOPS
                info.memory_bandwidth_gbps = 900
            else:  # Turing
                info.fp32_performance_tflops = sm_count * clock_ghz * 0.016
                info.fp16_performance_tflops = info.fp32_performance_tflops * 2
                info.memory_bandwidth_gbps = 448  # RTX 2080
        else:  # Pascal and older
            info.fp32_performance_tflops = sm_count * clock_ghz * 0.01
            info.fp16_performance_tflops = (
                info.fp32_performance_tflops
            )  # No tensor cores
            info.memory_bandwidth_gbps = 320  # GTX 1080

    def _check_backend_support(self, info: GPUInfo):
        """Check which attention backends are available."""
        # Check PyTorch SDPA
        info.has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Check Flash Attention
        try:
            import flash_attn

            info.has_flash_attn = True

            # Check version
            version = getattr(flash_attn, "__version__", "0.0.0")
            major_version = int(version.split(".")[0])

            if major_version >= 2:
                info.has_flash_attn_2 = True
            if major_version >= 3:
                info.has_flash_attn_3 = True
        except ImportError:
            pass

        # Check xformers
        try:
            import xformers
            import xformers.ops

            info.has_xformers = True
        except ImportError:
            pass

        # Determine recommended backend
        info.recommended_backend = self._select_recommended_backend(info)

    def _select_recommended_backend(self, info: GPUInfo) -> str:
        """Select the recommended attention backend."""
        major, minor = info.compute_capability

        # Special handling for different architectures
        if major < 7:  # Pascal or older
            if info.has_xformers:
                return "xformers"
            elif info.has_sdpa:
                return "sdpa"
            else:
                return "manual"

        elif major == 7 and minor == 0:  # V100
            # V100 doesn't support Flash Attention but has good SDPA
            if info.has_xformers:
                return "xformers"
            elif info.has_sdpa:
                return "sdpa"
            else:
                return "manual"

        elif major == 7 and minor >= 5:  # Turing (T4, RTX 2000)
            if info.has_flash_attn_2:
                return "flash_attn_2"
            elif info.has_flash_attn:
                return "flash_attn"
            elif info.has_sdpa:
                return "sdpa"
            elif info.has_xformers:
                return "xformers"
            else:
                return "manual"

        elif major == 8:  # Ampere
            if info.has_flash_attn_2:
                return "flash_attn_2"
            elif info.has_flash_attn:
                return "flash_attn"
            elif info.has_sdpa:
                return "sdpa"
            elif info.has_xformers:
                return "xformers"
            else:
                return "manual"

        elif major >= 9:  # Hopper and newer
            if info.has_flash_attn_3:
                return "flash_attn_3"
            elif info.has_flash_attn_2:
                return "flash_attn_2"
            elif info.has_sdpa:
                return "sdpa"
            elif info.has_xformers:
                return "xformers"
            else:
                return "manual"

        # Default fallback
        if info.has_sdpa:
            return "sdpa"
        else:
            return "manual"

    def _determine_optimal_settings(self, info: GPUInfo):
        """Determine optimal settings for the GPU."""
        major, minor = info.compute_capability

        # Optimal dtype
        if major < 7:  # Pascal or older
            info.optimal_dtype = torch.float32
        elif major >= 8 and info.supports_bf16:  # Ampere+ with BF16
            # BF16 often better for training due to larger range
            info.optimal_dtype = torch.bfloat16
        else:
            info.optimal_dtype = torch.float16

        # Optimal block size for attention
        if major >= 9:  # Hopper
            info.optimal_block_size = 256  # Larger blocks on H100
            info.optimal_num_warps = 8
        elif major == 8:  # Ampere
            info.optimal_block_size = 128
            info.optimal_num_warps = 4
        elif major == 7:  # Volta/Turing
            info.optimal_block_size = 64
            info.optimal_num_warps = 4
        else:  # Older
            info.optimal_block_size = 32
            info.optimal_num_warps = 2

    def select_backend_for_config(
        self,
        device: Optional[torch.device] = None,
        seq_len: int = 1024,
        has_custom_mask: bool = False,
        is_causal: bool = False,
        use_dilation: bool = False,
    ) -> str:
        """Select best backend for specific configuration."""
        info = self.get_gpu_info(device)

        # Custom mask limits backend options
        if has_custom_mask or use_dilation:
            # Flash Attention doesn't support custom masks well
            if info.has_xformers:
                return "xformers"
            elif info.has_sdpa:
                return "sdpa_with_mask"
            else:
                return "manual"

        # Very short sequences might be faster with manual
        if seq_len < 128:
            return "manual"

        # For standard attention, use recommended backend
        return info.recommended_backend

    def benchmark_backends(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        seq_len: int = 1024,
        num_heads: int = 8,
        head_dim: int = 64,
        warmup_iters: int = 10,
        bench_iters: int = 100,
    ) -> Dict[str, float]:
        """Benchmark available backends and return timing results."""
        info = self.get_gpu_info(device)

        # Check cache
        cache_key = f"b{batch_size}_s{seq_len}_h{num_heads}_d{head_dim}"
        if cache_key in info.backend_benchmarks:
            return info.backend_benchmarks[cache_key]

        results = {}

        # Create test tensors
        q = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=info.optimal_dtype,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Import what we need
        from ..flash_attention_utils import flash_attention_forward

        # Test each available backend
        backends_to_test = []
        if info.has_flash_attn_3:
            backends_to_test.append("flash_attn_3")
        if info.has_flash_attn_2:
            backends_to_test.append("flash_attn_2")
        if info.has_flash_attn and "flash_attn_2" not in backends_to_test:
            backends_to_test.append("flash_attn")
        if info.has_xformers:
            backends_to_test.append("xformers")
        if info.has_sdpa:
            backends_to_test.append("sdpa")
        backends_to_test.append("standard")  # Always test manual

        for backend in backends_to_test:
            try:
                # Warmup
                for _ in range(warmup_iters):
                    _ = flash_attention_forward(q, k, v, backend=backend)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                import time

                start = time.perf_counter()

                for _ in range(bench_iters):
                    _ = flash_attention_forward(q, k, v, backend=backend)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()

                avg_time_ms = (end - start) * 1000 / bench_iters
                results[backend] = avg_time_ms

            except Exception as e:
                results[backend] = float("inf")  # Failed backend
                if "standard" not in backend:  # Don't warn for standard failures
                    warnings.warn(f"Backend {backend} failed: {e}")

        # Cache results
        info.backend_benchmarks[cache_key] = results

        return results


# Module-level singleton instance
gpu_detector = GPUDetector()


# Convenience functions that use the singleton
def get_gpu_info(device: Optional[torch.device] = None) -> GPUInfo:
    """Get comprehensive GPU information."""
    return gpu_detector.get_gpu_info(device)


def select_attention_backend(
    device: Optional[torch.device] = None,
    seq_len: int = 1024,
    has_custom_mask: bool = False,
    is_causal: bool = False,
    use_dilation: bool = False,
) -> str:
    """Select best attention backend for configuration."""
    return gpu_detector.select_backend_for_config(
        device, seq_len, has_custom_mask, is_causal, use_dilation
    )


def benchmark_attention_backends(
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    seq_len: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
) -> Dict[str, float]:
    """Benchmark available attention backends."""
    return gpu_detector.benchmark_backends(
        device, batch_size, seq_len, num_heads, head_dim
    )
