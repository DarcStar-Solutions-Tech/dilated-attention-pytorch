"""
Constants and feature detection for Dilated Attention implementations.
"""

import logging

import torch

# PyTorch version checks
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

# Feature detection
HAS_SDPA = hasattr(torch.nn.functional, "scaled_dot_product_attention")
HAS_SDPA_KERNEL = hasattr(torch.backends.cuda, "enable_flash_sdp")

# Flash Attention detection
try:
    import flash_attn

    HAS_FLASH_ATTN = True
    FLASH_ATTN_VERSION = getattr(flash_attn, "__version__", "0.0.0")

    # Check for Flash Attention 3
    try:
        from flash_attn_interface import flash_attn_func_v3  # noqa: F401

        HAS_FLASH_ATTN_3 = True
    except ImportError:
        HAS_FLASH_ATTN_3 = False
except ImportError:
    HAS_FLASH_ATTN = False
    HAS_FLASH_ATTN_3 = False
    FLASH_ATTN_VERSION = None

# xFormers detection
try:
    import xformers
    import xformers.ops

    HAS_XFORMERS = True
    XFORMERS_VERSION = xformers.__version__
except ImportError:
    HAS_XFORMERS = False
    XFORMERS_VERSION = None

# DeepSpeed detection
try:
    import deepspeed  # noqa: PLC0415

    HAS_DEEPSPEED = True
    DEEPSPEED_VERSION = deepspeed.__version__
except ImportError:
    HAS_DEEPSPEED = False
    DEEPSPEED_VERSION = None

# FairScale detection
try:
    import fairscale

    HAS_FAIRSCALE = True
    FAIRSCALE_VERSION = fairscale.__version__
except ImportError:
    HAS_FAIRSCALE = False
    FAIRSCALE_VERSION = None

# APEX detection
try:
    import apex

    HAS_APEX = True
    APEX_VERSION = getattr(apex, "__version__", "0.0.0")
except ImportError:
    HAS_APEX = False
    APEX_VERSION = None

# Hardware detection with lazy evaluation
_GPU_TYPE_CACHE = None


def detect_gpu_type() -> str:
    """Detect GPU type for hardware-specific optimizations (cached)."""
    global _GPU_TYPE_CACHE

    if _GPU_TYPE_CACHE is not None:
        return _GPU_TYPE_CACHE

    if not torch.cuda.is_available():
        _GPU_TYPE_CACHE = "cpu"
        return _GPU_TYPE_CACHE

    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
    except Exception:
        # Handle cases where CUDA is available but device query fails
        _GPU_TYPE_CACHE = "generic_cuda"
        return _GPU_TYPE_CACHE

    if "h100" in gpu_name:
        _GPU_TYPE_CACHE = "h100"
    elif "a100" in gpu_name:
        _GPU_TYPE_CACHE = "a100"
    elif "v100" in gpu_name:
        _GPU_TYPE_CACHE = "v100"
    elif "mi300" in gpu_name or "mi250" in gpu_name:
        _GPU_TYPE_CACHE = "amd_instinct"
    elif "rtx 4090" in gpu_name or "rtx 4080" in gpu_name:
        _GPU_TYPE_CACHE = "rtx_40xx"
    elif "rtx 3090" in gpu_name or "rtx 3080" in gpu_name:
        _GPU_TYPE_CACHE = "rtx_30xx"
    else:
        _GPU_TYPE_CACHE = "generic_cuda"

    return _GPU_TYPE_CACHE


# Make it accessible as a module attribute
class _GPUTypeLazy:
    def __repr__(self) -> str:
        return detect_gpu_type()

    def __str__(self) -> str:
        return detect_gpu_type()

    def __eq__(self, other: object) -> bool:
        return detect_gpu_type() == other


GPU_TYPE = _GPUTypeLazy()

# Optimal settings based on hardware
OPTIMAL_SETTINGS = {
    "h100": {
        "use_flash_attn": True,
        "use_tf32": True,
        "block_size": 2048,
        "max_seq_len": 1_000_000,
    },
    "a100": {
        "use_flash_attn": True,
        "use_tf32": True,
        "block_size": 1024,
        "max_seq_len": 100_000,
    },
    "v100": {
        "use_flash_attn": False,
        "use_tf32": False,
        "block_size": 512,
        "max_seq_len": 50_000,
    },
    "generic_cuda": {
        "use_flash_attn": HAS_FLASH_ATTN,
        "use_tf32": True,
        "block_size": 512,
        "max_seq_len": 32_768,
    },
    "cpu": {
        "use_flash_attn": False,
        "use_tf32": False,
        "block_size": 256,
        "max_seq_len": 8192,
    },
}


# Get optimal settings for current hardware (lazy evaluation)
def get_current_optimal_settings() -> dict:
    """Get optimal settings for current hardware."""
    gpu_type = str(GPU_TYPE)  # Force evaluation
    return OPTIMAL_SETTINGS.get(gpu_type, OPTIMAL_SETTINGS["generic_cuda"])


# Lazy access to optimal settings
class _OptimalSettingsLazy(dict):
    def __getitem__(self, key: str) -> object:
        return get_current_optimal_settings()[key]

    def get(self, key: str, default: object = None) -> object:
        return get_current_optimal_settings().get(key, default)

    def __repr__(self) -> str:
        return repr(get_current_optimal_settings())


CURRENT_OPTIMAL_SETTINGS = _OptimalSettingsLazy()

# Setup logging
logger = logging.getLogger("dilated_attention_pytorch")


# Log feature availability on first import
def _log_available_features() -> None:
    """Log available features and optimizations."""
    features = []
    if HAS_FLASH_ATTN:
        features.append(f"Flash Attention {FLASH_ATTN_VERSION}")
        if HAS_FLASH_ATTN_3:
            features.append("Flash Attention 3")
    if HAS_XFORMERS:
        features.append(f"xFormers {XFORMERS_VERSION}")
    if HAS_SDPA:
        features.append("PyTorch SDPA")
    if HAS_DEEPSPEED:
        features.append(f"DeepSpeed {DEEPSPEED_VERSION}")
    if HAS_FAIRSCALE:
        features.append(f"FairScale {FAIRSCALE_VERSION}")
    if HAS_APEX:
        features.append(f"APEX {APEX_VERSION}")

    if features:
        logger.info(f"Available optimizations: {', '.join(features)}")
        logger.info(f"Detected GPU: {GPU_TYPE}")
        logger.debug(f"Optimal settings: {CURRENT_OPTIMAL_SETTINGS}")


# Only log once per process
if not hasattr(_log_available_features, "_called"):
    _log_available_features()
    _log_available_features._called = True
