"""Configuration classes for ring attention implementations."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ...core.config import DilatedAttentionConfig


@dataclass
class RingAttentionConfig(DilatedAttentionConfig):
    """Configuration for ring attention implementations.

    This extends the base DilatedAttentionConfig with ring-specific parameters.
    """

    # Ring communication settings
    ring_size: Optional[int] = None  # None means use world_size
    communication_backend: str = "nccl"  # nccl, gloo, or mpi
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 0.1  # seconds

    # Memory optimization settings
    use_memory_pool: bool = True
    preallocate_buffers: bool = True
    aggressive_memory_cleanup: bool = False

    # Performance settings
    overlap_communication: bool = True  # Overlap comm with computation
    use_fused_kernels: bool = True
    compile_mode: Optional[str] = None  # "default", "reduce-overhead", "max-autotune"

    # Debugging and monitoring
    enable_profiling: bool = False
    log_communication_stats: bool = False
    validate_gradients: bool = False

    # Hilbert curve optimization
    use_hilbert: bool = False
    hilbert_curve_level: int = 8  # For n=2^level resolution

    # Fault tolerance
    checkpoint_frequency: int = 0  # 0 means no checkpointing
    enable_watchdog: bool = False
    watchdog_timeout: float = 60.0  # seconds

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        # Validate communication backend
        valid_backends = ["nccl", "gloo", "mpi"]
        if self.communication_backend not in valid_backends:
            raise ValueError(
                f"Invalid communication backend: {self.communication_backend}. "
                f"Must be one of {valid_backends}"
            )

        # Validate retry settings
        if self.max_retry_attempts < 1:
            raise ValueError("max_retry_attempts must be at least 1")

        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        # Validate Hilbert settings
        if self.use_hilbert and self.hilbert_curve_level < 1:
            raise ValueError("hilbert_curve_level must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Get base class fields from dataclass
        from dataclasses import fields

        base_dict = {}

        # Get fields from parent class
        for field in fields(DilatedAttentionConfig):
            value = getattr(self, field.name)
            # Handle special types
            if field.name == "device" and value is not None:
                base_dict[field.name] = str(value)
            elif field.name == "dtype" and value is not None:
                base_dict[field.name] = str(value)
            else:
                base_dict[field.name] = value

        # Add ring-specific fields
        ring_fields = [
            "ring_size",
            "communication_backend",
            "enable_error_recovery",
            "max_retry_attempts",
            "retry_delay",
            "use_memory_pool",
            "preallocate_buffers",
            "aggressive_memory_cleanup",
            "overlap_communication",
            "use_fused_kernels",
            "compile_mode",
            "enable_profiling",
            "log_communication_stats",
            "validate_gradients",
            "use_hilbert",
            "hilbert_curve_level",
            "checkpoint_frequency",
            "enable_watchdog",
            "watchdog_timeout",
        ]

        for field_name in ring_fields:
            base_dict[field_name] = getattr(self, field_name)

        return base_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RingAttentionConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class RingCommunicationStats:
    """Statistics for ring communication performance monitoring."""

    # Basic counters
    total_sends: int = 0
    total_recvs: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0

    # Error tracking
    failed_sends: int = 0
    failed_recvs: int = 0
    retry_attempts: int = 0
    retry_successes: int = 0

    # Timing statistics (in seconds)
    total_comm_time: float = 0.0
    max_comm_time: float = 0.0
    min_comm_time: float = float("inf")

    # Ring passes
    total_ring_passes: int = 0

    def update_communication(
        self, bytes_transferred: int, comm_time: float, success: bool = True
    ) -> None:
        """Update statistics after a communication operation."""
        if success:
            self.total_sends += 1
            self.total_recvs += 1
            self.total_bytes_sent += bytes_transferred
            self.total_bytes_received += bytes_transferred
            self.total_comm_time += comm_time
            self.max_comm_time = max(self.max_comm_time, comm_time)
            self.min_comm_time = min(self.min_comm_time, comm_time)
        else:
            self.failed_sends += 1
            self.failed_recvs += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_ops = self.total_sends + self.total_recvs

        return {
            "total_operations": total_ops,
            "total_bytes": self.total_bytes_sent + self.total_bytes_received,
            "failure_rate": (self.failed_sends + self.failed_recvs) / max(1, total_ops),
            "avg_comm_time": self.total_comm_time / max(1, self.total_sends),
            "max_comm_time": self.max_comm_time,
            "min_comm_time": self.min_comm_time
            if self.min_comm_time != float("inf")
            else 0.0,
            "retry_success_rate": self.retry_successes / max(1, self.retry_attempts),
        }


@dataclass
class HilbertConfig:
    """Configuration specific to Hilbert curve optimization."""

    # Hilbert curve parameters
    curve_level: int = 8  # Resolution: n = 2^level
    apply_per_segment: bool = True  # Apply per-segment vs global
    min_segment_size: int = 64  # Minimum size to apply Hilbert

    # Performance tuning
    use_cached_indices: bool = True
    cache_size_mb: int = 128
    prefetch_distance: int = 2  # Prefetch next N segments

    # GPU optimization
    use_gpu_kernel: bool = True
    block_size: int = 256  # CUDA block size

    def validate(self) -> None:
        """Validate Hilbert configuration."""
        if self.curve_level < 1 or self.curve_level > 16:
            raise ValueError("curve_level must be between 1 and 16")

        if self.min_segment_size < 1:
            raise ValueError("min_segment_size must be positive")

        if self.cache_size_mb < 0:
            raise ValueError("cache_size_mb must be non-negative")


def create_ring_config(
    segment_lengths: List[int], dilation_rates: List[int], **kwargs
) -> RingAttentionConfig:
    """Convenience function to create ring attention configuration.

    Args:
        segment_lengths: List of segment lengths
        dilation_rates: List of dilation rates
        **kwargs: Additional configuration parameters

    Returns:
        RingAttentionConfig instance
    """
    return RingAttentionConfig(
        segment_lengths=segment_lengths, dilation_rates=dilation_rates, **kwargs
    )


# Preset configurations for common use cases
PRESETS = {
    "development": RingAttentionConfig(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        enable_profiling=True,
        log_communication_stats=True,
        validate_gradients=True,
        enable_error_recovery=True,
    ),
    "production": RingAttentionConfig(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        enable_profiling=False,
        log_communication_stats=False,
        validate_gradients=False,
        enable_error_recovery=True,
        use_fused_kernels=True,
        overlap_communication=True,
        compile_mode="reduce-overhead",
    ),
    "large_scale": RingAttentionConfig(
        segment_lengths=[4096, 8192, 16384, 32768],
        dilation_rates=[1, 2, 4, 8],
        aggressive_memory_cleanup=True,
        use_memory_pool=True,
        preallocate_buffers=True,
        checkpoint_frequency=100,
        enable_watchdog=True,
        compile_mode="max-autotune",
    ),
    "hilbert_optimized": RingAttentionConfig(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        use_hilbert=True,
        hilbert_curve_level=10,
        use_fused_kernels=True,
        overlap_communication=True,
    ),
}


def get_preset_config(preset_name: str) -> RingAttentionConfig:
    """Get a preset configuration.

    Args:
        preset_name: Name of preset ("development", "production", "large_scale", "hilbert_optimized")

    Returns:
        RingAttentionConfig instance

    Raises:
        KeyError: If preset name is not found
    """
    if preset_name not in PRESETS:
        raise KeyError(
            f"Unknown preset: {preset_name}. Available presets: {list(PRESETS.keys())}"
        )

    # Return a copy to avoid modifying the preset
    config = PRESETS[preset_name]
    return RingAttentionConfig(**config.to_dict())
