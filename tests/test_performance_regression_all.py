#!/usr/bin/env python3
"""
Performance regression test suite for ALL dilated attention implementations.

This comprehensive suite tests:
- Standard implementations (DilatedAttention, MultiheadDilatedAttention)
- Improved implementations
- Ring attention implementations
- Block sparse implementations
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch

# Import all implementations
from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
)

# Try to import ring implementations
try:
    from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
    from dilated_attention_pytorch.ring_multihead_dilated_attention import (
        RingMultiheadDilatedAttention,
    )

    RING_AVAILABLE = True
except ImportError:
    RING_AVAILABLE = False
    print("Warning: Ring implementations not available for testing")

# Try to import block sparse implementations
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
    )

    BLOCK_SPARSE_AVAILABLE = True
except ImportError:
    BLOCK_SPARSE_AVAILABLE = False
    print("Warning: Block sparse implementations not available for testing")


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.execution_time_ms: float = 0.0
        self.memory_allocated_mb: float = 0.0
        self.memory_reserved_mb: float = 0.0
        self.flops: int | None = None
        self.cuda_time_ms: float | None = None
        self.implementation_details: dict[str, any] = {}

    def to_dict(self) -> dict:
        return {
            "execution_time_ms": self.execution_time_ms,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "flops": self.flops,
            "cuda_time_ms": self.cuda_time_ms,
            "implementation_details": self.implementation_details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformanceMetrics":
        metrics = cls()
        metrics.execution_time_ms = data.get("execution_time_ms", 0.0)
        metrics.memory_allocated_mb = data.get("memory_allocated_mb", 0.0)
        metrics.memory_reserved_mb = data.get("memory_reserved_mb", 0.0)
        metrics.flops = data.get("flops")
        metrics.cuda_time_ms = data.get("cuda_time_ms")
        metrics.implementation_details = data.get("implementation_details", {})
        return metrics


class PerformanceBaseline:
    """Manages performance baselines for regression testing."""

    def __init__(self, baseline_dir: str = "tests/performance_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.baseline_dir / "baselines_all.json"
        self.history_file = self.baseline_dir / "history_all.json"
        self.baselines = self._load_baselines()
        self.history = self._load_history()

    def _load_baselines(self) -> dict[str, dict[str, PerformanceMetrics]]:
        """Load baseline performance metrics."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                data = json.load(f)
                baselines = {}
                for impl, configs in data.items():
                    baselines[impl] = {}
                    for config, metrics_dict in configs.items():
                        baselines[impl][config] = PerformanceMetrics.from_dict(metrics_dict)
                return baselines
        return {}

    def _load_history(self) -> list[dict]:
        """Load performance history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def save_baselines(self):
        """Save baselines to file."""
        data = {}
        for impl, configs in self.baselines.items():
            data[impl] = {}
            for config, metrics in configs.items():
                data[impl][config] = metrics.to_dict()

        with open(self.baseline_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_history(self):
        """Save history to file."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def update_baseline(self, implementation: str, config: str, metrics: PerformanceMetrics):
        """Update baseline for a specific implementation and configuration."""
        if implementation not in self.baselines:
            self.baselines[implementation] = {}
        self.baselines[implementation][config] = metrics
        self.save_baselines()

    def add_history_entry(
        self,
        implementation: str,
        config: str,
        metrics: PerformanceMetrics,
        passed: bool,
        regression_pct: float,
    ):
        """Add entry to performance history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "implementation": implementation,
            "config": config,
            "metrics": metrics.to_dict(),
            "passed": passed,
            "regression_pct": regression_pct,
        }
        self.history.append(entry)
        self.save_history()

    def get_baseline(self, implementation: str, config: str) -> PerformanceMetrics | None:
        """Get baseline metrics for comparison."""
        return self.baselines.get(implementation, {}).get(config)


def measure_performance(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    warmup_iterations: int = 3,
    measure_iterations: int = 10,
    use_cuda_events: bool = True,
) -> PerformanceMetrics:
    """Measure performance metrics for a module."""
    device = inputs[0].device
    metrics = PerformanceMetrics()

    # Warmup
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = module(*inputs)

    # Synchronize before measurement
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Time measurement
    if device.type == "cuda" and use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        start_time = time.perf_counter()

        for _ in range(measure_iterations):
            with torch.no_grad():
                _ = module(*inputs)

        end_event.record()
        end_time = time.perf_counter()

        torch.cuda.synchronize()
        metrics.cuda_time_ms = start_event.elapsed_time(end_event) / measure_iterations
        metrics.execution_time_ms = (end_time - start_time) * 1000 / measure_iterations
    else:
        start_time = time.perf_counter()

        for _ in range(measure_iterations):
            with torch.no_grad():
                _ = module(*inputs)

        end_time = time.perf_counter()
        metrics.execution_time_ms = (end_time - start_time) * 1000 / measure_iterations

    # Memory measurement
    if device.type == "cuda":
        metrics.memory_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        metrics.memory_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    # Store implementation details
    if hasattr(module, "segment_lengths"):
        metrics.implementation_details["segment_lengths"] = module.segment_lengths
    if hasattr(module, "dilation_rates"):
        metrics.implementation_details["dilation_rates"] = module.dilation_rates
    if hasattr(module, "ring_size"):
        metrics.implementation_details["ring_size"] = module.ring_size
    if hasattr(module, "sparsity_config"):
        metrics.implementation_details["sparsity_config"] = module.sparsity_config

    return metrics


def create_test_config(
    seq_len: int, batch_size: int = 1, num_heads: int = 8, head_dim: int = 64
) -> str:
    """Create configuration string for test."""
    return f"b{batch_size}_s{seq_len}_h{num_heads}_d{head_dim}"


@pytest.mark.performance
class TestPerformanceRegressionAll:
    """Performance regression tests for all dilated attention implementations."""

    # Regression threshold (percentage)
    REGRESSION_THRESHOLD = 20.0  # Allow up to 20% performance degradation

    # Test configurations
    CONFIGS = [  # noqa: RUF012
        (1, 2048, 8, 64),  # Small
        (1, 4096, 8, 64),  # Medium
        (1, 8192, 8, 64),  # Large
        (2, 2048, 8, 64),  # Small batch
    ]

    # Ring attention specific configs (must be divisible by ring_size * segment_length)
    RING_CONFIGS = [  # noqa: RUF012
        (1, 2048, 8, 64),  # Small
        (1, 4096, 8, 64),  # Medium
        (1, 8192, 8, 64),  # Large (will use ring_size=4)
    ]

    @pytest.fixture
    def baseline_manager(self):
        """Create baseline manager."""
        return PerformanceBaseline()

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_regression(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
        metric_name: str = "execution_time_ms",
    ) -> tuple[bool, float]:
        """Check if there's a performance regression."""
        baseline_value = getattr(baseline, metric_name)
        current_value = getattr(current, metric_name)

        if baseline_value == 0:
            return True, 0.0

        regression_pct = ((current_value - baseline_value) / baseline_value) * 100
        passed = regression_pct <= self.REGRESSION_THRESHOLD

        return passed, regression_pct

    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", CONFIGS)
    def test_dilated_attention_performance(
        self, baseline_manager, device, batch_size, seq_len, num_heads, head_dim
    ):
        """Test DilatedAttention performance regression."""
        config = create_test_config(seq_len, batch_size, num_heads, head_dim)
        implementation = "DilatedAttention"

        # Create module
        segment_lengths = [seq_len // 4, seq_len // 2, seq_len]
        dilation_rates = [1, 2, 4]
        module = DilatedAttention(segment_lengths, dilation_rates).to(device)

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Measure performance
        metrics = measure_performance(module, (q, k, v))

        # Compare with baseline
        baseline = baseline_manager.get_baseline(implementation, config)
        if baseline is None:
            baseline_manager.update_baseline(implementation, config, metrics)
            pytest.skip(f"No baseline for {implementation} {config}. Setting current as baseline.")

        # Check for regression
        passed, regression_pct = self.check_regression(baseline, metrics)
        baseline_manager.add_history_entry(implementation, config, metrics, passed, regression_pct)

        # Report results
        print(f"\n{implementation} {config}:")
        print(f"  Baseline: {baseline.execution_time_ms:.2f}ms")
        print(f"  Current:  {metrics.execution_time_ms:.2f}ms")
        print(f"  Change:   {regression_pct:+.1f}%")

        if device.type == "cuda":
            print(f"  Memory:   {metrics.memory_allocated_mb:.1f}MB allocated")

        assert (
            passed
        ), f"Performance regression detected: {regression_pct:.1f}% > {self.REGRESSION_THRESHOLD}%"

    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", CONFIGS)
    def test_improved_dilated_attention_performance(
        self, baseline_manager, device, batch_size, seq_len, num_heads, head_dim
    ):
        """Test ImprovedDilatedAttention performance regression."""
        config = create_test_config(seq_len, batch_size, num_heads, head_dim)
        implementation = "ImprovedDilatedAttention"

        # Create module
        segment_lengths = [seq_len // 4, seq_len // 2, seq_len]
        dilation_rates = [1, 2, 4]
        module = ImprovedDilatedAttention(segment_lengths, dilation_rates).to(device)

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Measure performance
        metrics = measure_performance(module, (q, k, v))

        # Compare with baseline
        baseline = baseline_manager.get_baseline(implementation, config)
        if baseline is None:
            baseline_manager.update_baseline(implementation, config, metrics)
            pytest.skip(f"No baseline for {implementation} {config}. Setting current as baseline.")

        # Check for regression
        passed, regression_pct = self.check_regression(baseline, metrics)
        baseline_manager.add_history_entry(implementation, config, metrics, passed, regression_pct)

        print(f"\n{implementation} {config}:")
        print(f"  Baseline: {baseline.execution_time_ms:.2f}ms")
        print(f"  Current:  {metrics.execution_time_ms:.2f}ms")
        print(f"  Change:   {regression_pct:+.1f}%")

        assert (
            passed
        ), f"Performance regression detected: {regression_pct:.1f}% > {self.REGRESSION_THRESHOLD}%"

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_heads,head_dim", CONFIGS[:2]
    )  # Skip large for multihead
    def test_multihead_dilated_attention_performance(
        self, baseline_manager, device, batch_size, seq_len, num_heads, head_dim
    ):
        """Test MultiheadDilatedAttention performance regression."""
        config = create_test_config(seq_len, batch_size, num_heads, head_dim)
        implementation = "MultiheadDilatedAttention"

        # Create module
        embed_dim = num_heads * head_dim
        segment_lengths = [seq_len // 4, seq_len // 2, seq_len]
        dilation_rates = [1, 2, 4]
        module = MultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        ).to(device)

        # Create inputs (multihead uses different format)
        q = torch.randn(batch_size, seq_len, embed_dim, device=device)
        k = torch.randn(batch_size, seq_len, embed_dim, device=device)
        v = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Measure performance
        metrics = measure_performance(module, (q, k, v))

        # Compare with baseline
        baseline = baseline_manager.get_baseline(implementation, config)
        if baseline is None:
            baseline_manager.update_baseline(implementation, config, metrics)
            pytest.skip(f"No baseline for {implementation} {config}. Setting current as baseline.")

        # Check for regression
        passed, regression_pct = self.check_regression(baseline, metrics)
        baseline_manager.add_history_entry(implementation, config, metrics, passed, regression_pct)

        print(f"\n{implementation} {config}:")
        print(f"  Baseline: {baseline.execution_time_ms:.2f}ms")
        print(f"  Current:  {metrics.execution_time_ms:.2f}ms")
        print(f"  Change:   {regression_pct:+.1f}%")

        assert (
            passed
        ), f"Performance regression detected: {regression_pct:.1f}% > {self.REGRESSION_THRESHOLD}%"

    @pytest.mark.skipif(not RING_AVAILABLE, reason="Ring implementations not available")
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", RING_CONFIGS)
    def test_ring_dilated_attention_performance(
        self, baseline_manager, device, batch_size, seq_len, num_heads, head_dim
    ):
        """Test RingDilatedAttention performance regression."""
        config = create_test_config(seq_len, batch_size, num_heads, head_dim)
        implementation = "RingDilatedAttention"

        # Create module with appropriate ring size
        segment_lengths = [seq_len // 4, seq_len // 2, seq_len]
        dilation_rates = [1, 2, 4]

        # Choose ring_size that evenly divides seq_len
        ring_size = 4
        while seq_len % (ring_size * max(segment_lengths)) != 0 and ring_size > 1:
            ring_size -= 1

        module = RingDilatedAttention(segment_lengths, dilation_rates, ring_size=ring_size).to(
            device
        )

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Measure performance
        metrics = measure_performance(module, (q, k, v))

        # Compare with baseline
        baseline = baseline_manager.get_baseline(implementation, config)
        if baseline is None:
            baseline_manager.update_baseline(implementation, config, metrics)
            pytest.skip(f"No baseline for {implementation} {config}. Setting current as baseline.")

        # Check for regression
        passed, regression_pct = self.check_regression(baseline, metrics)
        baseline_manager.add_history_entry(implementation, config, metrics, passed, regression_pct)

        print(f"\n{implementation} {config} (ring_size={ring_size}):")
        print(f"  Baseline: {baseline.execution_time_ms:.2f}ms")
        print(f"  Current:  {metrics.execution_time_ms:.2f}ms")
        print(f"  Change:   {regression_pct:+.1f}%")

        assert (
            passed
        ), f"Performance regression detected: {regression_pct:.1f}% > {self.REGRESSION_THRESHOLD}%"

    @pytest.mark.skipif(not RING_AVAILABLE, reason="Ring implementations not available")
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", RING_CONFIGS[:2])
    def test_ring_multihead_dilated_attention_performance(
        self, baseline_manager, device, batch_size, seq_len, num_heads, head_dim
    ):
        """Test RingMultiheadDilatedAttention performance regression."""
        config = create_test_config(seq_len, batch_size, num_heads, head_dim)
        implementation = "RingMultiheadDilatedAttention"

        # Create module
        embed_dim = num_heads * head_dim
        segment_lengths = [seq_len // 4, seq_len // 2, seq_len]
        dilation_rates = [1, 2, 4]

        ring_size = 4
        while seq_len % (ring_size * max(segment_lengths)) != 0 and ring_size > 1:
            ring_size -= 1

        module = RingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=ring_size,
        ).to(device)

        # Create inputs
        q = torch.randn(batch_size, seq_len, embed_dim, device=device)
        k = torch.randn(batch_size, seq_len, embed_dim, device=device)
        v = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Measure performance
        metrics = measure_performance(module, (q, k, v))

        # Compare with baseline
        baseline = baseline_manager.get_baseline(implementation, config)
        if baseline is None:
            baseline_manager.update_baseline(implementation, config, metrics)
            pytest.skip(f"No baseline for {implementation} {config}. Setting current as baseline.")

        # Check for regression
        passed, regression_pct = self.check_regression(baseline, metrics)
        baseline_manager.add_history_entry(implementation, config, metrics, passed, regression_pct)

        print(f"\n{implementation} {config} (ring_size={ring_size}):")
        print(f"  Baseline: {baseline.execution_time_ms:.2f}ms")
        print(f"  Current:  {metrics.execution_time_ms:.2f}ms")
        print(f"  Change:   {regression_pct:+.1f}%")

        assert (
            passed
        ), f"Performance regression detected: {regression_pct:.1f}% > {self.REGRESSION_THRESHOLD}%"

    @pytest.mark.skipif(
        not BLOCK_SPARSE_AVAILABLE, reason="Block sparse implementations not available"
    )
    @pytest.mark.parametrize(
        "batch_size,seq_len,num_heads,head_dim", CONFIGS[:2]
    )  # Test smaller configs
    def test_block_sparse_ring_dilated_attention_performance(
        self, baseline_manager, device, batch_size, seq_len, num_heads, head_dim
    ):
        """Test BlockSparseRingDilatedAttention performance regression."""
        config = create_test_config(seq_len, batch_size, num_heads, head_dim)
        implementation = "BlockSparseRingDilatedAttention"

        # Create module
        segment_lengths = [seq_len // 4, seq_len // 2, seq_len]
        dilation_rates = [1, 2, 4]
        sparsity_config = {
            "sparsity_ratio": 0.1,
            "block_size": 64,
            "pattern_type": "local_window",
            "local_window_size": 256,
        }

        module = BlockSparseRingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparsity_config=sparsity_config,
        ).to(device)

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Measure performance
        metrics = measure_performance(module, (q, k, v))

        # Compare with baseline
        baseline = baseline_manager.get_baseline(implementation, config)
        if baseline is None:
            baseline_manager.update_baseline(implementation, config, metrics)
            pytest.skip(f"No baseline for {implementation} {config}. Setting current as baseline.")

        # Check for regression
        passed, regression_pct = self.check_regression(baseline, metrics)
        baseline_manager.add_history_entry(implementation, config, metrics, passed, regression_pct)

        print(f"\n{implementation} {config} (sparsity=90%):")
        print(f"  Baseline: {baseline.execution_time_ms:.2f}ms")
        print(f"  Current:  {metrics.execution_time_ms:.2f}ms")
        print(f"  Change:   {regression_pct:+.1f}%")

        assert (
            passed
        ), f"Performance regression detected: {regression_pct:.1f}% > {self.REGRESSION_THRESHOLD}%"


@pytest.mark.benchmark
def test_update_all_baselines(baseline_manager):  # noqa: ARG001
    """Special test to update all baselines when needed."""
    # This test is skipped by default. Run with:
    # pytest tests/test_performance_regression_all.py::test_update_all_baselines -v
    pytest.skip("Run this test explicitly to update baselines")


def generate_performance_report(baseline_dir: str = "tests/performance_baselines"):
    """Generate a comprehensive performance report from history."""
    baseline_manager = PerformanceBaseline(baseline_dir)

    if not baseline_manager.history:
        print("No performance history available.")
        return

    # Group by implementation and config
    from collections import defaultdict

    history_by_impl = defaultdict(lambda: defaultdict(list))

    for entry in baseline_manager.history:
        impl = entry["implementation"]
        config = entry["config"]
        history_by_impl[impl][config].append(entry)

    # Generate report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE REGRESSION REPORT")
    print("=" * 80)

    # Group implementations by category
    categories = {
        "Standard": ["DilatedAttention", "MultiheadDilatedAttention"],
        "Improved": ["ImprovedDilatedAttention", "ImprovedMultiheadDilatedAttention"],
        "Ring": ["RingDilatedAttention", "RingMultiheadDilatedAttention"],
        "Block Sparse": [
            "BlockSparseRingDilatedAttention",
            "BlockSparseRingMultiheadDilatedAttention",
        ],
    }

    for category, implementations in categories.items():
        print(f"\n{category} Implementations:")
        print("=" * len(f"{category} Implementations:"))

        for impl in implementations:
            if impl not in history_by_impl:
                continue

            print(f"\n{impl}:")
            print("-" * len(impl))

            configs = history_by_impl[impl]
            for config, entries in sorted(configs.items()):
                recent_entries = entries[-5:]  # Last 5 runs

                print(f"\n  {config}:")
                for entry in recent_entries:
                    timestamp = entry["timestamp"][:19]  # Remove microseconds
                    metrics = entry["metrics"]
                    passed = "✓" if entry["passed"] else "✗"
                    regression = entry["regression_pct"]

                    print(
                        f"    {timestamp} {passed} {metrics['execution_time_ms']:.2f}ms ({regression:+.1f}%)"
                    )

                    # Show implementation details if available
                    details = metrics.get("implementation_details", {})
                    if details:
                        if "ring_size" in details:
                            print(f"      Ring size: {details['ring_size']}")
                        if "sparsity_config" in details:
                            print(f"      Sparsity: {details['sparsity_config']}")


if __name__ == "__main__":
    # Generate performance report when run directly
    generate_performance_report()
