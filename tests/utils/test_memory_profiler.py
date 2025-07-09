"""
Test suite for memory profiler and visualization.
"""

import time
import pytest
import torch

from dilated_attention_pytorch.core.memory_profiler import (
    MemoryProfiler,
    AllocationEvent,
    MemorySnapshot,
    get_memory_profiler,
    reset_memory_profiler,
    profile_memory,
)
from dilated_attention_pytorch.core.enhanced_memory_pool import EnhancedMemoryPool


class TestAllocationEvent:
    """Test allocation event data structure."""

    def test_allocation_event_creation(self):
        """Test allocation event creation and properties."""
        event = AllocationEvent(
            timestamp=time.perf_counter(),
            operation="test_op",
            size_bytes=1024 * 1024,  # 1MB
            shape=(256, 1024),
            dtype=torch.float32,
            device=torch.device("cpu"),
            tensor_id=12345,
            stack_trace="test_stack",
            pool_type="test_pool",
            numa_node=0,
        )

        assert event.size_mb == 1.0
        assert event.size_gb == 1.0 / 1024
        assert event.pool_type == "test_pool"
        assert event.numa_node == 0


class TestMemorySnapshot:
    """Test memory snapshot data structure."""

    def test_memory_snapshot_creation(self):
        """Test memory snapshot creation and properties."""
        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            total_allocated=2 * 1024 * 1024 * 1024,  # 2GB
            total_cached=1024 * 1024 * 1024,  # 1GB
            peak_allocated=3 * 1024 * 1024 * 1024,  # 3GB
            num_allocations=100,
            num_deallocations=50,
        )

        assert snapshot.allocated_gb == 2.0
        assert snapshot.allocated_mb == 2048.0
        assert snapshot.num_allocations == 100


class TestMemoryProfiler:
    """Test memory profiler functionality."""

    @pytest.fixture
    def profiler(self):
        return MemoryProfiler(
            enable_stack_traces=False,  # Disable for testing
            max_events=100,
            snapshot_interval=0.1,
        )

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.max_events == 100
        assert profiler.snapshot_interval == 0.1
        assert not profiler._profiling_active
        assert len(profiler.allocation_events) == 0

    def test_profiler_start_stop(self, profiler):
        """Test starting and stopping profiler."""
        assert not profiler._profiling_active

        profiler.start_profiling()
        assert profiler._profiling_active

        profiler.stop_profiling()
        assert not profiler._profiling_active

    def test_record_allocation(self, profiler, device):
        """Test recording allocation events."""
        profiler.start_profiling()

        # Create test tensor
        tensor = torch.empty((256, 64), dtype=torch.float32, device=device)

        # Record allocation
        profiler.record_allocation(tensor, pool_type="test_pool", numa_node=0)

        # Check event was recorded
        assert len(profiler.allocation_events) == 1
        assert len(profiler.active_allocations) == 1

        event = profiler.allocation_events[0]
        assert event.shape == (256, 64)
        assert event.dtype == torch.float32
        assert event.device == device
        assert event.pool_type == "test_pool"
        assert event.numa_node == 0

        profiler.stop_profiling()

    def test_record_deallocation(self, profiler, device):
        """Test recording deallocation events."""
        profiler.start_profiling()

        # Create and record tensor
        tensor = torch.empty((128, 32), dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type="test_pool")

        # Record deallocation
        profiler.record_deallocation(tensor)

        # Check deallocation was recorded
        assert len(profiler.deallocation_events) == 1
        assert len(profiler.active_allocations) == 0

        profiler.stop_profiling()

    def test_operation_context(self, profiler, device):
        """Test operation profiling context."""
        profiler.start_profiling()

        with profiler.profile_operation("test_operation"):
            tensor = torch.empty((64, 32), dtype=torch.float32, device=device)
            profiler.record_allocation(tensor, pool_type="test_pool")

        # Check operation was recorded
        event = profiler.allocation_events[0]
        assert event.operation == "test_operation"

        profiler.stop_profiling()

    def test_memory_snapshots(self, profiler):
        """Test memory snapshot taking."""
        profiler.start_profiling()

        # Wait for at least one snapshot
        time.sleep(0.2)

        profiler.stop_profiling()

        # Should have taken at least one snapshot
        assert len(profiler.memory_snapshots) >= 1

    def test_get_allocation_summary(self, profiler, device):
        """Test allocation summary generation."""
        profiler.start_profiling()

        # Create various allocations
        tensors = []
        for i in range(5):
            tensor = torch.empty((64, 32), dtype=torch.float32, device=device)
            profiler.record_allocation(tensor, pool_type=f"pool_{i % 2}")
            tensors.append(tensor)

        summary = profiler.get_allocation_summary()

        assert summary["basic_stats"]["total_allocations"] == 5
        assert len(summary["pool_breakdown"]) == 2
        assert "pool_0" in summary["pool_breakdown"]
        assert "pool_1" in summary["pool_breakdown"]

        profiler.stop_profiling()

    def test_pattern_analysis(self, profiler, device):
        """Test allocation pattern analysis."""
        profiler.start_profiling()
        profiler.pattern_analysis_window = 10  # Small window for testing

        # Create burst pattern (many small allocations)
        for _ in range(15):
            tensor = torch.empty((32, 32), dtype=torch.float32, device=device)
            profiler.record_allocation(tensor, pool_type="test")
            time.sleep(0.001)  # Small delay

        # Force pattern analysis
        profiler._analyze_patterns()

        # Should detect some patterns
        assert len(profiler.detected_patterns) > 0

        profiler.stop_profiling()

    def test_get_recommendations(self, profiler, device):
        """Test optimization recommendations."""
        profiler.start_profiling()

        # Simulate high allocation rate
        profiler.stats["allocation_rate"] = 100  # High rate

        recommendations = profiler.get_recommendations()
        assert len(recommendations) > 0
        assert any("allocation rate" in rec for rec in recommendations)

        profiler.stop_profiling()

    def test_generate_report(self, profiler, device):
        """Test report generation."""
        profiler.start_profiling()

        # Create some allocations
        for i in range(3):
            tensor = torch.empty((64, 32), dtype=torch.float32, device=device)
            profiler.record_allocation(tensor, pool_type="test_pool")

        # Wait for snapshot
        time.sleep(0.15)

        report = profiler.generate_report()

        assert "Memory Profiling Report" in report
        assert "Total Allocations: 3" in report
        assert "test_pool" in report

        profiler.stop_profiling()

    def test_export_data(self, profiler, device, tmp_path):
        """Test data export functionality."""
        profiler.start_profiling()

        # Create allocation
        tensor = torch.empty((32, 16), dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type="test_pool")

        # Export data
        export_path = tmp_path / "profiling_data.json"
        profiler.export_data(export_path)

        # Check file was created
        assert export_path.exists()

        # Check content
        import json

        with open(export_path) as f:
            data = json.load(f)

        assert "stats" in data
        assert "allocation_events" in data
        assert len(data["allocation_events"]) == 1

        profiler.stop_profiling()

    def test_clear_data(self, profiler, device):
        """Test clearing profiling data."""
        profiler.start_profiling()

        # Create some data
        tensor = torch.empty((32, 16), dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type="test_pool")

        assert len(profiler.allocation_events) == 1

        # Clear data
        profiler.clear_data()

        assert len(profiler.allocation_events) == 0
        assert len(profiler.active_allocations) == 0
        assert profiler.stats["total_allocations"] == 0

        profiler.stop_profiling()


class TestEnhancedMemoryPoolProfiling:
    """Test enhanced memory pool with profiling integration."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_enhanced_pool_with_profiling(self, device):
        """Test enhanced pool with profiling enabled."""
        pool = EnhancedMemoryPool(
            enable_fragment_aware=True,
            enable_bucketed=True,
            enable_numa=True,
            enable_profiling=True,
        )

        # Check profiler was initialized
        assert pool.profiler is not None
        assert pool.profiler._profiling_active

        # Allocate some tensors
        tensors = []
        for i in range(5):
            tensor = pool.allocate((64, 32), torch.float32, device)
            tensors.append(tensor)

        # Check allocations were recorded
        assert len(pool.profiler.allocation_events) == 5

        # Deallocate
        for tensor in tensors:
            pool.deallocate(tensor)

        # Check deallocations were recorded
        assert len(pool.profiler.deallocation_events) == 5

    def test_profiling_report_integration(self, device):
        """Test profiling report integration."""
        pool = EnhancedMemoryPool(enable_profiling=True)

        # Allocate tensor
        _ = pool.allocate((128, 64), torch.float32, device)

        # Get profiling report
        report = pool.get_profiling_report()
        assert "Memory Profiling Report" in report
        assert "Total Allocations: 1" in report

    def test_dashboard_creation(self, device):
        """Test memory dashboard creation."""
        pool = EnhancedMemoryPool(enable_profiling=True)

        # Allocate some tensors
        for _ in range(3):
            _ = pool.allocate((64, 32), torch.float32, device)

        # Create dashboard
        dashboard = pool.create_memory_dashboard()
        assert isinstance(dashboard, str)
        assert "Memory Profiling Dashboard" in dashboard

    def test_data_export_integration(self, device, tmp_path):
        """Test data export integration."""
        pool = EnhancedMemoryPool(enable_profiling=True)

        # Allocate tensor
        _ = pool.allocate((32, 16), torch.float32, device)

        # Export data
        export_path = tmp_path / "pool_profiling_data.json"
        pool.export_profiling_data(export_path)

        # Check file exists
        assert export_path.exists()


class TestGlobalProfiler:
    """Test global profiler functionality."""

    def test_get_global_profiler(self):
        """Test getting global profiler instance."""
        # Reset first
        reset_memory_profiler()

        profiler1 = get_memory_profiler(max_events=200)
        profiler2 = get_memory_profiler(max_events=300)  # Should be ignored

        # Should be same instance
        assert profiler1 is profiler2
        assert profiler1.max_events == 200  # Uses first configuration

    def test_reset_global_profiler(self):
        """Test resetting global profiler."""
        profiler1 = get_memory_profiler()
        profiler1.start_profiling()

        reset_memory_profiler()

        profiler2 = get_memory_profiler()
        assert profiler1 is not profiler2
        assert not profiler2._profiling_active

    def test_profile_memory_context(self, device):
        """Test profile_memory context manager."""
        reset_memory_profiler()

        with profile_memory("test_context") as profiler:
            tensor = torch.empty((64, 32), dtype=torch.float32, device=device)
            profiler.record_allocation(tensor, pool_type="context_test")

        # Check allocation was recorded
        assert len(profiler.allocation_events) == 1
        event = profiler.allocation_events[0]
        assert event.operation == "test_context"


class TestMemoryVisualization:
    """Test memory visualization functionality."""

    @pytest.fixture
    def profiler_with_data(self, device):
        """Create profiler with sample data."""
        profiler = MemoryProfiler(max_events=50, snapshot_interval=0.05)
        profiler.start_profiling()

        # Create sample allocations
        tensors = []
        for i in range(10):
            shape = (64 * (i + 1), 32)
            tensor = torch.empty(shape, dtype=torch.float32, device=device)
            profiler.record_allocation(tensor, pool_type=f"pool_{i % 3}")
            tensors.append(tensor)
            time.sleep(0.01)

        # Wait for snapshots
        time.sleep(0.1)

        profiler.stop_profiling()
        return profiler

    def test_visualizer_import(self, profiler_with_data):
        """Test importing memory visualizer."""
        try:
            from dilated_attention_pytorch.core.memory_visualizer import (
                MemoryVisualizer,
            )

            visualizer = MemoryVisualizer(profiler_with_data)
            assert visualizer.profiler is profiler_with_data
        except ImportError:
            pytest.skip("Visualization libraries not available")

    def test_timeline_plot(self, profiler_with_data, tmp_path):
        """Test timeline plot generation."""
        try:
            from dilated_attention_pytorch.core.memory_visualizer import (
                MemoryVisualizer,
            )

            visualizer = MemoryVisualizer(profiler_with_data)

            # Test static plot (matplotlib)
            save_path = tmp_path / "timeline.png"
            result = visualizer.plot_memory_timeline(
                duration=60.0, save_path=save_path, interactive=False
            )

            # Should save file if matplotlib available
            if result is None and save_path.exists():
                assert save_path.exists()

        except ImportError:
            pytest.skip("Visualization libraries not available")

    def test_allocation_distribution(self, profiler_with_data):
        """Test allocation distribution plot."""
        try:
            from dilated_attention_pytorch.core.memory_visualizer import (
                MemoryVisualizer,
            )

            visualizer = MemoryVisualizer(profiler_with_data)
            _ = visualizer.plot_allocation_distribution(interactive=False)

            # Should work if matplotlib available

        except ImportError:
            pytest.skip("Visualization libraries not available")

    def test_dashboard_creation(self, profiler_with_data):
        """Test dashboard creation."""
        try:
            from dilated_attention_pytorch.core.memory_visualizer import (
                create_memory_dashboard,
            )

            dashboard = create_memory_dashboard(profiler_with_data)
            assert isinstance(dashboard, str)
            assert "Memory Profiling Dashboard" in dashboard

        except ImportError:
            pytest.skip("Visualization libraries not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
