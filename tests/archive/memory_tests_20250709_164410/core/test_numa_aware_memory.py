"""
Test suite for NUMA-aware memory management.
"""

from unittest.mock import Mock, patch
import pytest
import torch

from dilated_attention_pytorch.core.numa_aware_pool import (
    NUMANode,
    NUMATopologyDetector,
    NUMAAwareMemoryPool,
    get_numa_aware_pool,
    reset_numa_pool,
)


class TestNUMANode:
    """Test NUMA node data structure."""

    def test_numa_node_creation(self):
        """Test NUMA node creation and properties."""
        node = NUMANode(
            node_id=0,
            cpu_cores={0, 1, 2, 3},
            memory_size=8 * 1024**3,  # 8GB
            gpu_devices={0, 1},
            distance_map={0: 10, 1: 21},
        )

        assert node.node_id == 0
        assert node.num_cores == 4
        assert node.memory_gb == 8.0
        assert node.gpu_devices == {0, 1}
        assert node.distance_map[1] == 21

    def test_numa_node_empty(self):
        """Test NUMA node with no resources."""
        node = NUMANode(
            node_id=1,
            cpu_cores=set(),
            memory_size=0,
            gpu_devices=set(),
            distance_map={},
        )

        assert node.num_cores == 0
        assert node.memory_gb == 0.0
        assert len(node.gpu_devices) == 0


class TestNUMATopologyDetector:
    """Test NUMA topology detection."""

    @pytest.fixture
    def detector(self):
        return NUMATopologyDetector()

    def test_parse_cpu_list(self, detector):
        """Test CPU list parsing."""
        # Simple range
        cores = detector._parse_cpu_list("0-3")
        assert cores == {0, 1, 2, 3}

        # Multiple ranges
        cores = detector._parse_cpu_list("0-1,4-5")
        assert cores == {0, 1, 4, 5}

        # Mixed ranges and singles
        cores = detector._parse_cpu_list("0,2-4,7")
        assert cores == {0, 2, 3, 4, 7}

        # Single core
        cores = detector._parse_cpu_list("5")
        assert cores == {5}

    @patch("pathlib.Path.exists")
    def test_numa_not_available(self, mock_exists, detector):
        """Test behavior when NUMA is not available."""
        mock_exists.return_value = False

        result = detector.detect()
        assert result is False
        assert not detector.detected

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.is_dir")
    def test_numa_detection_no_nodes(
        self, mock_is_dir, mock_glob, mock_exists, detector
    ):
        """Test NUMA detection with no nodes."""
        mock_exists.return_value = True
        mock_glob.return_value = []

        result = detector.detect()
        assert result is True
        assert detector.detected
        assert len(detector.numa_nodes) == 0

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.read_text")
    def test_numa_detection_success(
        self, mock_read_text, mock_is_dir, mock_glob, mock_exists, detector
    ):
        """Test successful NUMA detection."""
        # Mock NUMA availability
        mock_exists.return_value = True

        # Mock node directories
        mock_node0 = Mock()
        mock_node0.name = "node0"
        mock_node0.is_dir.return_value = True
        mock_node1 = Mock()
        mock_node1.name = "node1"
        mock_node1.is_dir.return_value = True

        mock_glob.return_value = [mock_node0, mock_node1]
        mock_is_dir.return_value = True

        # Mock file reads
        def mock_read_side_effect(path_obj):
            file_path = str(path_obj)
            if "node0/cpulist" in file_path:
                return "0-3"
            elif "node1/cpulist" in file_path:
                return "4-7"
            elif "node0/meminfo" in file_path:
                return "Node 0 MemTotal: 8388608 kB"
            elif "node1/meminfo" in file_path:
                return "Node 1 MemTotal: 8388608 kB"
            elif "distance" in file_path:
                return "10 21"
            return ""

        mock_read_text.side_effect = mock_read_side_effect

        # Mock path operations
        with patch("pathlib.Path.__truediv__") as mock_div:
            mock_div.return_value.exists.return_value = True
            mock_div.return_value.read_text = mock_read_text

            result = detector.detect()

        assert result is True
        assert detector.detected
        assert len(detector.numa_nodes) == 2

        # Check node 0
        node0 = detector.numa_nodes[0]
        assert node0.node_id == 0
        assert node0.cpu_cores == {0, 1, 2, 3}
        assert node0.memory_size == 8388608 * 1024  # kB to bytes

    def test_get_topology_info_empty(self, detector):
        """Test topology info when no NUMA detected."""
        info = detector.get_topology_info()

        assert info["numa_available"] is False
        assert info["num_nodes"] == 0
        assert info["nodes"] == {}
        assert info["gpu_affinity"] == {}

    def test_get_topology_info_with_nodes(self, detector):
        """Test topology info with NUMA nodes."""
        # Manually add nodes for testing
        detector.numa_nodes[0] = NUMANode(
            node_id=0,
            cpu_cores={0, 1},
            memory_size=4 * 1024**3,
            gpu_devices={0},
            distance_map={0: 10, 1: 21},
        )
        detector.detected = True

        info = detector.get_topology_info()

        assert info["numa_available"] is True
        assert info["num_nodes"] == 1
        assert 0 in info["nodes"]
        assert info["nodes"][0]["cpu_cores"] == [0, 1]
        assert info["nodes"][0]["memory_gb"] == 4.0


class TestNUMAAwareMemoryPool:
    """Test NUMA-aware memory pool."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def pool_no_numa(self):
        """Pool with NUMA disabled."""
        return NUMAAwareMemoryPool(enable_numa=False)

    @pytest.fixture
    def pool_mock_numa(self):
        """Pool with mocked NUMA topology."""
        pool = NUMAAwareMemoryPool(enable_numa=True, auto_detect_topology=False)

        # Mock NUMA topology
        pool.numa_available = True
        pool.detector.detected = True
        pool.detector.numa_nodes[0] = NUMANode(
            node_id=0,
            cpu_cores={0, 1, 2, 3},
            memory_size=8 * 1024**3,
            gpu_devices={0},
            distance_map={0: 10, 1: 21},
        )
        pool.detector.numa_nodes[1] = NUMANode(
            node_id=1,
            cpu_cores={4, 5, 6, 7},
            memory_size=8 * 1024**3,
            gpu_devices={1},
            distance_map={0: 21, 1: 10},
        )
        pool.detector.gpu_numa_affinity[0] = 0
        pool.detector.gpu_numa_affinity[1] = 1

        return pool

    def test_pool_initialization_numa_disabled(self, pool_no_numa):
        """Test pool initialization with NUMA disabled."""
        assert pool_no_numa.enable_numa is False
        assert pool_no_numa.numa_available is False
        assert pool_no_numa.stats["total_allocations"] == 0

    def test_pool_initialization_numa_enabled(self, pool_mock_numa):
        """Test pool initialization with NUMA enabled."""
        assert pool_mock_numa.enable_numa is True
        assert pool_mock_numa.numa_available is True
        assert len(pool_mock_numa.detector.numa_nodes) == 2

    def test_basic_allocation_no_numa(self, pool_no_numa, device):
        """Test basic allocation without NUMA."""
        shape = (256, 64)
        tensor = pool_no_numa.allocate(shape, torch.float32, device)

        assert tensor is not None
        assert tensor.shape == shape
        assert tensor.dtype == torch.float32
        assert tensor.device == device

        stats = pool_no_numa.get_stats()
        assert stats["total_allocations"] == 1

    def test_allocation_with_numa(self, pool_mock_numa, device):
        """Test allocation with NUMA awareness."""
        shape = (128, 32)
        tensor = pool_mock_numa.allocate(shape, torch.float32, device)

        assert tensor is not None
        assert tensor.shape == shape
        assert tensor.dtype == torch.float32

        stats = pool_mock_numa.get_stats()
        assert stats["total_allocations"] == 1

    def test_numa_node_selection_gpu(self, pool_mock_numa):
        """Test NUMA node selection for GPU devices."""
        # Test GPU 0 -> Node 0
        device = torch.device("cuda:0")
        selected_node = pool_mock_numa._select_numa_node(device, None)
        assert selected_node == 0

        # Test GPU 1 -> Node 1
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
            selected_node = pool_mock_numa._select_numa_node(device, None)
            assert selected_node == 1

    def test_numa_node_selection_prefer(self, pool_mock_numa):
        """Test NUMA node selection with preference."""
        device = torch.device("cpu")

        # Test explicit preference
        selected_node = pool_mock_numa._select_numa_node(device, prefer_numa_node=1)
        assert selected_node == 1

        # Test invalid preference (should warn and fallback)
        selected_node = pool_mock_numa._select_numa_node(device, prefer_numa_node=99)
        assert selected_node is not None  # Should fallback

    @patch("os.sched_getaffinity")
    @patch("os.sched_setaffinity")
    def test_numa_affinity_context(
        self, mock_set_affinity, mock_get_affinity, pool_mock_numa
    ):
        """Test NUMA affinity context manager."""
        mock_get_affinity.return_value = {0, 1, 2, 3, 4, 5, 6, 7}

        with pool_mock_numa.numa_affinity(1):
            # Should set affinity to node 1 cores
            mock_set_affinity.assert_called_with(0, {4, 5, 6, 7})

        # Should restore original affinity
        assert mock_set_affinity.call_count == 2

    def test_pinned_memory_allocation(self, pool_mock_numa):
        """Test pinned memory allocation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        shape = (64, 32)
        tensor = pool_mock_numa.allocate(
            shape, torch.float32, torch.device("cpu"), pin_memory=True
        )

        assert tensor is not None
        assert tensor.shape == shape
        assert tensor.is_pinned()

    def test_allocation_fallback(self, pool_mock_numa):
        """Test fallback when NUMA allocation fails."""
        # Force an error in NUMA-aware allocation
        with patch.object(
            pool_mock_numa,
            "_allocate_numa_aware",
            side_effect=RuntimeError("Mock error"),
        ):
            shape = (32, 16)
            tensor = pool_mock_numa.allocate(shape, torch.float32)

            assert tensor is not None
            assert tensor.shape == shape

    def test_get_stats_comprehensive(self, pool_mock_numa, device):
        """Test comprehensive statistics."""
        # Perform some allocations
        for _ in range(5):
            pool_mock_numa.allocate((64, 32), torch.float32, device)

        stats = pool_mock_numa.get_stats()

        assert stats["total_allocations"] == 5
        assert "topology" in stats
        assert "timing_by_node" in stats
        assert stats["topology"]["numa_available"] is True

    def test_numa_report_generation(self, pool_mock_numa, device):
        """Test NUMA report generation."""
        # Perform allocation
        pool_mock_numa.allocate((32, 16), torch.float32, device)

        report = pool_mock_numa.get_numa_report()

        assert "NUMA-Aware Memory Pool Report" in report
        assert "NUMA Status: ✅ Available" in report
        assert "NUMA Topology:" in report

    def test_numa_report_no_numa(self, pool_no_numa):
        """Test NUMA report when NUMA is not available."""
        report = pool_no_numa.get_numa_report()

        assert "NUMA Status: ❌ Not available" in report

    @patch("dilated_attention_pytorch.core.numa_aware_pool._NUMA_POOL", None)
    def test_global_pool_singleton(self):
        """Test global pool singleton pattern."""
        pool1 = get_numa_aware_pool(enable_numa=False)
        pool2 = get_numa_aware_pool(enable_numa=True)  # Should be ignored

        assert pool1 is pool2
        assert pool1.enable_numa is False  # Uses first configuration

        reset_numa_pool()

        pool3 = get_numa_aware_pool(enable_numa=True)
        assert pool3 is not pool1
        assert pool3.enable_numa is True

    def test_allocation_performance_tracking(self, pool_mock_numa, device):
        """Test allocation performance tracking."""
        # Perform multiple allocations
        for i in range(10):
            pool_mock_numa.allocate((32 + i, 16), torch.float32, device)

        stats = pool_mock_numa.get_stats()

        assert stats["total_allocations"] == 10
        assert len(stats["timing_by_node"]) > 0

        # Check timing statistics
        for node_stats in stats["timing_by_node"].values():
            assert "count" in node_stats
            assert "avg_time" in node_stats
            assert node_stats["avg_time"] > 0

    def test_cross_numa_allocation_tracking(self, pool_mock_numa):
        """Test cross-NUMA allocation tracking."""
        # Mock current NUMA node detection
        with patch.object(pool_mock_numa, "_get_current_numa_node", return_value=0):
            # Allocate on same node (should be local)
            pool_mock_numa.allocate((32, 16), torch.float32, prefer_numa_node=0)

            # Allocate on different node (should be remote)
            pool_mock_numa.allocate((32, 16), torch.float32, prefer_numa_node=1)

        stats = pool_mock_numa.get_stats()
        assert stats["local_allocations"] >= 1
        assert stats["remote_allocations"] >= 1


class TestNUMAIntegration:
    """Integration tests for NUMA-aware features."""

    def test_real_numa_detection(self):
        """Test real NUMA detection (if available)."""
        detector = NUMATopologyDetector()
        result = detector.detect()

        # Should either succeed or fail gracefully
        assert isinstance(result, bool)

        if result:
            assert detector.detected
            assert len(detector.numa_nodes) > 0
        else:
            assert not detector.detected

    def test_numa_pool_real_allocation(self):
        """Test NUMA pool with real allocation."""
        pool = NUMAAwareMemoryPool(enable_numa=True)

        # Should work regardless of NUMA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = pool.allocate((64, 32), torch.float32, device)

        assert tensor is not None
        assert tensor.shape == (64, 32)

        # Get stats
        stats = pool.get_stats()
        assert stats["total_allocations"] == 1

        # Generate report
        report = pool.get_numa_report()
        assert "NUMA-Aware Memory Pool Report" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
