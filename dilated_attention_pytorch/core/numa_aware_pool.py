"""
NUMA-aware memory pool implementation for Phase 1.4.

This module provides NUMA topology detection and affinity-based memory allocation
for optimal performance on multi-socket systems and distributed training setups.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Any

import torch
from torch import Tensor

logger = logging.getLogger("dilated_attention_pytorch.numa_aware_pool")


@dataclass
class NUMANode:
    """Represents a NUMA node in the system."""

    node_id: int
    cpu_cores: Set[int]
    memory_size: int  # Memory in bytes
    gpu_devices: Set[int]  # GPU device IDs on this node
    distance_map: Dict[int, int]  # Distance to other NUMA nodes

    @property
    def num_cores(self) -> int:
        """Number of CPU cores on this node."""
        return len(self.cpu_cores)

    @property
    def memory_gb(self) -> float:
        """Memory size in GB."""
        return self.memory_size / (1024**3)


class NUMATopologyDetector:
    """
    Detects NUMA topology on Linux systems.

    Features:
    - Parse /sys/devices/system/node/ for NUMA information
    - Detect CPU core assignments
    - Identify GPU-NUMA affinity
    - Calculate inter-node distances
    """

    def __init__(self):
        """Initialize the NUMA topology detector."""
        self.numa_nodes: Dict[int, NUMANode] = {}
        self.gpu_numa_affinity: Dict[int, int] = {}
        self.detected = False

    def detect(self) -> bool:
        """
        Detect NUMA topology.

        Returns:
            True if NUMA topology was successfully detected
        """
        if self.detected:
            return True

        try:
            if not self._is_numa_available():
                logger.info("NUMA not available on this system")
                return False

            self._detect_numa_nodes()
            self._detect_gpu_affinity()
            self._calculate_distances()

            self.detected = True
            logger.info(f"Detected {len(self.numa_nodes)} NUMA nodes")

            return True

        except Exception as e:
            logger.warning(f"Failed to detect NUMA topology: {e}")
            return False

    def _is_numa_available(self) -> bool:
        """Check if NUMA is available on this system."""
        return Path("/sys/devices/system/node").exists()

    def _detect_numa_nodes(self) -> None:
        """Detect NUMA nodes and their properties."""
        node_dir = Path("/sys/devices/system/node")

        for node_path in node_dir.glob("node*"):
            if not node_path.is_dir():
                continue

            try:
                node_id = int(node_path.name[4:])  # Remove "node" prefix

                # Read CPU cores
                cpu_cores = self._read_cpu_cores(node_path)

                # Read memory size
                memory_size = self._read_memory_size(node_path)

                # Create NUMA node
                numa_node = NUMANode(
                    node_id=node_id,
                    cpu_cores=cpu_cores,
                    memory_size=memory_size,
                    gpu_devices=set(),
                    distance_map={},
                )

                self.numa_nodes[node_id] = numa_node

            except (ValueError, FileNotFoundError) as e:
                logger.debug(f"Failed to read NUMA node {node_path}: {e}")
                continue

    def _read_cpu_cores(self, node_path: Path) -> Set[int]:
        """Read CPU cores assigned to a NUMA node."""
        try:
            cpulist_file = node_path / "cpulist"
            if not cpulist_file.exists():
                return set()

            cpulist = cpulist_file.read_text().strip()
            return self._parse_cpu_list(cpulist)

        except Exception as e:
            logger.debug(f"Failed to read CPU cores for {node_path}: {e}")
            return set()

    def _parse_cpu_list(self, cpulist: str) -> Set[int]:
        """Parse CPU list string (e.g., '0-3,8-11')."""
        cores = set()

        for part in cpulist.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cores.update(range(start, end + 1))
            else:
                cores.add(int(part))

        return cores

    def _read_memory_size(self, node_path: Path) -> int:
        """Read memory size for a NUMA node."""
        try:
            meminfo_file = node_path / "meminfo"
            if not meminfo_file.exists():
                return 0

            meminfo = meminfo_file.read_text()

            # Parse "Node X MemTotal: Y kB"
            for line in meminfo.split("\n"):
                if "MemTotal:" in line:
                    parts = line.split()
                    if len(parts) >= 3 and parts[-1] == "kB":
                        return int(parts[-2]) * 1024  # Convert kB to bytes

        except Exception as e:
            logger.debug(f"Failed to read memory size for {node_path}: {e}")

        return 0

    def _detect_gpu_affinity(self) -> None:
        """Detect which NUMA node each GPU is closest to."""
        if not torch.cuda.is_available():
            return

        num_gpus = torch.cuda.device_count()

        for gpu_id in range(num_gpus):
            try:
                # Try to read GPU NUMA node from sysfs
                numa_node = self._read_gpu_numa_node(gpu_id)

                if numa_node is not None and numa_node in self.numa_nodes:
                    self.numa_nodes[numa_node].gpu_devices.add(gpu_id)
                    self.gpu_numa_affinity[gpu_id] = numa_node
                else:
                    # Fallback: assign to node 0
                    if 0 in self.numa_nodes:
                        self.numa_nodes[0].gpu_devices.add(gpu_id)
                        self.gpu_numa_affinity[gpu_id] = 0

            except Exception as e:
                logger.debug(f"Failed to detect NUMA affinity for GPU {gpu_id}: {e}")

    def _read_gpu_numa_node(self, gpu_id: int) -> Optional[int]:
        """Read NUMA node for a specific GPU."""
        try:
            # Try different paths where GPU NUMA info might be
            possible_paths = [
                f"/sys/class/drm/card{gpu_id}/device/numa_node",
                "/sys/bus/pci/devices/*/numa_node",  # Would need PCI device lookup
            ]

            for path_pattern in possible_paths:
                if "*" not in path_pattern:
                    numa_file = Path(path_pattern)
                    if numa_file.exists():
                        numa_node = int(numa_file.read_text().strip())
                        if numa_node >= 0:  # -1 means no NUMA info
                            return numa_node

        except Exception as e:
            logger.debug(f"Failed to read GPU {gpu_id} NUMA node: {e}")

        return None

    def _calculate_distances(self) -> None:
        """Calculate distances between NUMA nodes."""
        try:
            distance_file = Path("/sys/devices/system/node/node0/distance")
            if not distance_file.exists():
                return

            # Read distance matrix
            distances = distance_file.read_text().strip().split()
            num_nodes = len(distances)

            # Parse distance matrix for each node
            for node_id in self.numa_nodes:
                if node_id < num_nodes:
                    node_distance_file = Path(
                        f"/sys/devices/system/node/node{node_id}/distance"
                    )
                    if node_distance_file.exists():
                        node_distances = list(
                            map(int, node_distance_file.read_text().strip().split())
                        )

                        for other_node_id, distance in enumerate(node_distances):
                            self.numa_nodes[node_id].distance_map[other_node_id] = (
                                distance
                            )

        except Exception as e:
            logger.debug(f"Failed to calculate NUMA distances: {e}")

    def get_topology_info(self) -> Dict[str, Any]:
        """Get comprehensive topology information."""
        return {
            "numa_available": self.detected,
            "num_nodes": len(self.numa_nodes),
            "nodes": {
                node_id: {
                    "cpu_cores": sorted(node.cpu_cores),
                    "num_cores": node.num_cores,
                    "memory_gb": node.memory_gb,
                    "gpu_devices": sorted(node.gpu_devices),
                    "distances": node.distance_map,
                }
                for node_id, node in self.numa_nodes.items()
            },
            "gpu_affinity": self.gpu_numa_affinity,
        }


class NUMAAwareMemoryPool:
    """
    NUMA-aware memory pool for optimal allocation performance.

    Features:
    - Automatic NUMA topology detection
    - NUMA-affinity memory allocation
    - Cross-NUMA transfer optimization
    - GPU-NUMA affinity management
    - Performance monitoring
    """

    def __init__(
        self,
        enable_numa: bool = True,
        prefer_local_allocation: bool = True,
        auto_detect_topology: bool = True,
    ):
        """
        Initialize NUMA-aware memory pool.

        Args:
            enable_numa: Enable NUMA awareness
            prefer_local_allocation: Prefer allocation on local NUMA node
            auto_detect_topology: Automatically detect NUMA topology
        """
        self.enable_numa = enable_numa
        self.prefer_local_allocation = prefer_local_allocation

        # NUMA topology
        self.detector = NUMATopologyDetector()
        self.numa_available = False

        if enable_numa and auto_detect_topology:
            self.numa_available = self.detector.detect()

        # Allocation statistics
        self.stats = {
            "total_allocations": 0,
            "local_allocations": 0,
            "remote_allocations": 0,
            "cross_numa_transfers": 0,
            "allocation_time_by_node": {},
        }

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"NUMA-aware memory pool initialized: "
            f"numa_enabled={enable_numa}, numa_available={self.numa_available}"
        )

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        prefer_numa_node: Optional[int] = None,
        pin_memory: bool = False,
    ) -> Tensor:
        """
        Allocate memory with NUMA awareness.

        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            prefer_numa_node: Preferred NUMA node (auto-detected if None)
            pin_memory: Whether to pin memory for GPU transfers

        Returns:
            Allocated tensor
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with self._lock:
            self.stats["total_allocations"] += 1

            # Determine target NUMA node
            target_numa_node = self._select_numa_node(device, prefer_numa_node)

            # Allocate with NUMA affinity
            start_time = time.perf_counter()

            try:
                if self.numa_available and target_numa_node is not None:
                    tensor = self._allocate_numa_aware(
                        shape, dtype, device, target_numa_node, pin_memory
                    )

                    # Track local vs remote allocation
                    current_node = self._get_current_numa_node()
                    if current_node == target_numa_node:
                        self.stats["local_allocations"] += 1
                    else:
                        self.stats["remote_allocations"] += 1
                else:
                    # Fallback to standard allocation
                    tensor = self._allocate_standard(shape, dtype, device, pin_memory)

                # Record timing
                alloc_time = time.perf_counter() - start_time
                node_key = (
                    str(target_numa_node)
                    if target_numa_node is not None
                    else "fallback"
                )

                if node_key not in self.stats["allocation_time_by_node"]:
                    self.stats["allocation_time_by_node"][node_key] = []
                self.stats["allocation_time_by_node"][node_key].append(alloc_time)

                return tensor

            except Exception as e:
                logger.warning(
                    f"NUMA-aware allocation failed, falling back to standard: {e}"
                )
                return self._allocate_standard(shape, dtype, device, pin_memory)

    def _select_numa_node(
        self, device: torch.device, prefer_numa_node: Optional[int]
    ) -> Optional[int]:
        """Select optimal NUMA node for allocation."""
        if not self.numa_available:
            return None

        # Use explicitly preferred node
        if prefer_numa_node is not None:
            if prefer_numa_node in self.detector.numa_nodes:
                return prefer_numa_node
            else:
                logger.warning(f"Preferred NUMA node {prefer_numa_node} not found")

        # For GPU devices, use GPU-NUMA affinity
        if device.type == "cuda" and device.index is not None:
            gpu_id = device.index
            if gpu_id in self.detector.gpu_numa_affinity:
                return self.detector.gpu_numa_affinity[gpu_id]

        # For CPU or unknown GPU, use current thread's NUMA node
        current_node = self._get_current_numa_node()
        if current_node is not None:
            return current_node

        # Fallback to first available node
        if self.detector.numa_nodes:
            return min(self.detector.numa_nodes.keys())

        return None

    def _get_current_numa_node(self) -> Optional[int]:
        """Get current thread's NUMA node."""
        try:
            # Try to read current CPU
            with open("/proc/self/stat", "r") as f:
                stat_line = f.read().split()
                if len(stat_line) > 38:
                    cpu_id = int(stat_line[38])  # Last CPU used

                    # Find which NUMA node this CPU belongs to
                    for node_id, node in self.detector.numa_nodes.items():
                        if cpu_id in node.cpu_cores:
                            return node_id

        except Exception as e:
            logger.debug(f"Failed to get current NUMA node: {e}")

        return None

    @contextmanager
    def numa_affinity(self, numa_node: int):
        """Context manager for NUMA affinity."""
        if not self.numa_available or numa_node not in self.detector.numa_nodes:
            yield
            return

        old_affinity = None

        try:
            # Save current CPU affinity
            old_affinity = os.sched_getaffinity(0)

            # Set CPU affinity to NUMA node
            node = self.detector.numa_nodes[numa_node]
            os.sched_setaffinity(0, node.cpu_cores)

            yield

        except Exception as e:
            logger.debug(f"Failed to set NUMA affinity: {e}")
            yield

        finally:
            # Restore original affinity
            if old_affinity is not None:
                try:
                    os.sched_setaffinity(0, old_affinity)
                except Exception as e:
                    logger.debug(f"Failed to restore CPU affinity: {e}")

    def _allocate_numa_aware(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        numa_node: int,
        pin_memory: bool,
    ) -> Tensor:
        """Allocate memory with NUMA affinity."""
        with self.numa_affinity(numa_node):
            if device.type == "cuda":
                # For CUDA tensors, allocate directly on device
                return torch.empty(shape, dtype=dtype, device=device)
            else:
                # For CPU tensors, use pinned memory if requested
                if pin_memory and torch.cuda.is_available():
                    return torch.empty(shape, dtype=dtype, pin_memory=True)
                else:
                    return torch.empty(shape, dtype=dtype, device=device)

    def _allocate_standard(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> Tensor:
        """Standard allocation without NUMA awareness."""
        if pin_memory and device.type == "cpu" and torch.cuda.is_available():
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def get_stats(self) -> Dict[str, Any]:
        """Get NUMA allocation statistics."""
        with self._lock:
            stats = self.stats.copy()

            # Calculate timing statistics
            timing_stats = {}
            for node, times in self.stats["allocation_time_by_node"].items():
                if times:
                    timing_stats[node] = {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                    }

            stats["timing_by_node"] = timing_stats
            stats["topology"] = self.detector.get_topology_info()

            return stats

    def get_numa_report(self) -> str:
        """Generate NUMA topology and performance report."""
        lines = ["NUMA-Aware Memory Pool Report", "=" * 35, ""]

        stats = self.get_stats()
        topology = stats["topology"]

        # NUMA availability
        if topology["numa_available"]:
            lines.extend(
                [
                    f"NUMA Status: ✅ Available ({topology['num_nodes']} nodes)",
                    f"Total Allocations: {stats['total_allocations']:,}",
                    f"Local Allocations: {stats['local_allocations']:,} ({stats['local_allocations'] / max(1, stats['total_allocations']) * 100:.1f}%)",
                    f"Remote Allocations: {stats['remote_allocations']:,} ({stats['remote_allocations'] / max(1, stats['total_allocations']) * 100:.1f}%)",
                    "",
                ]
            )

            # NUMA topology
            lines.append("NUMA Topology:")
            lines.append("-" * 15)

            for node_id, node_info in topology["nodes"].items():
                gpus = node_info["gpu_devices"]
                gpu_str = f", GPUs: {gpus}" if gpus else ""
                lines.append(
                    f"Node {node_id}: {node_info['num_cores']} cores, "
                    f"{node_info['memory_gb']:.1f}GB{gpu_str}"
                )

            lines.append("")

            # Performance by node
            if stats["timing_by_node"]:
                lines.append("Performance by Node:")
                lines.append("-" * 20)

                for node, timing in stats["timing_by_node"].items():
                    lines.append(
                        f"Node {node}: {timing['count']} allocs, "
                        f"avg {timing['avg_time'] * 1000:.3f}ms"
                    )
        else:
            lines.append("NUMA Status: ❌ Not available")

        return "\n".join(lines)


# Global NUMA-aware pool instance
_NUMA_POOL: Optional[NUMAAwareMemoryPool] = None
_NUMA_LOCK = threading.Lock()


def get_numa_aware_pool(**kwargs) -> NUMAAwareMemoryPool:
    """
    Get the global NUMA-aware memory pool instance.

    Args:
        **kwargs: Configuration for the pool (only used on first call)

    Returns:
        Global NUMA-aware memory pool instance
    """
    global _NUMA_POOL

    if _NUMA_POOL is None:
        with _NUMA_LOCK:
            if _NUMA_POOL is None:
                _NUMA_POOL = NUMAAwareMemoryPool(**kwargs)

    return _NUMA_POOL


def reset_numa_pool() -> None:
    """Reset the global NUMA-aware memory pool."""
    global _NUMA_POOL

    with _NUMA_LOCK:
        _NUMA_POOL = None
