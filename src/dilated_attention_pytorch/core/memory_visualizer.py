"""
Memory visualization and analysis tools for Phase 1.4.

This module provides visualization capabilities for memory profiling data,
including timeline plots, allocation heatmaps, and pattern analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "matplotlib not available - visualization features disabled. "
        "Install with: pip install matplotlib"
    )

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .memory_profiler import MemoryProfiler, MemorySnapshot

logger = logging.getLogger("dilated_attention_pytorch.memory_visualizer")


class MemoryVisualizer:
    """
    Memory visualization and analysis tools.

    Features:
    - Memory timeline plots
    - Allocation size distributions
    - Pool efficiency heatmaps
    - NUMA allocation patterns
    - Interactive visualizations with Plotly
    - Static plots with Matplotlib
    """

    def __init__(self, profiler: MemoryProfiler):
        """
        Initialize the memory visualizer.

        Args:
            profiler: Memory profiler instance to visualize
        """
        self.profiler = profiler

        if not HAS_MATPLOTLIB and not HAS_PLOTLY:
            raise ImportError(
                "No visualization libraries available. "
                "Install matplotlib and/or plotly: pip install matplotlib plotly"
            )

    def plot_memory_timeline(
        self,
        duration: float = 300.0,
        save_path: Optional[Path] = None,
        interactive: bool = True,
    ) -> Optional[str]:
        """
        Plot memory usage timeline.

        Args:
            duration: Duration in seconds to plot
            save_path: Path to save the plot
            interactive: Use Plotly for interactive plot

        Returns:
            HTML string if interactive, else None
        """
        timeline = self.profiler.get_memory_timeline(duration)

        if not timeline:
            logger.warning("No timeline data available")
            return None

        if interactive and HAS_PLOTLY:
            return self._plot_timeline_plotly(timeline, save_path)
        elif HAS_MATPLOTLIB:
            return self._plot_timeline_matplotlib(timeline, save_path)
        else:
            logger.error("No visualization library available")
            return None

    def _plot_timeline_plotly(
        self, timeline: List[MemorySnapshot], save_path: Optional[Path]
    ) -> str:
        """Create interactive timeline plot with Plotly."""
        # Prepare data
        timestamps = [snapshot.timestamp for snapshot in timeline]
        allocated_gb = [snapshot.allocated_gb for snapshot in timeline]
        cached_gb = [snapshot.total_cached / (1024**3) for snapshot in timeline]
        num_allocations = [snapshot.num_allocations for snapshot in timeline]

        # Convert timestamps to relative time
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Memory Usage Over Time", "Active Allocations"),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # Memory usage plot
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=allocated_gb,
                mode="lines+markers",
                name="Allocated Memory",
                line=dict(color="blue", width=2),
                hovertemplate="<b>Allocated</b><br>Time: %{x:.1f}s<br>Memory: %{y:.2f} GB<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=cached_gb,
                mode="lines+markers",
                name="Cached Memory",
                line=dict(color="orange", width=2),
                hovertemplate="<b>Cached</b><br>Time: %{x:.1f}s<br>Memory: %{y:.2f} GB<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Active allocations plot
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=num_allocations,
                mode="lines+markers",
                name="Active Allocations",
                line=dict(color="green", width=2),
                hovertemplate="<b>Active Allocations</b><br>Time: %{x:.1f}s<br>Count: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=dict(text="Memory Profiling Timeline", x=0.5, font=dict(size=16)),
            height=600,
            showlegend=True,
            hovermode="x unified",
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        # Update y-axis labels
        fig.update_yaxes(title_text="Memory (GB)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        # Save if requested
        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Timeline plot saved to {save_path}")

        return fig.to_html()

    def _plot_timeline_matplotlib(
        self, timeline: List[MemorySnapshot], save_path: Optional[Path]
    ) -> None:
        """Create static timeline plot with Matplotlib."""
        # Prepare data
        timestamps = [snapshot.timestamp for snapshot in timeline]
        allocated_gb = [snapshot.allocated_gb for snapshot in timeline]
        cached_gb = [snapshot.total_cached / (1024**3) for snapshot in timeline]
        num_allocations = [snapshot.num_allocations for snapshot in timeline]

        # Convert to relative time
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Memory Profiling Timeline", fontsize=16)

        # Memory usage plot
        ax1.plot(
            relative_times,
            allocated_gb,
            "b-o",
            label="Allocated Memory",
            linewidth=2,
            markersize=4,
        )
        ax1.plot(
            relative_times,
            cached_gb,
            "orange",
            label="Cached Memory",
            linewidth=2,
            markersize=4,
        )
        ax1.set_ylabel("Memory (GB)")
        ax1.set_title("Memory Usage Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Active allocations plot
        ax2.plot(
            relative_times,
            num_allocations,
            "g-o",
            label="Active Allocations",
            linewidth=2,
            markersize=4,
        )
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Count")
        ax2.set_title("Active Allocations")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Timeline plot saved to {save_path}")

        plt.show()

    def plot_allocation_distribution(
        self,
        save_path: Optional[Path] = None,
        interactive: bool = True,
    ) -> Optional[str]:
        """
        Plot allocation size distribution.

        Args:
            save_path: Path to save the plot
            interactive: Use Plotly for interactive plot

        Returns:
            HTML string if interactive, else None
        """
        if not self.profiler.allocation_events:
            logger.warning("No allocation events available")
            return None

        # Extract allocation sizes
        sizes_mb = [event.size_mb for event in self.profiler.allocation_events]

        if interactive and HAS_PLOTLY:
            return self._plot_distribution_plotly(sizes_mb, save_path)
        elif HAS_MATPLOTLIB:
            return self._plot_distribution_matplotlib(sizes_mb, save_path)
        else:
            logger.error("No visualization library available")
            return None

    def _plot_distribution_plotly(
        self, sizes_mb: List[float], save_path: Optional[Path]
    ) -> str:
        """Create interactive distribution plot with Plotly."""
        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=sizes_mb,
                nbinsx=50,
                name="Allocation Size Distribution",
                hovertemplate="<b>Size Range</b><br>%{x} MB<br><b>Count</b>: %{y}<extra></extra>",
            )
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=sizes_mb,
                name="Size Distribution",
                visible=False,
                hovertemplate="<b>Allocation Size</b><br>%{y:.2f} MB<extra></extra>",
            )
        )

        # Update layout with toggle buttons
        fig.update_layout(
            title="Allocation Size Distribution",
            xaxis_title="Size (MB)",
            yaxis_title="Count",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"visible": [True, False]}],
                                label="Histogram",
                                method="restyle",
                            ),
                            dict(
                                args=[{"visible": [False, True]}],
                                label="Box Plot",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top",
                ),
            ],
        )

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Distribution plot saved to {save_path}")

        return fig.to_html()

    def _plot_distribution_matplotlib(
        self, sizes_mb: List[float], save_path: Optional[Path]
    ) -> None:
        """Create static distribution plot with Matplotlib."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Allocation Size Distribution", fontsize=16)

        # Histogram
        ax1.hist(sizes_mb, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax1.set_xlabel("Size (MB)")
        ax1.set_ylabel("Count")
        ax1.set_title("Histogram")
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(sizes_mb, vert=True)
        ax2.set_ylabel("Size (MB)")
        ax2.set_title("Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Distribution plot saved to {save_path}")

        plt.show()

    def plot_pool_efficiency(
        self,
        save_path: Optional[Path] = None,
        interactive: bool = True,
    ) -> Optional[str]:
        """
        Plot memory pool efficiency breakdown.

        Args:
            save_path: Path to save the plot
            interactive: Use Plotly for interactive plot

        Returns:
            HTML string if interactive, else None
        """
        summary = self.profiler.get_allocation_summary()
        pool_breakdown = summary.get("pool_breakdown", {})

        if not pool_breakdown:
            logger.warning("No pool breakdown data available")
            return None

        if interactive and HAS_PLOTLY:
            return self._plot_pool_efficiency_plotly(pool_breakdown, save_path)
        elif HAS_MATPLOTLIB:
            return self._plot_pool_efficiency_matplotlib(pool_breakdown, save_path)
        else:
            logger.error("No visualization library available")
            return None

    def _plot_pool_efficiency_plotly(
        self, pool_breakdown: Dict[str, Any], save_path: Optional[Path]
    ) -> str:
        """Create interactive pool efficiency plot with Plotly."""
        # Prepare data
        pool_names = list(pool_breakdown.keys())
        allocation_counts = [stats["count"] for stats in pool_breakdown.values()]
        memory_sizes_gb = [
            stats["total_size"] / (1024**3) for stats in pool_breakdown.values()
        ]

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Allocation Count by Pool", "Memory Usage by Pool"),
            specs=[[{"type": "pie"}, {"type": "pie"}]],
        )

        # Allocation count pie chart
        fig.add_trace(
            go.Pie(
                labels=pool_names,
                values=allocation_counts,
                name="Allocation Count",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Memory usage pie chart
        fig.add_trace(
            go.Pie(
                labels=pool_names,
                values=memory_sizes_gb,
                name="Memory Usage",
                hovertemplate="<b>%{label}</b><br>Memory: %{value:.2f} GB<br>Percentage: %{percent}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(title="Memory Pool Efficiency", showlegend=True)

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Pool efficiency plot saved to {save_path}")

        return fig.to_html()

    def _plot_pool_efficiency_matplotlib(
        self, pool_breakdown: Dict[str, Any], save_path: Optional[Path]
    ) -> None:
        """Create static pool efficiency plot with Matplotlib."""
        # Prepare data
        pool_names = list(pool_breakdown.keys())
        allocation_counts = [stats["count"] for stats in pool_breakdown.values()]
        memory_sizes_gb = [
            stats["total_size"] / (1024**3) for stats in pool_breakdown.values()
        ]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Memory Pool Efficiency", fontsize=16)

        # Allocation count pie chart
        ax1.pie(allocation_counts, labels=pool_names, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Allocation Count by Pool")

        # Memory usage pie chart
        ax2.pie(memory_sizes_gb, labels=pool_names, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Memory Usage by Pool")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Pool efficiency plot saved to {save_path}")

        plt.show()

    def create_dashboard(
        self,
        save_path: Optional[Path] = None,
        duration: float = 300.0,
    ) -> str:
        """
        Create comprehensive memory profiling dashboard.

        Args:
            save_path: Path to save the dashboard HTML
            duration: Duration for timeline plots

        Returns:
            HTML string of the dashboard
        """
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly required for dashboard. Install with: pip install plotly"
            )

        # Get profiling summary
        summary = self.profiler.get_allocation_summary()
        recommendations = self.profiler.get_recommendations()

        # Create dashboard HTML
        html_parts = [
            "<html><head><title>Memory Profiling Dashboard</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".section { margin-bottom: 30px; }",
            ".stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }",
            ".stat-card { background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }",
            ".stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }",
            ".stat-label { font-size: 14px; color: #7f8c8d; }",
            ".recommendations { background: #e8f5e8; padding: 15px; border-radius: 8px; }",
            ".recommendation { margin: 5px 0; }",
            "</style></head><body>",
            "<h1>Memory Profiling Dashboard</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # Basic statistics section
        basic_stats = summary["basic_stats"]
        html_parts.extend(
            [
                "<div class='section'>",
                "<h2>Basic Statistics</h2>",
                "<div class='stats-grid'>",
                f"<div class='stat-card'><div class='stat-value'>{basic_stats['total_allocations']:,}</div><div class='stat-label'>Total Allocations</div></div>",
                f"<div class='stat-card'><div class='stat-value'>{basic_stats['peak_memory'] / (1024**3):.2f} GB</div><div class='stat-label'>Peak Memory</div></div>",
                f"<div class='stat-card'><div class='stat-value'>{summary['active_allocations']:,}</div><div class='stat-label'>Active Allocations</div></div>",
                f"<div class='stat-card'><div class='stat-value'>{basic_stats['allocation_rate']:.1f}/sec</div><div class='stat-label'>Allocation Rate</div></div>",
                f"<div class='stat-card'><div class='stat-value'>{basic_stats['average_allocation_size'] / (1024**2):.1f} MB</div><div class='stat-label'>Avg Allocation Size</div></div>",
                f"<div class='stat-card'><div class='stat-value'>{basic_stats['profiling_overhead']:.3f}s</div><div class='stat-label'>Profiling Overhead</div></div>",
                "</div></div>",
            ]
        )

        # Timeline plot
        timeline_html = self.plot_memory_timeline(duration=duration, interactive=True)
        if timeline_html:
            html_parts.extend(
                [
                    "<div class='section'>",
                    "<h2>Memory Timeline</h2>",
                    timeline_html,
                    "</div>",
                ]
            )

        # Allocation distribution
        dist_html = self.plot_allocation_distribution(interactive=True)
        if dist_html:
            html_parts.extend(
                [
                    "<div class='section'>",
                    "<h2>Allocation Size Distribution</h2>",
                    dist_html,
                    "</div>",
                ]
            )

        # Pool efficiency
        pool_html = self.plot_pool_efficiency(interactive=True)
        if pool_html:
            html_parts.extend(
                [
                    "<div class='section'>",
                    "<h2>Pool Efficiency</h2>",
                    pool_html,
                    "</div>",
                ]
            )

        # Recommendations
        if recommendations:
            html_parts.extend(
                [
                    "<div class='section'>",
                    "<h2>Optimization Recommendations</h2>",
                    "<div class='recommendations'>",
                ]
            )

            for rec in recommendations[:10]:  # Top 10
                html_parts.append(f"<div class='recommendation'>• {rec}</div>")

            html_parts.extend(["</div>", "</div>"])

        html_parts.append("</body></html>")

        dashboard_html = "\n".join(html_parts)

        if save_path:
            with open(save_path, "w") as f:
                f.write(dashboard_html)
            logger.info(f"Dashboard saved to {save_path}")

        return dashboard_html

    def export_plots(
        self,
        output_dir: Path,
        duration: float = 300.0,
        format: str = "html",
    ) -> None:
        """
        Export all plots to files.

        Args:
            output_dir: Directory to save plots
            duration: Duration for timeline plots
            format: Output format ("html", "png", "both")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Timeline plot
        if format in ["html", "both"]:
            timeline_path = output_dir / f"memory_timeline_{timestamp}.html"
            self.plot_memory_timeline(
                duration=duration, save_path=timeline_path, interactive=True
            )

        if format in ["png", "both"] and HAS_MATPLOTLIB:
            timeline_path = output_dir / f"memory_timeline_{timestamp}.png"
            self.plot_memory_timeline(
                duration=duration, save_path=timeline_path, interactive=False
            )

        # Distribution plot
        if format in ["html", "both"]:
            dist_path = output_dir / f"allocation_distribution_{timestamp}.html"
            self.plot_allocation_distribution(save_path=dist_path, interactive=True)

        if format in ["png", "both"] and HAS_MATPLOTLIB:
            dist_path = output_dir / f"allocation_distribution_{timestamp}.png"
            self.plot_allocation_distribution(save_path=dist_path, interactive=False)

        # Pool efficiency plot
        if format in ["html", "both"]:
            pool_path = output_dir / f"pool_efficiency_{timestamp}.html"
            self.plot_pool_efficiency(save_path=pool_path, interactive=True)

        if format in ["png", "both"] and HAS_MATPLOTLIB:
            pool_path = output_dir / f"pool_efficiency_{timestamp}.png"
            self.plot_pool_efficiency(save_path=pool_path, interactive=False)

        # Dashboard
        if format in ["html", "both"]:
            dashboard_path = output_dir / f"memory_dashboard_{timestamp}.html"
            self.create_dashboard(save_path=dashboard_path, duration=duration)

        logger.info(f"All plots exported to {output_dir}")


def create_memory_dashboard(
    profiler: MemoryProfiler,
    save_path: Optional[Path] = None,
    duration: float = 300.0,
) -> str:
    """
    Create and return memory profiling dashboard.

    Args:
        profiler: Memory profiler instance
        save_path: Path to save dashboard
        duration: Duration for timeline

    Returns:
        HTML string of dashboard
    """
    visualizer = MemoryVisualizer(profiler)
    return visualizer.create_dashboard(save_path=save_path, duration=duration)
