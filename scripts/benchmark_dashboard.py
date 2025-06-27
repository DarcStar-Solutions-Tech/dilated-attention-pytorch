#!/usr/bin/env python3
"""
Create a dashboard showing benchmark performance across commits.

This script:
1. Scans all benchmark results
2. Groups them by git commit
3. Shows performance trends
4. Identifies regressions/improvements
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
docs_benchmarks_dir = Path(__file__).parent.parent / "docs" / "benchmarks"
sys.path.insert(0, str(benchmarks_dir))

from core import BenchmarkAggregator


def find_all_benchmark_results():
    """Find all benchmark result files."""
    results = []
    
    # Check new format location (docs/benchmarks/by-type/)
    by_type_dir = docs_benchmarks_dir / "by-type"
    if by_type_dir.exists():
        for json_file in by_type_dir.rglob("*.json"):
            if not json_file.is_symlink():  # Skip symlinks
                results.append(json_file)
    
    # Also check by-date location
    by_date_dir = docs_benchmarks_dir / "by-date"
    if by_date_dir.exists():
        for json_file in by_date_dir.rglob("*.json"):
            if not json_file.is_symlink() and json_file not in results:
                results.append(json_file)
    
    # Also check old locations
    for json_file in benchmarks_dir.glob("*.json"):
        results.append(json_file)
    
    return results


def analyze_benchmark_data():
    """Analyze all benchmark data and create summary."""
    result_files = find_all_benchmark_results()
    print(f"Found {len(result_files)} benchmark result files")
    
    # Group results by type and commit
    results_by_type = defaultdict(list)
    results_by_commit = defaultdict(list)
    
    for file_path in result_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Check if it's new format
            if "metadata" in data:
                benchmark_type = data["metadata"].get("benchmark_type", "unknown")
                git_commit = data["metadata"].get("git_commit", "unknown")
                timestamp = data["metadata"].get("timestamp", "unknown")
                
                results_by_type[benchmark_type].append({
                    "file": file_path.name,
                    "commit": git_commit,
                    "timestamp": timestamp,
                    "data": data
                })
                
                results_by_commit[git_commit].append({
                    "type": benchmark_type,
                    "timestamp": timestamp,
                    "file": file_path.name,
                    "data": data
                })
            else:
                # Old format
                results_by_type["legacy"].append({
                    "file": file_path.name,
                    "commit": "unknown",
                    "timestamp": file_path.stem,
                    "data": data
                })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return results_by_type, results_by_commit


def print_dashboard():
    """Print benchmark dashboard."""
    results_by_type, results_by_commit = analyze_benchmark_data()
    
    print("\n" + "="*80)
    print("BENCHMARK PERFORMANCE DASHBOARD")
    print("="*80)
    
    # Summary by type
    print("\n📊 Benchmark Results by Type:")
    for bench_type, results in sorted(results_by_type.items()):
        print(f"\n  {bench_type}: {len(results)} runs")
        
        # Get unique commits
        commits = {r["commit"] for r in results if r["commit"] != "unknown"}
        if commits:
            print(f"    Commits tracked: {len(commits)}")
            # Show last 3 commits
            recent_results = sorted(results, key=lambda x: x["timestamp"], reverse=True)[:3]
            print("    Recent runs:")
            for r in recent_results:
                commit_short = r["commit"][:8] if r["commit"] != "unknown" else "unknown"
                print(f"      - {r['timestamp']} (commit: {commit_short})")
    
    # Performance trends for specific implementations
    print("\n📈 Performance Trends (if available):")
    
    # Try to use aggregator for attention-comparison type
    if "attention-comparison" in results_by_type:
        attention_results = results_by_type["attention-comparison"]
        
        # Group by implementation
        impl_performance = defaultdict(list)
        
        for result in attention_results:
            data = result["data"]
            if "results" in data:
                for key, value in data["results"].items():
                    if isinstance(value, dict):
                        # Extract performance metrics
                        for impl in ["vanilla", "dilated", "multihead"]:
                            if impl in value and isinstance(value[impl], dict):
                                if "mean_time_ms" in value[impl]:
                                    impl_performance[impl].append({
                                        "commit": result["commit"][:8],
                                        "timestamp": result["timestamp"],
                                        "time_ms": value[impl]["mean_time_ms"],
                                        "seq_len": value.get("seq_len", "unknown")
                                    })
        
        # Show trends
        for impl, perf_data in impl_performance.items():
            if perf_data:
                print(f"\n  {impl}:")
                # Sort by timestamp
                perf_data.sort(key=lambda x: x["timestamp"])
                # Show last 5
                for p in perf_data[-5:]:
                    print(f"    {p['timestamp']}: {p['time_ms']:.2f}ms (commit: {p['commit']})")
    
    # Commit summary
    print("\n📋 Results by Commit:")
    commit_list = sorted(results_by_commit.keys(), reverse=True)[:5]  # Last 5 commits
    
    for commit in commit_list:
        if commit == "unknown":
            continue
        
        results = results_by_commit[commit]
        print(f"\n  Commit: {commit[:8]}")
        print(f"    Benchmarks run: {len(results)}")
        
        types = {r["type"] for r in results}
        print(f"    Types: {', '.join(sorted(types))}")
        
        # Try to find performance summary
        for r in results:
            if r["type"] == "all-implementations" and "summary" in r["data"]:
                summary = r["data"]["summary"]
                if summary:
                    print("    Performance summary:")
                    for impl, stats in sorted(summary.items())[:3]:  # Top 3
                        if isinstance(stats, dict) and "avg_time_ms" in stats:
                            print(f"      {impl}: {stats['avg_time_ms']:.2f}ms avg")
                    break
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if len(results_by_type) == 0:
        print("  - No benchmark results found in new format")
        print("  - Run benchmarks using updated scripts")
    elif len(results_by_commit) <= 1 or all(c == "unknown" for c in results_by_commit):
        print("  - Git commit tracking not working properly")
        print("  - Update benchmark scripts to use BenchmarkOutputManager")
    else:
        print("  - Benchmark tracking is working!")
        print("  - Run benchmarks regularly to track performance")
        
        # Check if we have recent results
        all_timestamps = []
        for results in results_by_type.values():
            for r in results:
                if r["timestamp"] != "unknown":
                    all_timestamps.append(r["timestamp"])
        
        if all_timestamps:
            latest = max(all_timestamps)
            try:
                latest_dt = datetime.strptime(latest[:10], "%Y-%m-%d")
                days_old = (datetime.now() - latest_dt).days
                if days_old > 7:
                    print(f"  - Latest results are {days_old} days old - consider running new benchmarks")
            except:
                pass
    
    print("\n" + "="*80)


def main():
    """Main function."""
    print_dashboard()
    
    # Try to create aggregated report
    print("\n📊 Attempting to create aggregated performance report...")
    
    aggregator = BenchmarkAggregator()
    
    # Find results
    by_date_dir = docs_benchmarks_dir / "by-date"
    if by_date_dir.exists():
        result_files = list(by_date_dir.rglob("*.json"))
        result_files = [f for f in result_files if not f.is_symlink()]
        
        if result_files:
            print(f"Loading {len(result_files)} result files...")
            
            for f in result_files:
                aggregator.add_result(f)
            
            # Analyze trends
            trends = aggregator.analyze_trends("all-implementations")
            
            if trends:
                print("\n📈 Performance Trends Found:")
                for impl, trend_data in trends.items():
                    print(f"\n  {impl}:")
                    if "direction" in trend_data:
                        print(f"    Trend: {trend_data['direction']}")
                    if "change_percent" in trend_data:
                        print(f"    Change: {trend_data['change_percent']:.1f}%")
                    if "best_commit" in trend_data:
                        print(f"    Best: {trend_data['best_commit'][:8]}")
                    if "worst_commit" in trend_data:
                        print(f"    Worst: {trend_data['worst_commit'][:8]}")


if __name__ == "__main__":
    main()