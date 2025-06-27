#!/usr/bin/env python3
"""
Migrate legacy benchmark data to the new unified format.

This script:
1. Finds old benchmark files
2. Detects their format
3. Converts to new format with metadata
4. Saves using BenchmarkOutputManager
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import re

benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
sys.path.insert(0, str(benchmarks_dir))

from core import BenchmarkOutputManager


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename."""
    # Try various patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2}-\d{4})',  # YYYY-MM-DD-HHMM
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{4}\d{2}\d{2})',  # YYYYMMDD
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            timestamp = match.group(1)
            # Convert to standard format
            if len(timestamp) == 8:  # YYYYMMDD
                return f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}-0000-UTC"
            elif len(timestamp) == 10:  # YYYY-MM-DD
                return f"{timestamp}-0000-UTC"
            else:
                return f"{timestamp}-UTC"
    
    # Default to current time
    return datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")


def detect_format_and_type(data: dict, filename: str) -> tuple[str, str]:
    """Detect the format and type of benchmark data."""
    # Check if it's already in new format
    if "metadata" in data and "results" in data:
        return "new", data["metadata"].get("benchmark_type", "unknown")
    
    # Detect based on content
    if "DilatedAttention" in data or "MultiheadDilatedAttention" in data:
        # It's a comparison benchmark with multiple implementations
        return "old_comparison", "all-implementations"
    
    if "memory_scaling" in data:
        return "old_memory", "memory-scaling"
    
    if "sequence_lengths" in data and "times" in data:
        return "old_sequence", "sequence-benchmark"
    
    if isinstance(data, list):
        # List of results
        if data and isinstance(data[0], dict):
            if "implementation" in data[0]:
                return "old_list", "implementation-comparison"
            elif "seq_len" in data[0]:
                return "old_list", "sequence-benchmark"
    
    return "unknown", "legacy"


def convert_old_comparison_format(data: dict, filename: str) -> dict:
    """Convert old comparison format to new format."""
    results = {}
    
    # Extract implementation results
    for impl_name in ["DilatedAttention", "MultiheadDilatedAttention", 
                      "ImprovedDilatedAttention", "ImprovedMultiheadDilatedAttention",
                      "RingDilatedAttention", "RingMultiheadDilatedAttention"]:
        if impl_name in data:
            results[impl_name] = data[impl_name]
    
    # Try to extract parameters
    params = {}
    if isinstance(data, dict):
        for key in ["batch_size", "num_heads", "head_dim", "seq_lengths"]:
            if key in data:
                params[key] = data[key]
    
    return {
        "metadata": {
            "benchmark_type": "all-implementations",
            "timestamp": extract_timestamp_from_filename(filename),
            "git_commit": "legacy",
            "git_dirty": False,
            "hardware": {
                "platform": "unknown",
                "gpu_count": 1,
                "gpu_names": ["Unknown GPU"]
            },
            "python_version": "unknown",
            "torch_version": "unknown",
            "parameters": params
        },
        "results": results,
        "summary": {}
    }


def convert_old_list_format(data: list, filename: str) -> dict:
    """Convert old list format to new format."""
    # Group by implementation
    results = {}
    
    for item in data:
        if "implementation" in item:
            impl_name = item["implementation"]
            if impl_name not in results:
                results[impl_name] = []
            results[impl_name].append(item)
        else:
            # Generic list
            if "results" not in results:
                results["results"] = []
            results["results"].append(item)
    
    return {
        "metadata": {
            "benchmark_type": "implementation-comparison",
            "timestamp": extract_timestamp_from_filename(filename),
            "git_commit": "legacy",
            "git_dirty": False,
            "hardware": {
                "platform": "unknown",
                "gpu_count": 1,
                "gpu_names": ["Unknown GPU"]
            },
            "python_version": "unknown",
            "torch_version": "unknown",
            "parameters": {}
        },
        "results": results,
        "summary": {}
    }


def migrate_file(file_path: Path) -> bool:
    """Migrate a single file."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        format_type, benchmark_type = detect_format_and_type(data, file_path.name)
        print(f"  Detected format: {format_type}")
        print(f"  Benchmark type: {benchmark_type}")
        
        if format_type == "new":
            print("  ✓ Already in new format")
            return True
        
        # Convert based on format
        if format_type == "old_comparison":
            converted_data = convert_old_comparison_format(data, file_path.name)
        elif format_type == "old_list":
            converted_data = convert_old_list_format(data, file_path.name)
        else:
            print(f"  ✗ Unknown format, creating minimal wrapper")
            converted_data = {
                "metadata": {
                    "benchmark_type": benchmark_type,
                    "timestamp": extract_timestamp_from_filename(file_path.name),
                    "git_commit": "legacy",
                    "git_dirty": False,
                    "hardware": {"platform": "unknown"},
                    "python_version": "unknown",
                    "torch_version": "unknown",
                    "parameters": {}
                },
                "results": {"data": data},
                "summary": {}
            }
        
        # Save using BenchmarkOutputManager
        output_manager = BenchmarkOutputManager(
            benchmark_type=f"migrated-{benchmark_type}",
            timestamp=converted_data["metadata"]["timestamp"]
        )
        
        # Override metadata with converted data
        output_manager.metadata = converted_data["metadata"]
        
        # Add results
        for key, value in converted_data["results"].items():
            output_manager.add_result(key, value)
        
        # Save
        output_paths = output_manager.save_results()
        
        print(f"  ✓ Migrated successfully")
        print(f"  Saved to: {output_paths['json']}")
        
        # Archive original
        archive_dir = benchmarks_dir / "archive" / "migrated"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / file_path.name
        file_path.rename(archive_path)
        print(f"  Archived original to: {archive_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def find_legacy_files():
    """Find all legacy benchmark files."""
    legacy_files = []
    
    # Check benchmarks directory
    for json_file in benchmarks_dir.glob("*.json"):
        legacy_files.append(json_file)
    
    # Check for txt files that might contain results
    for txt_file in benchmarks_dir.glob("*.txt"):
        if "benchmark" in txt_file.name.lower():
            legacy_files.append(txt_file)
    
    return legacy_files


def main():
    """Main migration function."""
    print("🔍 Searching for legacy benchmark files...")
    
    legacy_files = find_legacy_files()
    
    if not legacy_files:
        print("✅ No legacy files found!")
        return
    
    print(f"\nFound {len(legacy_files)} legacy files:")
    for f in legacy_files:
        print(f"  - {f.name}")
    
    # Ask for confirmation
    response = input("\nMigrate these files? [y/N]: ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Migrate files
    success_count = 0
    for file_path in legacy_files:
        if file_path.suffix == '.json':
            if migrate_file(file_path):
                success_count += 1
        else:
            print(f"\nSkipping non-JSON file: {file_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Migration complete: {success_count}/{len(legacy_files)} files migrated")
    
    if success_count > 0:
        print("\n📊 Run the benchmark dashboard to see migrated data:")
        print("   python scripts/benchmark_dashboard.py")


if __name__ == "__main__":
    main()