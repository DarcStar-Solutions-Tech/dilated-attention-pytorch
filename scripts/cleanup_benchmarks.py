#!/usr/bin/env python3
"""
Clean up and organize benchmark files.
"""

import re
import shutil
from pathlib import Path


def organize_benchmark_files():
    """Organize benchmark files into proper structure."""
    docs_benchmarks = Path("docs/benchmarks")
    
    # Create necessary directories
    by_type_dir = docs_benchmarks / "by-type"
    by_date_dir = docs_benchmarks / "by-date"
    latest_dir = docs_benchmarks / "latest"
    
    for d in [by_type_dir, by_date_dir, latest_dir]:
        d.mkdir(exist_ok=True)
    
    # Process files in root of benchmarks
    files_moved = 0
    for file_path in docs_benchmarks.iterdir():
        if not file_path.is_file():
            continue
            
        # Skip already organized files
        if file_path.name in ["README.md", "ORGANIZATION_PROPOSAL.md", "trend-report.md", "index.html"]:
            continue
            
        # Extract type and timestamp from filename
        # Pattern: {type}-{timestamp}.{ext}
        match = re.match(r"([\w-]+?)-(2\d{3}-\d{2}-\d{2}-\d{4}-UTC)\.(json|png|md)", file_path.name)
        if match:
            bench_type = match.group(1)
            timestamp = match.group(2)
            
            # Create type directory
            type_dir = by_type_dir / bench_type / timestamp
            type_dir.mkdir(parents=True, exist_ok=True)
            
            # Move file
            dest = type_dir / file_path.name
            if not dest.exists():
                shutil.move(str(file_path), str(dest))
                files_moved += 1
                print(f"Moved: {file_path.name} → {dest.relative_to(docs_benchmarks)}")
    
    # Clean up UUID directories
    for item in docs_benchmarks.iterdir():
        if item.is_dir() and re.match(r"[\w-]{36}", item.name):
            # Move contents and remove
            for f in item.iterdir():
                if f.is_file():
                    dest = docs_benchmarks / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                        files_moved += 1
            try:
                item.rmdir()
                print(f"Removed UUID directory: {item.name}")
            except OSError:
                # Directory not empty, remove it with contents
                shutil.rmtree(item)
                print(f"Removed UUID directory with contents: {item.name}")
    
    # Move orphaned benchmark result files from project root
    root_files = [
        Path("memory_scaling_report.json"),
        Path("benchmarks/sequence_benchmark_results.txt"),
    ]
    
    for f in root_files:
        if f.exists():
            # Generate timestamp filename
            from datetime import datetime
            timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
            bench_type = f.stem.replace("_", "-")
            
            type_dir = by_type_dir / bench_type / timestamp
            type_dir.mkdir(parents=True, exist_ok=True)
            
            new_name = f"{bench_type}-{timestamp}{f.suffix}"
            dest = type_dir / new_name
            
            shutil.move(str(f), str(dest))
            files_moved += 1
            print(f"Moved: {f} → {dest.relative_to(docs_benchmarks)}")
    
    return files_moved


def update_latest_symlinks():
    """Update latest symlinks for each benchmark type."""
    docs_benchmarks = Path("docs/benchmarks")
    by_type_dir = docs_benchmarks / "by-type"
    latest_dir = docs_benchmarks / "latest"
    
    # Clear existing links
    for item in latest_dir.iterdir():
        item.unlink(missing_ok=True)
    
    # Create new links
    links_created = 0
    for type_dir in by_type_dir.iterdir():
        if not type_dir.is_dir():
            continue
            
        # Find most recent timestamp
        timestamps = sorted([d.name for d in type_dir.iterdir() if d.is_dir()], reverse=True)
        if timestamps:
            latest_run = type_dir / timestamps[0]
            
            for file_path in latest_run.glob("*"):
                if file_path.is_file():
                    link_name = f"{type_dir.name}{file_path.suffix}"
                    link_path = latest_dir / link_name
                    
                    # Just copy instead of symlink for simplicity
                    shutil.copy2(file_path, link_path)
                    links_created += 1
    
    return links_created


def generate_index():
    """Generate index of all benchmarks."""
    docs_benchmarks = Path("docs/benchmarks")
    by_type_dir = docs_benchmarks / "by-type"
    
    lines = ["# Benchmark Index\n"]
    lines.append(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    
    # Count benchmarks by type
    for type_dir in sorted(by_type_dir.iterdir()):
        if not type_dir.is_dir():
            continue
            
        timestamps = sorted([d.name for d in type_dir.iterdir() if d.is_dir()], reverse=True)
        if timestamps:
            lines.append(f"\n## {type_dir.name.replace('-', ' ').title()}")
            lines.append(f"Total runs: {len(timestamps)}\n")
            
            # Show last 5 runs
            for ts in timestamps[:5]:
                lines.append(f"- [{ts}](by-type/{type_dir.name}/{ts}/)")
    
    # Save index
    index_path = docs_benchmarks / "INDEX.md"
    index_path.write_text("\n".join(lines))
    
    return index_path


def main():
    """Run cleanup process."""
    print("Cleaning up benchmark files...\n")
    
    # Step 1: Organize files
    files_moved = organize_benchmark_files()
    print(f"\n✓ Organized {files_moved} files")
    
    # Step 2: Update latest links
    links_created = update_latest_symlinks()
    print(f"✓ Updated {links_created} latest links")
    
    # Step 3: Generate index
    index_path = generate_index()
    print(f"✓ Generated index at {index_path}")
    
    print("\nCleanup complete!")


if __name__ == "__main__":
    from datetime import datetime
    main()