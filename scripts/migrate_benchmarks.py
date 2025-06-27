#!/usr/bin/env python3
"""
Migrate existing benchmark scripts to use the new unified output system.

This script:
1. Updates existing benchmark scripts to use BenchmarkOutputManager
2. Cleans up duplicate benchmark results
3. Organizes existing results into the new structure
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def clean_duplicate_benchmarks():
    """Remove duplicate benchmark files and organize remaining ones."""
    docs_benchmarks = Path("docs/benchmarks")
    
    # Files to process
    all_files = list(docs_benchmarks.glob("*.json")) + list(docs_benchmarks.glob("*.png")) + list(docs_benchmarks.glob("*.md"))
    
    # Group by type and date
    file_groups = {}
    for file_path in all_files:
        # Extract benchmark type and timestamp from filename
        match = re.match(r"([\w-]+?)-(2\d{3}-\d{2}-\d{2}-\d{4}-UTC)\.(json|png|md)", file_path.name)
        if match:
            bench_type = match.group(1)
            timestamp = match.group(2)
            extension = match.group(3)
            
            key = (bench_type, timestamp)
            if key not in file_groups:
                file_groups[key] = {}
            file_groups[key][extension] = file_path
    
    # Process each group
    print("Organizing benchmark files...")
    for (bench_type, timestamp), files in file_groups.items():
        # Skip if already organized
        if any(p.parent.name == timestamp for p in files.values()):
            continue
            
        # Create organized directory
        type_dir = docs_benchmarks / "by-type" / bench_type / timestamp
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files
        for ext, file_path in files.items():
            dest = type_dir / file_path.name
            if not dest.exists():
                shutil.move(str(file_path), str(dest))
                print(f"  Moved: {file_path.name} → {dest.relative_to(docs_benchmarks)}")
    
    # Clean up UUID directories
    for uuid_dir in docs_benchmarks.glob("*-*-*-*-*"):
        if uuid_dir.is_dir() and len(uuid_dir.name) == 36:  # UUID format
            # Move contents up and remove directory
            for file_path in uuid_dir.iterdir():
                if file_path.is_file():
                    dest = docs_benchmarks / file_path.name
                    if not dest.exists():
                        shutil.move(str(file_path), str(dest))
            uuid_dir.rmdir()
            print(f"  Removed UUID directory: {uuid_dir.name}")
    
    # Update latest symlinks
    update_latest_links()


def update_latest_links():
    """Update the latest symlinks for each benchmark type."""
    docs_benchmarks = Path("docs/benchmarks")
    by_type_dir = docs_benchmarks / "by-type"
    latest_dir = docs_benchmarks / "latest"
    
    # Clear old links
    for link in latest_dir.glob("*"):
        if link.is_symlink() or link.is_file():
            link.unlink()
    
    # Create new links for each type
    for type_dir in by_type_dir.iterdir():
        if not type_dir.is_dir():
            continue
            
        # Find most recent timestamp
        timestamps = sorted([d.name for d in type_dir.iterdir() if d.is_dir()], reverse=True)
        if timestamps:
            latest_timestamp = timestamps[0]
            latest_run_dir = type_dir / latest_timestamp
            
            # Link each file
            for file_path in latest_run_dir.glob("*"):
                if file_path.is_file():
                    link_name = f"{type_dir.name}{file_path.suffix}"
                    link_path = latest_dir / link_name
                    try:
                        # Calculate relative path from latest to the file
                        rel_path = Path("..") / file_path.relative_to(docs_benchmarks)
                        link_path.symlink_to(rel_path)
                    except OSError:
                        # Copy if symlinks not supported
                        shutil.copy2(file_path, link_path)


def update_benchmark_script(script_path: Path):
    """Update a benchmark script to use the new output system."""
    # Read the script
    content = script_path.read_text()
    
    # Check if already using new system
    if "BenchmarkOutputManager" in content:
        return False
    
    # Pattern replacements
    replacements = [
        # Replace BenchmarkOutputHelper with BenchmarkOutputManager
        (r"from benchmark_output_helper import BenchmarkOutputHelper",
         "from benchmarks.core import BenchmarkOutputManager"),
        
        (r"helper = BenchmarkOutputHelper\(__file__\)",
         "output_manager = BenchmarkOutputManager.from_existing_script(__file__)"),
        
        # Replace save methods
        (r"helper\.save_json\(([\w\s,]+)\)",
         r"output_manager.add_result('results', \1)\noutput_manager.save_results()"),
        
        (r"helper\.save_plot\(([\w\s,]+)\)",
         r"output_manager.save_plot(\1)"),
        
        # Add proper imports
        (r"(import [\w\s,]+\n)+",
         r"\1from benchmarks.core import BenchmarkOutputManager\n"),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    if modified:
        # Backup original
        backup_path = script_path.with_suffix(script_path.suffix + ".bak")
        shutil.copy2(script_path, backup_path)
        
        # Write updated content
        script_path.write_text(content)
        print(f"Updated: {script_path.name}")
        return True
    
    return False


def create_benchmark_dashboard():
    """Create an HTML dashboard for benchmark results."""
    # Add benchmarks to path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from benchmarks.core import BenchmarkAggregator, BenchmarkStorage
    
    storage = BenchmarkStorage()
    aggregator = BenchmarkAggregator(storage)
    
    # Generate trend report
    trend_report = aggregator.generate_trend_report(days=30)
    
    # Create dashboard HTML
    dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Dashboard - Dilated Attention PyTorch</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #007acc; }}
        .regression {{ background: #fee; border-left-color: #d00; }}
        .improvement {{ background: #efe; border-left-color: #0a0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        .trend-up {{ color: #d00; }}
        .trend-down {{ color: #0a0; }}
        .trend-stable {{ color: #666; }}
        .generated {{ color: #666; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Dashboard</h1>
        <div class="generated">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
        
        <div class="markdown-content">
            {convert_markdown_to_html(trend_report)}
        </div>
        
        <h2>Quick Links</h2>
        <ul>
            <li><a href="latest/">Latest Benchmark Results</a></li>
            <li><a href="by-type/">Results by Type</a></li>
            <li><a href="by-date/">Results by Date</a></li>
            <li><a href="archive/">Archived Results</a></li>
        </ul>
    </div>
</body>
</html>
"""
    
    # Save dashboard
    dashboard_path = Path("docs/benchmarks/index.html")
    dashboard_path.write_text(dashboard_html)
    print(f"Created dashboard: {dashboard_path}")


def convert_markdown_to_html(markdown: str) -> str:
    """Simple markdown to HTML conversion."""
    html = markdown
    
    # Headers
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    
    # Line breaks
    html = re.sub(r"  $", r"<br>", html, flags=re.MULTILINE)
    
    # Paragraphs
    html = re.sub(r"\n\n", r"</p><p>", html)
    html = f"<p>{html}</p>"
    
    # Lists
    html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    html = re.sub(r"(<li>.*</li>\n?)+", r"<ul>\g<0></ul>", html)
    
    # Tables (simple support)
    lines = html.split("\n")
    in_table = False
    new_lines = []
    
    for line in lines:
        if "|" in line and "---" not in line:
            if not in_table:
                new_lines.append("<table>")
                in_table = True
            
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if any("**" in cell for cell in cells):
                # Header row
                row = "<tr>" + "".join(f"<th>{cell}</th>" for cell in cells) + "</tr>"
            else:
                row = "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
            new_lines.append(row)
        elif in_table and "|" not in line:
            new_lines.append("</table>")
            in_table = False
            if line.strip():
                new_lines.append(line)
        elif "---" not in line:
            new_lines.append(line)
    
    if in_table:
        new_lines.append("</table>")
    
    return "\n".join(new_lines)


def main():
    """Run the migration process."""
    print("Starting benchmark migration...")
    
    # Step 1: Clean up duplicates and organize files
    print("\n1. Cleaning duplicate benchmarks...")
    clean_duplicate_benchmarks()
    
    # Step 2: Update benchmark scripts
    print("\n2. Updating benchmark scripts...")
    benchmark_dir = Path("benchmarks")
    updated_count = 0
    
    for script_path in benchmark_dir.glob("benchmark_*.py"):
        if script_path.name == "benchmark_output_helper.py":
            continue
        
        if update_benchmark_script(script_path):
            updated_count += 1
    
    print(f"  Updated {updated_count} scripts")
    
    # Step 3: Create dashboard
    print("\n3. Creating benchmark dashboard...")
    create_benchmark_dashboard()
    
    # Step 4: Generate summary report
    print("\n4. Generating summary report...")
    # Import is already done with sys.path modification above
    from benchmarks.core import BenchmarkAggregator, BenchmarkStorage
    
    storage = BenchmarkStorage()
    aggregator = BenchmarkAggregator(storage)
    
    # Save trend report
    report = aggregator.generate_trend_report(days=30)
    report_path = Path("docs/benchmarks/trend-report.md")
    report_path.write_text(report)
    
    print(f"\n✅ Migration complete!")
    print(f"   - Organized benchmark files")
    print(f"   - Updated {updated_count} scripts")
    print(f"   - Created dashboard at docs/benchmarks/index.html")
    print(f"   - Generated trend report at docs/benchmarks/trend-report.md")


if __name__ == "__main__":
    main()