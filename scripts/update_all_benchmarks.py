#!/usr/bin/env python3
"""
Update all benchmark scripts to use the unified benchmark storage system.

This ensures all benchmarks:
1. Track git commits
2. Save results in a consistent format
3. Can be aggregated across runs
"""

import json
import sys
from pathlib import Path

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
sys.path.insert(0, str(benchmarks_dir))


def find_benchmark_scripts():
    """Find all benchmark scripts that need updating."""
    scripts_to_check = [
        "benchmark_all.py",
        "benchmark_all_implementations.py",
        "benchmark_distributed.py",
        "benchmark_extreme_sequences.py",
        "benchmark_flash_attention_3.py",
        "benchmark_long_sequences.py",
        "benchmark_ring_billion_tokens.py",
        "benchmark_sequence_limits.py",
        "comprehensive_benchmark_comparison.py",
        "simple_benchmark_comparison.py",
    ]

    scripts_needing_update = []
    scripts_already_updated = []

    for script_name in scripts_to_check:
        script_path = benchmarks_dir / script_name
        if not script_path.exists():
            continue

        # Check if it already uses BenchmarkOutputManager
        content = script_path.read_text()
        if "BenchmarkOutputManager" in content:
            scripts_already_updated.append(script_name)
        else:
            scripts_needing_update.append(script_name)

    return scripts_already_updated, scripts_needing_update


def add_output_manager_to_script(script_path: Path):
    """Add BenchmarkOutputManager usage to a script."""
    content = script_path.read_text()

    # Check if it's already importing sys
    if "import sys" not in content:
        # Add sys import after other imports
        import_index = content.find("import torch")
        if import_index != -1:
            content = content[:import_index] + "import sys\n" + content[import_index:]

    # Add BenchmarkOutputManager import
    import_section = """
# Import unified benchmark output management
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager
"""

    # Find a good place to add the import
    torch_import = content.find("import torch")
    if torch_import != -1:
        # Find the end of the import section
        _ = content[:torch_import].split("\n")
        import_end = torch_import
        for i, line in enumerate(content[torch_import:].split("\n")):
            if (
                line
                and not line.startswith(("import", "from", "#"))
                and not line.strip() == ""
            ):
                break
            import_end += len(line) + 1

        content = content[:import_end] + import_section + content[import_end:]

    # Add example usage at the end of main()
    example_usage = """
    # Save results using unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="{script_type}",
        parameters={{
            # Add relevant parameters here
            "device": str(device),
            "dtype": str(dtype),
        }}
    )
    
    # Add results (adapt based on your result structure)
    output_manager.add_result("benchmark_results", results)
    
    # Save all outputs
    output_paths = output_manager.save_results()
    print(f"\\nResults saved to: {{output_paths}}")
"""

    # Update based on script name
    script_type = script_path.stem.replace("benchmark_", "").replace("_", "-")
    example_usage = example_usage.format(script_type=script_type)

    return content, example_usage


def main():
    """Main function."""
    print("🔍 Checking benchmark scripts...")

    already_updated, needs_update = find_benchmark_scripts()

    print(f"\n✅ Already using BenchmarkOutputManager ({len(already_updated)}):")
    for script in already_updated:
        print(f"  - {script}")

    print(f"\n❌ Need to be updated ({len(needs_update)}):")
    for script in needs_update:
        print(f"  - {script}")

    # Generate patches for scripts that need updating
    if needs_update:
        print("\n📝 Generating update patches...")

        patches_dir = Path(__file__).parent / "benchmark_patches"
        patches_dir.mkdir(exist_ok=True)

        for script_name in needs_update:
            script_path = benchmarks_dir / script_name

            # Read current content
            _ = script_path.read_text()

            # Generate patch
            updated_content, example_usage = add_output_manager_to_script(script_path)

            # Save patch
            patch_path = patches_dir / f"{script_name}.patch"
            with open(patch_path, "w") as f:
                f.write("=== Import Section ===\n")
                f.write("Add this import section:\n\n")
                f.write("# Import unified benchmark output management\n")
                f.write("from pathlib import Path\n")
                f.write("sys.path.insert(0, str(Path(__file__).parent))\n")
                f.write("from core import BenchmarkOutputManager\n\n")

                f.write("=== At End of main() or where results are collected ===\n")
                f.write(example_usage)

            print(f"  Created patch: {patch_path}")

    # Check for JSON files that need migration
    print("\n🔍 Checking for benchmark results that need migration...")

    json_files = list(benchmarks_dir.glob("*.json"))
    txt_files = list(benchmarks_dir.glob("*.txt"))

    if json_files or txt_files:
        print(
            f"\nFound {len(json_files)} JSON and {len(txt_files)} TXT files in benchmarks/"
        )
        print("These should be migrated to the new storage structure.")

        # Check format of JSON files
        old_format_files = []
        new_format_files = []

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Check if it has the new format
                if "metadata" in data and "results" in data:
                    new_format_files.append(json_file.name)
                else:
                    old_format_files.append(json_file.name)
            except:
                pass

        if old_format_files:
            print(f"\n🔄 Files needing format migration ({len(old_format_files)}):")
            for f in old_format_files[:5]:  # Show first 5
                print(f"  - {f}")
            if len(old_format_files) > 5:
                print(f"  ... and {len(old_format_files) - 5} more")

    print("\n✅ Analysis complete!")

    if needs_update:
        print("\nNext steps:")
        print("1. Review the generated patches in scripts/benchmark_patches/")
        print("2. Apply the patches to update each benchmark script")
        print("3. Test the updated scripts to ensure they save results properly")
        print("4. Run scripts/migrate_benchmarks.py to migrate old results")


if __name__ == "__main__":
    main()
