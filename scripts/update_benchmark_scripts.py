#!/usr/bin/env python3
"""
Update all benchmark scripts to use BenchmarkOutputManager.
"""

from pathlib import Path

# Scripts that need updating
SCRIPTS_TO_UPDATE = [
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

benchmarks_dir = Path(__file__).parent.parent / "benchmarks"


def update_benchmark_script(script_name: str) -> bool:
    """Update a single benchmark script."""
    script_path = benchmarks_dir / script_name

    if not script_path.exists():
        print(f"❌ {script_name} not found")
        return False

    content = script_path.read_text()

    # Check if already updated
    if "BenchmarkOutputManager" in content:
        print(f"✓ {script_name} already updated")
        return True

    # Add imports
    import_block = """
# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager
"""

    # Find where to insert imports
    if "from pathlib import Path" not in content:
        # Add Path import
        import_idx = content.find("import torch")
        if import_idx == -1:
            import_idx = content.find("import numpy")
        if import_idx != -1:
            content = (
                content[:import_idx]
                + "from pathlib import Path\n"
                + content[import_idx:]
            )

    # Add sys import if needed
    if "import sys" not in content:
        import_idx = content.find("import torch")
        if import_idx == -1:
            import_idx = content.find("import numpy")
        if import_idx != -1:
            content = content[:import_idx] + "import sys\n" + content[import_idx:]

    # Add BenchmarkOutputManager import after torch import
    torch_idx = content.find("import torch")
    if torch_idx != -1:
        # Find end of imports
        lines = content[torch_idx:].split("\n")
        insert_idx = torch_idx
        for i, line in enumerate(lines):
            if (
                line
                and not line.startswith(("import", "from", "#", " "))
                and line.strip()
            ):
                break
            insert_idx += len(line) + 1

        content = content[:insert_idx] + import_block + content[insert_idx:]

    # Now add output saving logic based on script type
    if script_name == "benchmark_all_implementations.py":
        content = add_all_implementations_output(content)
    elif script_name == "benchmark_distributed.py":
        content = add_distributed_output(content)
    elif script_name == "benchmark_extreme_sequences.py":
        content = add_extreme_sequences_output(content)
    elif script_name == "benchmark_flash_attention_3.py":
        content = add_flash_attention_output(content)
    elif script_name == "benchmark_long_sequences.py":
        content = add_long_sequences_output(content)
    elif script_name == "benchmark_ring_billion_tokens.py":
        content = add_ring_billion_output(content)
    elif script_name == "benchmark_sequence_limits.py":
        content = add_sequence_limits_output(content)
    elif script_name == "comprehensive_benchmark_comparison.py":
        content = add_comprehensive_output(content)
    elif script_name == "simple_benchmark_comparison.py":
        content = add_simple_comparison_output(content)

    # Save updated script
    script_path.write_text(content)
    print(f"✅ Updated {script_name}")
    return True


def add_all_implementations_output(content: str) -> str:
    """Add output manager to benchmark_all_implementations.py."""
    # Find where results are stored - usually after a loop that collects timings

    # Look for the save_results function or similar
    if "json.dump" in content:
        # Replace json.dump with output manager
        save_code = """
    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="all-implementations",
        parameters={
            "num_runs": num_runs,
            "warm_up": warm_up,
        }
    )
    
    # Add results for each implementation
    for impl_name, impl_results in results.items():
        output_manager.add_result(impl_name, impl_results)
    
    # Save results
    output_paths = output_manager.save_results()
    print(f"\\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")
"""

        # Find and replace the json.dump section
        json_start = content.find("with open")
        if json_start != -1:
            json_end = content.find("\n\n", json_start)
            if json_end == -1:
                json_end = content.find("\n    print", json_start)
            if json_end != -1:
                content = content[:json_start] + save_code + content[json_end:]

    elif "if __name__" in content:
        # Add before the final print statements
        main_idx = content.rfind("if __name__")
        # Find the end of main
        lines = content[main_idx:].split("\n")
        insert_point = None
        _ = "    "

        for i, line in enumerate(lines):
            if "print" in line and "Results" not in line:
                insert_point = main_idx + sum(len(l) + 1 for l in lines[:i])
                break

        if insert_point:
            save_code = """
    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="all-implementations",
        parameters={}
    )
    
    # Add all results
    output_manager.add_result("benchmark_results", results)
    
    # Save results
    output_paths = output_manager.save_results()
    print(f"\\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")

"""
            content = content[:insert_point] + save_code + content[insert_point:]

    return content


def add_distributed_output(content: str) -> str:
    """Add output manager to benchmark_distributed.py."""
    save_code = """
    # Use unified benchmark output management
    if rank == 0:  # Only save from rank 0
        output_manager = BenchmarkOutputManager(
            benchmark_type="distributed",
            parameters={
                "world_size": world_size,
                "backend": backend,
            }
        )
        
        # Add results
        output_manager.add_result("distributed_results", results)
        
        # Save results
        output_paths = output_manager.save_results()
        print(f"\\nResults saved to:")
        for path_type, path in output_paths.items():
            print(f"  {path_type}: {path}")
"""

    # Add at the end of main
    if "if __name__" in content:
        # Find a good insertion point
        main_end = content.rfind("}")
        if main_end == -1:
            main_end = len(content) - 1

        # Insert before the last few lines
        lines = content[:main_end].split("\n")
        insert_idx = len(content)

        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() and not lines[i].strip().startswith("print"):
                insert_idx = sum(len(l) + 1 for l in lines[: i + 1])
                break

        content = content[:insert_idx] + save_code + content[insert_idx:]

    return content


def add_extreme_sequences_output(content: str) -> str:
    """Add output manager to benchmark_extreme_sequences.py."""
    return add_generic_output(content, "extreme-sequences")


def add_flash_attention_output(content: str) -> str:
    """Add output manager to benchmark_flash_attention_3.py."""
    return add_generic_output(content, "flash-attention-3")


def add_long_sequences_output(content: str) -> str:
    """Add output manager to benchmark_long_sequences.py."""
    return add_generic_output(content, "long-sequences")


def add_ring_billion_output(content: str) -> str:
    """Add output manager to benchmark_ring_billion_tokens.py."""
    return add_generic_output(content, "ring-billion-tokens")


def add_sequence_limits_output(content: str) -> str:
    """Add output manager to benchmark_sequence_limits.py."""
    return add_generic_output(content, "sequence-limits")


def add_comprehensive_output(content: str) -> str:
    """Add output manager to comprehensive_benchmark_comparison.py."""
    return add_generic_output(content, "comprehensive-comparison")


def add_simple_comparison_output(content: str) -> str:
    """Add output manager to simple_benchmark_comparison.py."""
    return add_generic_output(content, "simple-comparison")


def add_generic_output(content: str, benchmark_type: str) -> str:
    """Add generic output manager code."""
    # Look for where results are collected
    result_vars = ["results", "all_results", "benchmark_results", "timings"]

    for var in result_vars:
        if f"{var} = " in content or f"{var}[" in content:
            # Found results variable
            save_code = f"""
    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="{benchmark_type}",
        parameters={{}}
    )
    
    # Add results
    output_manager.add_result("results", {var})
    
    # Save results
    output_paths = output_manager.save_results()
    print(f"\\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {{path_type}}: {{path}}")
"""

            # Find end of main or end of file
            if "if __name__" in content:
                # Add at end of main block
                main_idx = content.rfind("if __name__")
                remaining = content[main_idx:]

                # Find last print or end
                lines = remaining.split("\n")
                insert_offset = len(remaining)

                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() and not lines[i].startswith(("#", "print")):
                        insert_offset = sum(len(l) + 1 for l in lines[: i + 1])
                        break

                insert_idx = main_idx + insert_offset
                content = content[:insert_idx] + save_code + content[insert_idx:]
                break

    return content


def main():
    """Update all benchmark scripts."""
    print("🔧 Updating benchmark scripts...")

    success_count = 0
    for script in SCRIPTS_TO_UPDATE:
        if update_benchmark_script(script):
            success_count += 1

    print(f"\n✅ Updated {success_count}/{len(SCRIPTS_TO_UPDATE)} scripts")

    if success_count == len(SCRIPTS_TO_UPDATE):
        print("\n🎉 All benchmark scripts have been updated!")
        print("\nNext steps:")
        print("1. Run the updated benchmarks to generate new data")
        print("2. Use scripts/migrate_legacy_benchmarks.py to convert old data")
    else:
        print("\n⚠️  Some scripts could not be updated automatically")
        print("Please check the failed scripts manually")


if __name__ == "__main__":
    main()
