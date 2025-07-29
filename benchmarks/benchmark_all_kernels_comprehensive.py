#!/usr/bin/env python3
"""
Comprehensive benchmark of ALL Hilbert kernel implementations.
Tests each kernel file to identify any compilation issues.
"""

import torch
import torch.nn as nn
import time
import sys
import importlib
from pathlib import Path
from typing import Dict

sys.path.append("..")


def find_hilbert_classes(module):
    """Find all classes in a module that look like Hilbert attention implementations."""
    classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if "hilbert" in name.lower() or "unified" in name.lower():
                classes.append((name, obj))
    return classes


def test_kernel_file(kernel_path: Path) -> Dict[str, any]:
    """Test a single kernel file."""
    results = {
        "file": kernel_path.name,
        "import_success": False,
        "classes": [],
        "errors": [],
    }

    # Convert path to module name
    module_name = f"dilated_attention_pytorch.kernels.{kernel_path.stem}"

    try:
        # Import the module
        module = importlib.import_module(module_name)
        results["import_success"] = True

        # Find Hilbert classes
        classes = find_hilbert_classes(module)

        for class_name, class_obj in classes:
            class_result = {
                "name": class_name,
                "instantiate_success": False,
                "forward_success": False,
                "time_ms": None,
                "error": None,
            }

            try:
                # Try to instantiate
                kwargs = {
                    "hidden_dim": 512,
                    "num_heads": 8,
                    "segment_size": 128,
                    "dilation_rate": 1,
                }

                # Try common parameter variations
                for extra_params in [
                    {},  # Basic
                    {"hilbert_threshold": 1024},  # Common param
                    {"hilbert_threshold": 1024, "cache_size": 32},  # With cache
                    {
                        "hilbert_threshold": 1024,
                        "enable_8k_optimization": True,
                    },  # Enhanced
                ]:
                    try:
                        test_kwargs = {**kwargs, **extra_params}
                        instance = class_obj(**test_kwargs).cuda()
                        class_result["instantiate_success"] = True
                        break
                    except TypeError:
                        continue

                if class_result["instantiate_success"]:
                    # Try forward pass
                    x = torch.randn(2, 1024, 512).cuda()

                    with torch.no_grad():
                        try:
                            # Warmup
                            out = instance(x)
                            torch.cuda.synchronize()

                            # Time it
                            start = time.perf_counter()
                            for _ in range(5):
                                _ = instance(x)
                            torch.cuda.synchronize()
                            end = time.perf_counter()

                            class_result["forward_success"] = True
                            class_result["time_ms"] = (end - start) / 5 * 1000

                        except Exception as e:
                            class_result["error"] = str(e)
                            if "CompilationError" in str(type(e)):
                                class_result["error"] = (
                                    f"Triton Compilation Error: {str(e)}"
                                )

            except Exception as e:
                class_result["error"] = str(e)

            results["classes"].append(class_result)

    except Exception as e:
        results["errors"].append(f"Import error: {str(e)}")

    return results


def main():
    print("=== Comprehensive Hilbert Kernel Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print()

    # Find all kernel files
    kernel_dir = (
        Path(__file__).parent.parent / "src" / "dilated_attention_pytorch" / "kernels"
    )
    kernel_files = sorted(kernel_dir.glob("hilbert_*.py"))

    print(f"Found {len(kernel_files)} kernel files")
    print()

    all_results = []

    for kernel_file in kernel_files:
        print(f"\n--- Testing {kernel_file.name} ---")
        results = test_kernel_file(kernel_file)
        all_results.append(results)

        if results["import_success"]:
            print("✓ Import successful")

            if results["classes"]:
                for class_result in results["classes"]:
                    print(f"\n  Class: {class_result['name']}")

                    if class_result["instantiate_success"]:
                        print("    ✓ Instantiation successful")

                        if class_result["forward_success"]:
                            print(
                                f"    ✓ Forward pass successful ({class_result['time_ms']:.2f}ms)"
                            )
                        else:
                            print(f"    ✗ Forward pass failed: {class_result['error']}")
                    else:
                        print(f"    ✗ Instantiation failed: {class_result['error']}")
            else:
                print("  No Hilbert classes found")
        else:
            print(f"✗ Import failed: {results['errors']}")

    # Summary
    print("\n\n=== Summary ===")
    print(f"{'Kernel File':<45} | {'Classes':<5} | {'Working':<7} | {'Issues'}")
    print("-" * 80)

    for results in all_results:
        num_classes = len(results["classes"])
        num_working = sum(1 for c in results["classes"] if c["forward_success"])

        issues = []
        if not results["import_success"]:
            issues.append("Import failed")
        else:
            for c in results["classes"]:
                if not c["forward_success"]:
                    if "CompilationError" in str(c.get("error", "")):
                        issues.append(f"{c['name']}: Triton compilation error")
                    elif not c["instantiate_success"]:
                        issues.append(f"{c['name']}: Can't instantiate")
                    else:
                        issues.append(f"{c['name']}: Forward failed")

        issue_str = "; ".join(issues) if issues else "All working"
        print(
            f"{results['file']:<45} | {num_classes:<5} | {num_working:<7} | {issue_str}"
        )

    # Performance comparison of working implementations
    print("\n\n=== Performance Comparison (1024 tokens) ===")
    working_impls = []

    for results in all_results:
        for class_result in results["classes"]:
            if class_result["forward_success"] and class_result["time_ms"] is not None:
                working_impls.append(
                    {
                        "file": results["file"],
                        "class": class_result["name"],
                        "time_ms": class_result["time_ms"],
                    }
                )

    # Sort by performance
    working_impls.sort(key=lambda x: x["time_ms"])

    print(f"{'Implementation':<60} | {'Time (ms)':<10} | {'Relative'}")
    print("-" * 85)

    if working_impls:
        baseline = working_impls[0]["time_ms"]
        for impl in working_impls:
            relative = impl["time_ms"] / baseline
            print(
                f"{impl['file']}/{impl['class']:<60} | {impl['time_ms']:<10.2f} | {relative:.2f}x"
            )


if __name__ == "__main__":
    main()
