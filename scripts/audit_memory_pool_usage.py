"""
Audit memory pool usage across all attention modules.

This script analyzes all attention implementations to verify:
1. Which modules support memory pooling
2. Default settings for memory pool usage
3. Consistency of implementation
4. Opportunities for improvement
"""

import ast
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime


class MemoryPoolAuditor(ast.NodeVisitor):
    """AST visitor to analyze memory pool usage patterns."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.classes = []
        self.current_class = None
        self.has_memory_pool_import = False
        self.memory_pool_params = []
        self.memory_pool_usage = []
        self.allocate_calls = []
        self.deallocate_calls = []

    def visit_Import(self, node):
        """Check for memory pool imports."""
        for alias in node.names:
            if "memory_pool" in alias.name or "MemoryPool" in alias.name:
                self.has_memory_pool_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check for memory pool imports from modules."""
        if node.module and "memory_pool" in node.module:
            self.has_memory_pool_import = True
        for alias in node.names:
            if "memory_pool" in alias.name or "MemoryPool" in alias.name:
                self.has_memory_pool_import = True
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Track class definitions."""
        self.current_class = node.name
        self.classes.append(
            {
                "name": node.name,
                "bases": [self._get_name(base) for base in node.bases],
                "has_init": False,
                "memory_pool_param": None,
                "memory_pool_default": None,
            }
        )
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Analyze function definitions for memory pool usage."""
        if self.current_class and node.name == "__init__":
            # Check for memory pool parameters
            for arg in node.args.args:
                if "memory_pool" in arg.arg.lower():
                    self.classes[-1]["has_init"] = True
                    self.classes[-1]["memory_pool_param"] = arg.arg

                    # Try to find default value
                    defaults = node.args.defaults
                    if defaults:
                        default_idx = len(node.args.args) - len(defaults)
                        arg_idx = node.args.args.index(arg)
                        if arg_idx >= default_idx:
                            default = defaults[arg_idx - default_idx]
                            self.classes[-1]["memory_pool_default"] = self._get_value(
                                default
                            )

        # Check for allocate/deallocate patterns
        if node.name == "_allocate_tensor" or "allocate" in node.name:
            self.allocate_calls.append(
                {
                    "function": node.name,
                    "class": self.current_class,
                    "line": node.lineno,
                }
            )

        if node.name == "_deallocate_tensor" or "deallocate" in node.name:
            self.deallocate_calls.append(
                {
                    "function": node.name,
                    "class": self.current_class,
                    "line": node.lineno,
                }
            )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track memory pool attribute access."""
        if isinstance(node.attr, str) and "memory_pool" in node.attr:
            self.memory_pool_usage.append(
                {
                    "attribute": node.attr,
                    "class": self.current_class,
                    "line": node.lineno,
                }
            )
        self.generic_visit(node)

    def _get_name(self, node):
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _get_value(self, node):
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        return str(node)


def audit_file(filepath: Path) -> Dict:
    """Audit a single Python file for memory pool usage."""
    with open(filepath, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
        auditor = MemoryPoolAuditor(str(filepath))
        auditor.visit(tree)

        return {
            "filepath": str(filepath),
            "filename": filepath.name,
            "has_memory_pool_import": auditor.has_memory_pool_import,
            "classes": auditor.classes,
            "memory_pool_usage_count": len(auditor.memory_pool_usage),
            "allocate_calls": auditor.allocate_calls,
            "deallocate_calls": auditor.deallocate_calls,
        }
    except Exception as e:
        return {"filepath": str(filepath), "filename": filepath.name, "error": str(e)}


def analyze_memory_pool_consistency(results: List[Dict]) -> Dict:
    """Analyze consistency of memory pool implementations."""
    consistency_report = {
        "total_files": len(results),
        "files_with_imports": 0,
        "files_with_usage": 0,
        "classes_with_param": 0,
        "default_enabled": 0,
        "default_disabled": 0,
        "allocate_patterns": set(),
        "deallocate_patterns": set(),
        "inconsistencies": [],
    }

    for result in results:
        if result.get("error"):
            continue

        if result["has_memory_pool_import"]:
            consistency_report["files_with_imports"] += 1

        if result["memory_pool_usage_count"] > 0:
            consistency_report["files_with_usage"] += 1

        for cls in result.get("classes", []):
            if cls.get("memory_pool_param"):
                consistency_report["classes_with_param"] += 1

                default = cls.get("memory_pool_default")
                if default is True:
                    consistency_report["default_enabled"] += 1
                elif default is False:
                    consistency_report["default_disabled"] += 1

        for call in result.get("allocate_calls", []):
            consistency_report["allocate_patterns"].add(call["function"])

        for call in result.get("deallocate_calls", []):
            consistency_report["deallocate_patterns"].add(call["function"])

        # Check for inconsistencies
        if result["has_memory_pool_import"] and result["memory_pool_usage_count"] == 0:
            consistency_report["inconsistencies"].append(
                {"file": result["filename"], "issue": "Import without usage"}
            )

    consistency_report["allocate_patterns"] = list(
        consistency_report["allocate_patterns"]
    )
    consistency_report["deallocate_patterns"] = list(
        consistency_report["deallocate_patterns"]
    )

    return consistency_report


def generate_recommendations(results: List[Dict], consistency: Dict) -> List[Dict]:
    """Generate recommendations for memory pool improvements."""
    recommendations = []

    # Files that could benefit from memory pooling
    large_tensor_modules = [
        "multihead_dilated_attention.py",
        "improved_multihead_dilated_attention.py",
        "distributed_dilated_attention.py",
        "improved_distributed_dilated_attention.py",
        "transformer.py",
        "long_net.py",
    ]

    for result in results:
        if result.get("error"):
            continue

        filename = result["filename"]

        # Check if module could benefit from memory pooling
        if filename in large_tensor_modules and not result["has_memory_pool_import"]:
            recommendations.append(
                {
                    "file": filename,
                    "priority": "HIGH",
                    "recommendation": "Add memory pool support for large tensor allocations",
                    "reason": "This module handles large attention matrices that could benefit from pooled allocation",
                }
            )

        # Check for incomplete implementations
        if result["has_memory_pool_import"] and not result.get("allocate_calls"):
            recommendations.append(
                {
                    "file": filename,
                    "priority": "MEDIUM",
                    "recommendation": "Complete memory pool integration",
                    "reason": "Memory pool is imported but no allocation methods found",
                }
            )

        # Check for missing deallocation
        if len(result.get("allocate_calls", [])) > len(
            result.get("deallocate_calls", [])
        ):
            recommendations.append(
                {
                    "file": filename,
                    "priority": "HIGH",
                    "recommendation": "Add corresponding deallocate calls",
                    "reason": f"Found {len(result['allocate_calls'])} allocate calls but only {len(result['deallocate_calls'])} deallocate calls",
                }
            )

    # Global recommendations
    if consistency["default_enabled"] > 0:
        recommendations.append(
            {
                "file": "GLOBAL",
                "priority": "MEDIUM",
                "recommendation": "Consider disabling memory pool by default",
                "reason": f"{consistency['default_enabled']} classes have memory pool enabled by default, which may add overhead for small sequences",
            }
        )

    return recommendations


def main():
    """Run the memory pool audit."""
    # Find all Python files in the project
    project_root = Path(__file__).parent.parent
    attention_dir = project_root / "dilated_attention_pytorch"

    python_files = []
    for file in attention_dir.rglob("*.py"):
        if not any(part.startswith(".") for part in file.parts):
            python_files.append(file)

    print(f"Auditing {len(python_files)} Python files...")

    # Audit each file
    results = []
    for filepath in sorted(python_files):
        result = audit_file(filepath)
        results.append(result)

        # Print progress
        if result.get("classes"):
            for cls in result["classes"]:
                if cls.get("memory_pool_param"):
                    print(f"  ✓ {filepath.name}: {cls['name']} has memory pool support")

    # Analyze consistency
    consistency = analyze_memory_pool_consistency(results)

    # Generate recommendations
    recommendations = generate_recommendations(results, consistency)

    # Create detailed report
    report = {
        "audit_date": datetime.now().isoformat(),
        "summary": {
            "total_files": consistency["total_files"],
            "files_with_memory_pool": consistency["files_with_imports"],
            "files_actively_using": consistency["files_with_usage"],
            "classes_with_support": consistency["classes_with_param"],
            "default_enabled": consistency["default_enabled"],
            "default_disabled": consistency["default_disabled"],
        },
        "patterns": {
            "allocate_methods": consistency["allocate_patterns"],
            "deallocate_methods": consistency["deallocate_patterns"],
        },
        "recommendations": recommendations,
        "detailed_results": results,
    }

    # Save report
    report_path = project_root / "memory_pool_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit complete! Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MEMORY POOL AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total files analyzed: {consistency['total_files']}")
    print(f"Files with memory pool imports: {consistency['files_with_imports']}")
    print(f"Files actively using memory pool: {consistency['files_with_usage']}")
    print(f"Classes with memory pool parameter: {consistency['classes_with_param']}")
    print(f"  - Default enabled: {consistency['default_enabled']}")
    print(f"  - Default disabled: {consistency['default_disabled']}")

    print(
        f"\nHigh priority recommendations: {sum(1 for r in recommendations if r['priority'] == 'HIGH')}"
    )
    print(
        f"Medium priority recommendations: {sum(1 for r in recommendations if r['priority'] == 'MEDIUM')}"
    )

    if consistency["inconsistencies"]:
        print(f"\nInconsistencies found: {len(consistency['inconsistencies'])}")
        for inc in consistency["inconsistencies"][:3]:
            print(f"  - {inc['file']}: {inc['issue']}")


if __name__ == "__main__":
    main()
