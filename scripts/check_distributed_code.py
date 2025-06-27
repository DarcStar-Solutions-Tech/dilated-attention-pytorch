#!/usr/bin/env python3
"""
Check distributed code for common pitfalls and issues.

This script performs static analysis to catch common distributed training bugs.
"""

import ast
import sys
from pathlib import Path


class DistributedCodeChecker(ast.NodeVisitor):
    """AST visitor to check for distributed code issues."""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: list[tuple[int, str]] = []
        self.in_distributed_context = False
        self.has_rank_check = False
        self.has_init_check = False
        self.collective_ops: set[str] = {
            "all_reduce",
            "all_gather",
            "broadcast",
            "reduce",
            "scatter",
            "gather",
            "all_to_all",
            "barrier",
            "send",
            "recv",
            "isend",
            "irecv",
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions."""
        # Check if function uses distributed operations
        func_src = ast.unparse(node)
        if any(op in func_src for op in self.collective_ops):
            self.in_distributed_context = True

            # Check for proper initialization checks
            if not self._has_init_check(node):
                self.issues.append(
                    (
                        node.lineno,
                        f"Function '{node.name}' uses distributed ops but doesn't check torch.distributed.is_initialized()",
                    )
                )

        self.generic_visit(node)
        self.in_distributed_context = False

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Check for collective operations
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.collective_ops:
                # Check if we're in a try-except block
                if not self._in_try_except(node):
                    self.issues.append(
                        (
                            node.lineno,
                            f"Collective operation '{node.func.attr}' not wrapped in try-except for error handling",
                        )
                    )

            # Check for common mistakes
            if node.func.attr == "cuda" and isinstance(node.func.value, ast.Name):
                # Check for hardcoded device indices
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, int):
                        self.issues.append(
                            (
                                node.lineno,
                                f"Hardcoded CUDA device index {node.args[0].value}. Use get_rank() or LOCAL_RANK instead",
                            )
                        )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignments."""
        # Check for common distributed variables
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in ["rank", "world_size", "local_rank"]:
                    # Make sure these are assigned from distributed module
                    if not self._is_from_distributed(node.value):
                        self.issues.append(
                            (
                                node.lineno,
                                f"Variable '{target.id}' should be assigned from torch.distributed",
                            )
                        )

        self.generic_visit(node)

    def _has_init_check(self, node: ast.FunctionDef) -> bool:
        """Check if function has distributed initialization check."""
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute) and stmt.func.attr == "is_initialized":
                    return True
        return False

    def _in_try_except(self, node: ast.AST) -> bool:
        """Check if node is within a try-except block."""
        # Simple check - in real implementation would walk up the AST
        return False

    def _is_from_distributed(self, node: ast.AST) -> bool:
        """Check if expression is from torch.distributed."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                return "distributed" in ast.unparse(node.func)
        return False


def check_file(filepath: Path) -> list[tuple[str, int, str]]:
    """Check a single Python file for distributed code issues."""
    issues = []

    try:
        with open(filepath) as f:
            content = f.read()

        tree = ast.parse(content)
        checker = DistributedCodeChecker(str(filepath))
        checker.visit(tree)

        for line, issue in checker.issues:
            issues.append((str(filepath), line, issue))

    except Exception as e:
        print(f"Error checking {filepath}: {e}")

    return issues


def main():
    """Main function to check all distributed code."""
    # Find all relevant Python files
    project_root = Path(__file__).parent.parent
    distributed_files = [
        "dilated_attention_pytorch/ring_distributed_dilated_attention.py",
        "dilated_attention_pytorch/improved_distributed_dilated_attention.py",
        "dilated_attention_pytorch/block_sparse_ring_distributed_dilated_attention.py",
    ]

    all_issues = []

    print("Checking distributed code for common issues...")
    print("=" * 60)

    for file_path in distributed_files:
        full_path = project_root / file_path
        if full_path.exists():
            issues = check_file(full_path)
            all_issues.extend(issues)

            if issues:
                print(f"\n{file_path}:")
                for _, line, issue in issues:
                    print(f"  Line {line}: {issue}")

    # Additional checks
    print("\n" + "=" * 60)
    print("Additional Checks:")

    # Check for synchronization issues
    sync_patterns = [
        ("torch.cuda.synchronize()", "May cause deadlock in distributed setting"),
        ("time.sleep", "Can cause rank desynchronization"),
        ("input()", "Will hang in distributed training"),
        ("breakpoint()", "Will cause ranks to desynchronize"),
    ]

    for file_path in distributed_files:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path) as f:
                content = f.read()

            for pattern, issue in sync_patterns:
                if pattern in content:
                    print(f"{file_path}: Contains '{pattern}' - {issue}")

    # Summary
    print("\n" + "=" * 60)
    if all_issues:
        print(f"Found {len(all_issues)} potential issues in distributed code")
        sys.exit(1)
    else:
        print("No critical issues found in distributed code ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
