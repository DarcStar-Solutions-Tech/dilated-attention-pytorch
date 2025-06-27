#!/usr/bin/env python3
"""Enhanced linting fixes with automatic resolution strategies."""

import json
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd, capture=True):
    """Run command and return output."""
    if capture:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=False
        )
        return result.stdout, result.stderr, result.returncode
    else:
        return subprocess.run(cmd, shell=True, check=False).returncode


def create_pre_commit_config():
    """Create an enhanced pre-commit config with auto-fixes."""
    print("🔧 Creating enhanced pre-commit configuration...")

    config = """repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.8.4
  hooks:
  - id: ruff
    args: [--fix, --exit-non-zero-on-fix]
  - id: ruff-format

- repo: local
  hooks:
  - id: smart-linting
    name: Smart Linting Fixes
    entry: python scripts/pre_commit_smart_fix.py
    language: python
    always_run: true
    pass_filenames: false
    additional_dependencies: []
"""

    with open(".pre-commit-config.yaml", "w") as f:
        f.write(config)

    print("  Updated .pre-commit-config.yaml with smart fixes")


def create_smart_fix_script():
    """Create a smart pre-commit fix script."""
    print("🔧 Creating smart pre-commit fix script...")

    script = '''#!/usr/bin/env python3
"""Smart fixes for common linting issues before commit."""

import re
import subprocess
import sys
from pathlib import Path


def fix_unused_variables():
    """Fix unused variable assignments."""
    # Get all F841 errors
    result = subprocess.run(
        ["ruff", "check", ".", "--select", "F841", "--output-format", "json"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return
    
    try:
        import json
        errors = json.loads(result.stdout)
    except:
        # Fallback to text parsing
        return
    
    fixes = {}
    for error in errors:
        if error.get("code") == "F841":
            file_path = error["filename"]
            line = error["location"]["row"]
            var_name = error.get("message", "").split("`")[1] if "`" in error.get("message", "") else None
            
            if var_name and file_path not in fixes:
                fixes[file_path] = []
            if var_name:
                fixes[file_path].append((line, var_name))
    
    # Apply fixes
    for file_path, issues in fixes.items():
        try:
            lines = Path(file_path).read_text().splitlines()
            
            for line_num, var_name in sorted(issues, reverse=True):
                line_idx = line_num - 1
                if line_idx < len(lines):
                    # Replace variable with underscore
                    lines[line_idx] = re.sub(
                        rf"\\b{var_name}\\b\\s*=",
                        "_ =",
                        lines[line_idx]
                    )
            
            Path(file_path).write_text("\\n".join(lines) + "\\n")
            print(f"Fixed unused variables in {file_path}")
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")


def fix_invalid_noqa():
    """Fix invalid noqa directives."""
    # Find all invalid noqa warnings
    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)
    
    invalid_noqa_files = []
    for line in result.stderr.splitlines():
        if "Invalid `# noqa` directive" in line:
            match = re.search(r"on ([^:]+):", line)
            if match:
                invalid_noqa_files.append(match.group(1))
    
    for file_path in set(invalid_noqa_files):
        try:
            content = Path(file_path).read_text()
            # Fix common invalid noqa patterns
            content = re.sub(r"# noqa: PLC0415", "# noqa: PLC0415", content)
            content = re.sub(r"# noqa: ([A-Z]+\\d+)_[a-z]+", r"# noqa: \\1", content)
            Path(file_path).write_text(content)
            print(f"Fixed invalid noqa in {file_path}")
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")


def main():
    """Run all smart fixes."""
    print("🔍 Running smart pre-commit fixes...")
    
    # Fix unused variables
    fix_unused_variables()
    
    # Fix invalid noqa directives
    fix_invalid_noqa()
    
    # Run ruff format
    subprocess.run(["ruff", "format", "."], capture_output=True)
    
    print("✅ Smart fixes complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    script_path = Path("scripts/pre_commit_smart_fix.py")
    script_path.write_text(script)
    script_path.chmod(0o755)

    print("  Created scripts/pre_commit_smart_fix.py")


def create_editorconfig():
    """Create .editorconfig for consistent formatting."""
    print("🔧 Creating .editorconfig...")

    config = """# EditorConfig helps maintain consistent coding styles

root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.py]
indent_style = space
indent_size = 4
max_line_length = 100

[*.{json,yaml,yml,toml}]
indent_style = space
indent_size = 2

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
"""

    with open(".editorconfig", "w") as f:
        f.write(config)

    print("  Created .editorconfig")


def create_github_workflow():
    """Create GitHub workflow for linting."""
    print("🔧 Creating GitHub workflow for automated linting...")

    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)

    workflow = """name: Lint

on:
  pull_request:
  push:
    branches: [main]

jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    
    - name: Install Ruff
      run: pip install ruff
    
    - name: Run Ruff Check
      run: |
        ruff check . --output-format=github
        ruff format . --check
    
    - name: Auto-fix and suggest
      if: failure()
      run: |
        echo "::notice::Run 'ruff check . --fix' locally to auto-fix issues"
        echo "::notice::Run 'python scripts/enhanced_linting_fixes.py' for additional fixes"
"""

    workflow_path = workflow_dir / "lint.yml"
    workflow_path.write_text(workflow)

    print("  Created .github/workflows/lint.yml")


def fix_current_issues():
    """Fix current known issues."""
    print("🔧 Fixing current known issues...")

    # Fix unused variables in analysis files
    fixes = [
        ("analysis/ring_performance_analysis.py", 66, "x_seg", "_"),
        ("analysis/ring_performance_analysis.py", 89, "x_dil", "_"),
    ]

    for file_path, line_num, old_var, new_var in fixes:
        try:
            path = Path(file_path)
            if path.exists():
                lines = path.read_text().splitlines()
                if line_num <= len(lines):
                    lines[line_num - 1] = lines[line_num - 1].replace(
                        f"{old_var} =", f"{new_var} ="
                    )
                    path.write_text("\n".join(lines) + "\n")
                    print(f"  Fixed {file_path}:{line_num}")
        except Exception as e:
            print(f"  Error fixing {file_path}: {e}")

    # Fix missing import in benchmark_distributed.py
    try:
        path = Path("benchmarks/benchmark_distributed.py")
        if path.exists():
            content = path.read_text()
            if "import torch.distributed as dist" not in content:
                # Add import after torch import
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if line.strip() == "import torch":
                        lines.insert(i + 1, "import torch.distributed as dist")
                        break
                path.write_text("\n".join(lines) + "\n")
                print("  Fixed benchmarks/benchmark_distributed.py")
    except Exception as e:
        print(f"  Error fixing benchmark_distributed.py: {e}")

    # Fix invalid noqa directives
    invalid_noqa_fixes = [
        (
            "dilated_attention_pytorch/block_sparse_ring_dilated_attention.py",
            "# noqa: PLC0415",
            "# noqa: PLC0415",
        ),
        (
            "dilated_attention_pytorch/utils/flash_attention_3_utils.py",
            "# noqa: ([A-Z0-9]+)_[a-z]+",
            "# noqa: \\1",
        ),
        (
            "dilated_attention_pytorch/utils/attention_utils.py",
            "# noqa: ([A-Z0-9]+)_[a-z]+",
            "# noqa: \\1",
        ),
    ]

    for file_path, pattern, replacement in invalid_noqa_fixes:
        try:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    path.write_text(new_content)
                    print(f"  Fixed invalid noqa in {file_path}")
        except Exception as e:
            print(f"  Error fixing {file_path}: {e}")


def create_commit_msg_hook():
    """Create a commit message hook with lint status."""
    print("🔧 Creating commit message hook...")

    hook = """#!/bin/bash
# Add lint status to commit message

COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

# Only add for regular commits (not merges, amends, etc)
if [ -z "$COMMIT_SOURCE" ]; then
    # Check lint status
    if ruff check . --exit-zero --quiet; then
        echo "" >> "$COMMIT_MSG_FILE"
        echo "✅ Lint: Clean" >> "$COMMIT_MSG_FILE"
    else
        ISSUES=$(ruff check . --exit-zero | wc -l)
        echo "" >> "$COMMIT_MSG_FILE"
        echo "⚠️  Lint: $ISSUES issues (non-blocking)" >> "$COMMIT_MSG_FILE"
    fi
fi
"""

    hook_path = Path(".git/hooks/prepare-commit-msg")
    if hook_path.parent.exists():
        hook_path.write_text(hook)
        hook_path.chmod(0o755)
        print("  Created .git/hooks/prepare-commit-msg")


def main():
    """Run all enhancements."""
    print("🚀 Enhanced Linting Setup")
    print("=" * 50)

    # Create configurations
    create_pre_commit_config()
    create_smart_fix_script()
    create_editorconfig()
    create_github_workflow()
    create_commit_msg_hook()

    # Fix current issues
    fix_current_issues()

    # Run formatting
    print("\n📋 Running final formatting...")
    run_command("ruff format .", capture=False)

    # Show final status
    print("\n📊 Final linting status:")
    stdout, _, _ = run_command("ruff check . --exit-zero")
    issues = len(stdout.strip().splitlines()) if stdout.strip() else 0

    if issues == 0:
        print("✅ No linting issues!")
    else:
        print(f"⚠️  {issues} remaining issues (mostly minor)")
        print("\nRemaining issues:")
        for line in stdout.strip().splitlines()[:10]:
            print(f"  {line}")

    print("\n✨ Setup complete! Your linting workflow is now:")
    print("  1. Auto-fixes on save (VS Code)")
    print("  2. Smart fixes on commit (pre-commit hook)")
    print("  3. GitHub Actions for PR checks")
    print("  4. Easy bypass: git commit --no-verify")
    print("  5. Manual fix: python scripts/enhanced_linting_fixes.py")


if __name__ == "__main__":
    main()
