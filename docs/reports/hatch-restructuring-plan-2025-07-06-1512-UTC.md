# Hatch Standards Restructuring Plan

**Date**: 2025-07-06 15:12 UTC  
**Objective**: Restructure codebase to fully conform to Hatch standards

## Current State Analysis

The project already uses Hatchling as the build backend and has good Hatch configuration. However, it doesn't follow the recommended `src/` layout.

### Current Structure:
```
dilated-attention-pytorch/
├── dilated_attention_pytorch/    # Package source (should be in src/)
├── tests/                        # Tests (correct location)
├── benchmarks/                   # Benchmarks (correct location)
├── docs/                         # Documentation (correct location)
├── examples/                     # Examples (correct location)
└── pyproject.toml               # Hatch config (already good)
```

### Target Structure (Hatch Standards):
```
dilated-attention-pytorch/
├── src/
│   └── dilated_attention_pytorch/    # Package source
├── tests/                            # Tests
├── benchmarks/                       # Benchmarks
├── docs/                            # Documentation
├── examples/                        # Examples
├── scripts/                         # Scripts
├── pyproject.toml                   # Hatch config
├── README.md                        # Project docs
├── LICENSE                          # License
└── .gitignore                       # Git config
```

## Benefits of src/ Layout

1. **Clear separation**: Source code is clearly separated from project files
2. **Import safety**: Prevents accidentally importing from the repo root
3. **Testing isolation**: Tests run against installed package, not local files
4. **Industry standard**: Follows Python packaging best practices
5. **Tool compatibility**: Better support from various Python tools

## Migration Steps

### Step 1: Create src/ Directory Structure
```bash
mkdir -p src
git mv dilated_attention_pytorch src/
```

### Step 2: Update pyproject.toml
```toml
[tool.hatch.version]
path = "src/dilated_attention_pytorch/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/dilated_attention_pytorch"]

[tool.hatch.build]
sources = ["src"]
```

### Step 3: Update Import Paths in Tests
All test imports need to remain the same (they import the installed package):
```python
# No changes needed in tests
from dilated_attention_pytorch import DilatedAttention
```

### Step 4: Update Development Scripts
Update any scripts that directly import from the repository:
```python
# Old: sys.path.append('.')
# New: sys.path.append('src')
```

### Step 5: Update Documentation
- Update installation instructions
- Update development setup guides
- Update CLAUDE.md file paths

### Step 6: Update CI/CD
- Update GitHub Actions workflows
- Update any deployment scripts
- Update coverage paths

### Step 7: Add Hatch-Specific Files
```
.hatch/              # Hatch cache (add to .gitignore)
hatch.toml           # Global Hatch config (optional)
```

## Additional Hatch Best Practices

### 1. Environment Management
```toml
[tool.hatch.envs.default]
type = "virtual"
path = ".venv"
dependencies = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]

[tool.hatch.envs.test]
dependencies = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
]

[tool.hatch.envs.docs]
dependencies = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
]
```

### 2. Matrix Testing
```toml
[[tool.hatch.envs.test.matrix]]
python = ["3.10", "3.11", "3.12", "3.13"]
pytorch = ["2.0", "2.1", "2.2"]
```

### 3. Scripts Enhancement
```toml
[tool.hatch.envs.default.scripts]
# Development workflow
dev = "python -m pytest -xvs {args}"
cov = "pytest --cov=src/dilated_attention_pytorch {args}"
lint = ["ruff check {args:.}", "mypy {args:src}"]
fmt = ["ruff format {args:.}", "ruff check --fix {args:.}"]
all = ["fmt", "lint", "cov"]

# Release workflow
build = "hatch build"
publish = "hatch publish"
version = "hatch version {args}"
```

### 4. Build Configuration
```toml
[tool.hatch.build]
sources = ["src"]
reproducible = true
exclude = [
    "*.pyc",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
    "/LICENSE",
    "/pyproject.toml",
]

[tool.hatch.build.targets.wheel]
packages = ["src/dilated_attention_pytorch"]
```

## Migration Risks and Mitigations

### Risk 1: Breaking Existing Imports
**Mitigation**: The package name remains the same, only the source location changes

### Risk 2: CI/CD Failures
**Mitigation**: Update all CI/CD configurations before merging

### Risk 3: Development Workflow Disruption
**Mitigation**: Provide clear migration instructions for contributors

### Risk 4: Documentation Outdated
**Mitigation**: Update all documentation in the same PR

## Testing the Migration

1. Create a new branch for the migration
2. Run all tests after each step
3. Test installation from source
4. Test building wheels and sdists
5. Test in a fresh virtual environment

## Rollback Plan

If issues arise:
1. Git history preserves the original structure
2. Can revert the directory move
3. Keep the old structure in a branch temporarily

## Timeline

- **Phase 1** (1 hour): Create src/ structure and move files
- **Phase 2** (30 min): Update configurations
- **Phase 3** (1 hour): Update and test all imports
- **Phase 4** (30 min): Update documentation
- **Phase 5** (1 hour): Comprehensive testing

Total estimated time: 4 hours

## Success Criteria

- [ ] All tests pass with new structure
- [ ] Package builds successfully
- [ ] Package installs successfully
- [ ] Development commands work
- [ ] Documentation is updated
- [ ] CI/CD passes
- [ ] No import errors in examples