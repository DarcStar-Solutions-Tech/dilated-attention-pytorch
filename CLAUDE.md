# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial PyTorch implementation of **DilatedAttention** from the LongNet paper ("LongNet: Scaling Transformers to 1,000,000,000 Tokens"), plus **Ring Attention** (O(n/k) memory) and **Block-Sparse** variants for very long sequences. Version `0.2.0` (see `__version__` in `src/dilated_attention_pytorch/__init__.py`).

> **`src/` layout.** The importable package lives at `src/dilated_attention_pytorch/`. You `import dilated_attention_pytorch` (hatch builds with `sources = ["src"]`). When referencing files in this repo, use the real path — e.g. `src/dilated_attention_pytorch/base/dilated_attention.py`, not `dilated_attention_pytorch/...`.

> **Source of truth for the public API:** `src/dilated_attention_pytorch/__init__.py` (`__all__`). The class list below can drift — verify against `__all__` and the factory registries before relying on a name. Several class renames have happened recently (see "Renamed / removed names" below); do not trust class names from older docs.

## Core Architecture

Around 20 attention implementations are exported, grouped into base, dynamic, ring, and block-sparse families, all built on a shared `core/` layer (base classes, type-safe configs, factory registry, memory pools, feature detection).

### Public attention implementations (current names)

**Base** (`src/dilated_attention_pytorch/base/`):
- `DilatedAttention` — core dilated scaled-dot-product attention (segment lengths + dilation rates, pattern caching).
- `MultiheadDilatedAttention` — drop-in `nn.MultiheadAttention` replacement with MAGNETO init.
- `ImprovedDilatedAttention` / `ImprovedMultiheadDilatedAttention` — optimized (SDPA backend selection, TF32, fused QKV, memory-pool integration).
- `DistributedMultiheadDilatedAttention` (`base/distributed_dilated_attention.py`) and `HeadParallelDilatedAttentionOptimized` (`base/head_parallel_dilated_attention_optimized.py`) — multi-GPU variants.

**Dynamic segment sizing** (`src/dilated_attention_pytorch/dynamic_dilated_attention.py`):
- `DynamicDilatedAttention` / `DynamicMultiheadDilatedAttention` — auto-select segment sizes from GPU memory/hardware via `utils/dynamic_segment_selector.py`.

**Ring attention** (`src/dilated_attention_pytorch/ring/`) — O(n/k) memory via sequence splitting + `isend/irecv` ring communication:
- Standardized (v0.3.0+): `StandardRingAttention`, `HilbertRingAttention`, `DistributedRingAttention`, `BlockSparseRingAttention` (re-exported from the top-level package as `RingBlockSparseAttention` to avoid a name clash). Config: `RingAttentionConfig`.
- Legacy but still exported: `RingDilatedAttentionCorrect`, `RingDilatedAttentionSDPA`, `RingDilatedAttentionHilbertGPUOptimized`, `EnterpriseDistributedDilatedAttention` (the enterprise/DeepSpeed implementation, formerly `RingDistributedDilatedAttention`; `RingDistributedDilatedAttention` still exists as a thin subclass at `ring/distributed/ring_distributed_dilated_attention.py`).

**Block-sparse** (`src/dilated_attention_pytorch/sparse/`) — combine sparsity with the above:
- `BlockSparseAttention`, `BlockSparseDilatedAttention`, `BlockSparseMultiheadAttention`, `BlockSparseRingDistributedDilatedAttention`, `BlockSparseAdaptive` (content-adaptive, learns the pattern). `*Fixed`/`*Hilbert` wrappers also exist in the submodule.

**Models** (`src/dilated_attention_pytorch/models/`): `LongNet`, `DilatedTransformerEncoderLayer`, `DilatedTransformerDecoderLayer`.

### Renamed / removed names — do NOT reference these
- `RingDilatedAttentionProduction` / `RingDilatedAttentionProductionFixed` — **removed** ("not actually ring attention"). `RingDilatedAttentionProduction` survives only as an alias to `StandardRingAttention`; prefer `StandardRingAttention` (or `create_ring_attention`).
- `RingDilatedAttentionHilbertOptimizedFixed` — **does not exist** (use `RingDilatedAttentionHilbertGPUOptimized` or `HilbertRingAttention`).
- `BlockSparseRingDilatedAttention` → `BlockSparseRingAttention` (now `ring/block_sparse_ring_attention.py`).
- `BlockSparseRingMultiheadDilatedAttention` → `BlockSparseMultiheadAttention`.
- `BlockSparseRingDilatedAttentionHilbertPostPattern` → `BlockSparseAttentionHilbert` (`sparse/block_sparse_attention_hilbert.py`).

### Core layer (`src/dilated_attention_pytorch/core/`)
- `base.py` — `BaseDilatedAttention`, `BaseMultiheadDilatedAttention` (shared interface + caching).
- `config.py` — type-safe dataclasses with validation: `DilatedAttentionConfig`, `MultiheadConfig`, `RingAttentionConfig`, `SparseAttentionConfig`, `DistributedConfig`, `MemoryPoolConfig`.
- `factory.py` — the implementation registry behind `create_dilated_attention` / `create_multihead_dilated_attention`.
- `constants.py` — feature/hardware detection: `HAS_FLASH_ATTN`, `HAS_FLASH_ATTN_3`, `HAS_SDPA`, `HAS_XFORMERS`, `HAS_DEEPSPEED`, `GPU_TYPE` (h100/a100/v100/rtx_40xx/…/cpu), `CURRENT_OPTIMAL_SETTINGS` (per-GPU block sizes & max seq len).
- `unified_memory_pool.py` — **the current memory pool** (`UnifiedMemoryPool`). The other pool modules (`memory_pool.py`, `enhanced_memory_pool.py`, `bucketed_memory_pool.py`, `fragment_aware_pool.py`, `numa_aware_pool.py`) are **DEPRECATED (v0.4.0)** — do not build on them.
- Pattern caches: `pattern_cache.py`, `optimized_pattern_cache.py`, `simple_gpu_cache.py`.

### Key parameters (all dilated modules)
- `segment_lengths`: geometric sequence, e.g. `[2048, 4096, 8192]`.
- `dilation_rates`: matching rates, e.g. `[1, 2, 4]`.
- Sequence length must be divisible by the largest segment length.
- Shapes: `(batch, seq_len, num_heads, head_dim)` for raw attention; `(batch, seq_len, embed_dim)` for multihead. `batch_first=True`. `is_causal` selects causal/non-causal.

## Factory pattern (preferred API)

Three factories cover the families. Always prefer the factory over direct class imports for new code (`"auto"` does hardware-aware selection).

```python
from dilated_attention_pytorch import (
    create_dilated_attention, create_multihead_dilated_attention,  # core/factory.py
    create_ring_attention,                                          # ring/factory.py
    create_block_sparse_attention, get_block_sparse_preset,         # sparse/block_sparse_factory.py
    create_adaptive_block_sparse, create_multihead_block_sparse,
)
```

- `create_dilated_attention(impl, ...)` / `create_multihead_dilated_attention(impl, ...)` accept the same keys:
  `"auto"`, `"standard"`, `"improved"`, `"ring"`, `"ring_standard"`, `"ring_distributed"`, `"ring_hilbert"`, `"ring_hilbert_gpu"`, `"ring_correct"`, `"ring_sdpa"`, `"block_sparse_ring"`. (`"ring"` → `StandardRingAttention`.) The multihead factory maps these to `multihead_*` registrations internally.
- `create_ring_attention(impl, config=None, ...)` accepts `"auto"`, `"standard"`, `"hilbert"`, `"distributed"`, `"block_sparse"`.
- `create_block_sparse_attention(..., variant=...)` accepts `"auto"`, `"base"`, `"adaptive"`, `"multihead"`, `"distributed"`, `"dilated"`. Pattern types (`SparsePatternConfig.pattern_type`): `"local_window"`, `"dilated_sparse"`, `"global_local"`. Presets via `get_block_sparse_preset`: `"local"`, `"dilated"`, `"global_local"`, `"adaptive_standard"`, `"ultra_sparse"`, `"hilbert_standard"`, `"hilbert_ultra"`. (`variant="hilbert"`/`"hierarchical"` are deprecated and raise a helpful `ValueError`.)

```python
attention = create_multihead_dilated_attention(
    "auto", embed_dim=768, num_heads=12,
    segment_lengths=[2048, 4096, 8192], dilation_rates=[1, 2, 4], dropout=0.1,
)

sparse = create_block_sparse_attention(
    embed_dim=768, num_heads=12, sparsity_ratio=0.1, pattern_type="dilated_sparse",
)
```

## Development Commands

### Toolchain
- **Hatch** — environment management + task runner (config in `pyproject.toml`). The package targets `requires-python >= 3.10`; the `default`/`benchmark`/`distributed` hatch envs pin Python 3.12, and the `test` matrix covers 3.10–3.13.
- **uv** — fast dependency installation (use instead of pip).
- **torchrun** — required for any multi-GPU script.

### Installing
```bash
uv pip install -e .                # core deps: torch, einops, torchscale
uv pip install -e .[cuda]          # xformers + flash-attn (GPU acceleration)
uv pip install -e .[test]          # pytest, pytest-cov, pytest-xdist, ruff, mypy
uv pip install -e .[dev]           # test extras + pre-commit + hatch
uv pip install -e .[benchmark]     # plotly, kaleido, timm
uv pip install -e .[distributed]   # deepspeed, fairscale, lightning
uv pip install -e .[all]           # test + dev + benchmark + distributed
```
Note: `xformers` and `flash-attn` are **not** core dependencies — they live in the `[cuda]` extra. `plotly` is in `[benchmark]`.

### Testing (hatch `default` env scripts)
```bash
hatch run test          # pytest tests  (NO coverage)
hatch run test-cov      # pytest with coverage (term-missing + xml + html)
hatch run test-fast     # pytest -x  (stop on first failure)
hatch run test-debug    # pytest -xvs

# Single file / single test (run pytest directly)
pytest tests/base/test_dilated_attention.py
pytest tests/base/test_dilated_attention.py::TestDilatedAttention::test_forward -v
pytest tests/ -k "ring and not distributed"

# Coverage directly
pytest tests/ --cov=src/dilated_attention_pytorch --cov-report=html

# Parallel (test env): hatch run test:parallel  ->  pytest -n auto
```
Pytest markers (`pyproject.toml`): `slow`, `gpu`, `distributed`, `performance`, `benchmark`. Note `filterwarnings = ["error", ...]` — warnings that aren't explicitly ignored fail the suite.

### Multi-GPU tests (MUST use torchrun)
```bash
torchrun --nproc_per_node=2 tests/ring/base/test_ring_attention.py
torchrun --nproc_per_node=4 tests/ring/distributed/test_distributed_ring_attention.py
# distributed env helper: hatch run distributed:test-distributed  ->  pytest tests/ -k distributed
```

### Quick verification (root scripts)
```bash
python verify_all_components.py        # imports + smoke-tests each component
python validate_changes.py             # AST-based method/structure validation (no test run)
python scripts/test_comprehensive.py   # quick comprehensive harness
```

### Code quality
```bash
hatch run lint          # ruff check .
hatch run format        # ruff format .
hatch run format-check  # ruff format --check .
hatch run fix           # ruff format + ruff check --fix
hatch run typecheck     # mypy src/dilated_attention_pytorch
hatch run all           # format -> lint -> typecheck -> test-cov
```
Ruff: line length 100, target py312, broad rule set (E/W/F/I/B/C4/UP/ARG/SIM/N/RUF/PL/TRY/PERF). Mypy is strict (`disallow_untyped_defs`, etc.) and runs against `python_version = "3.13"`.

### Benchmarking
```bash
# Entry points live under benchmarks/ (run directly):
python benchmarks/run_benchmark.py
torchrun --nproc_per_node=4 benchmarks/benchmark_block_sparse_ring_attention.py
# Shared utilities: benchmarks/core/ (base_benchmark.py + utils/{distributed,memory,timing,data}.py)
```
Heads-up: the hatch `benchmark:run` script is `python benchmark.py`, but there is **no** `benchmark.py` in the repo — invoke the scripts in `benchmarks/` directly (or fix the script) rather than `hatch run benchmark:run`.

## Implementation guidelines

### Device & dtype
- CUDA when available, else CPU. Prefer float16/bfloat16 on Ampere+ (CC ≥ 8.0); fall back to float32 on Pascal/older (detected via `torch.cuda.get_device_capability()` and `utils/gpu_utils.py`).
- Backend auto-selection order: Flash Attention 3 (H100/H200, 1.5–2×) → FA2 (A100/RTX 30xx/40xx) → xformers → SDPA → math. See `core/constants.py` and `utils/flash_attention_utils.py` / `utils/flash_attention_3_utils.py`.

### Ring Attention — CRITICAL implementation rules
Most ring-attention bugs come from violating these:

1. **Process local sequences only — split before projecting.**
   ```python
   # WRONG — defeats O(n/k) memory: projects the full sequence first
   qkv = self.qkv_proj(x)            # x is [batch, seq_len, embed_dim]
   # CORRECT — slice the local chunk, then project
   if self.world_size > 1 and not already_split:
       x_local = x[:, start:end, :].contiguous()
   qkv = self.qkv_proj(x_local)
   ```
2. **Never use `all_gather`** — it is O(n²) communication and defeats ring attention. Use `isend`/`irecv`. (Implementations that used `all_gather` were removed.)
3. **Ring communication pattern** — pass to `(rank+1) % world_size`, receive from `(rank-1) % world_size` with `isend`/`irecv`, then `wait()`.
4. **Tensor contiguity + aggressive cleanup** — `.contiguous()` before any P2P; `gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()` for long sequences; pre-allocate comm buffers.
5. **Backend hint by LOCAL length** — `seq_len_hint = max(segment_lengths); if dist.is_initialized(): seq_len_hint //= dist.get_world_size()`.

### Multi-GPU Ring Attention fixes (apply these; lucidrains-style)
```python
send_tensor = send_tensor.contiguous(); receive_buffer = receive_buffer.contiguous()
ops = [dist.P2POp(dist.isend, send_tensor, send_to_rank),
       dist.P2POp(dist.irecv, receive_buffer, recv_from_rank)]
reqs = dist.batch_isend_irecv(ops)
for req in reqs: req.wait()
dist.barrier()  # prevents race conditions
```
Without these: CUDA illegal-memory-access errors, non-contiguous warnings, hangs. See `docs/guides/ring-attention-multi-gpu-fixes.md`.

### NCCL env vars (network tuning)
`NCCL_SOCKET_IFNAME` (interface), `NCCL_IB_DISABLE` (no InfiniBand), `NCCL_P2P_DISABLE` (compatibility).

### Hilbert curve optimization
Apply per-segment for cache locality; use GPU-aware backend selection; preserve numerical stability with LSE accumulation; benchmark against standard ordering. Utilities: `utils/hilbert_curve.py`, `utils/hilbert_attention_mixin.py`, experimental kernels in `src/dilated_attention_pytorch/kernels/`.

### Performance expectations
- Single GPU standard attention: ~32K tokens.
- Multi-GPU ring: linear scaling to billions of tokens (memory O(n/k)); ~10–15% communication overhead with a correct implementation.

## Project Structure Rules

### Maintain the organized directory structure
Place new files in the correct directory — never in the repo root unless project-level config.

1. **Documentation** → `docs/guides/` (guides/tutorials) or `docs/reports/` (reports/analysis). Never in root.
2. **Tests** → `tests/<area>/test_*.py` (areas: `base`, `ring`, `ring/base`, `ring/distributed`, `ring/hilbert`, `sparse`, `core`, `utils`, `models`, `integration`, `misc`). Never in root.
3. **Benchmarks** → `benchmarks/benchmark_*.py`; results as `benchmarks/*.md`/`*.txt`.
4. **Analysis** → `analysis/*_analysis.py`.
5. **Utility scripts** → `scripts/` (`scripts/debug/`, `scripts/demo/`).
6. **Source code** → `src/dilated_attention_pytorch/{base,ring,sparse,models,utils,core,kernels}/`.

Rules: check the appropriate directory exists before creating; when in doubt, ask where to place a file; keep naming consistent within each directory.

Root directory should contain only: `README.md`, `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CLAUDE.md`, `PROJECT_STRUCTURE.md`, package/git config (`pyproject.toml`, `setup.py`, `.gitignore`, …), and `validate_changes.py` / `verify_all_components.py`.

## Documentation naming conventions

- `docs/guides/` — **permanent** names, kebab-case: `{feature}-guide.md`, `api-{module}.md`, `tutorial-{topic}.md`.
- **Timestamped** results use `YYYY-MM-DD-HHMM-UTC` (UTC; `datetime.utcnow().strftime('%Y-%m-%d-%H%M-UTC')`), timestamp last before the extension:
  - `docs/benchmarks/benchmark-{description}-<ts>.{md,png,json}`
  - `docs/feasibility/{topic}-feasibility-<ts>.md`
  - `docs/reports/defect-{type}-<ts>.md`, `docs/reports/{analysis}-<ts>.md`
- `docs/archive/` — historical/obsolete docs.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
ALWAYS follow the Project Structure Rules above when creating new files.
