# Reference materials

Cached external references for the unified attention architecture
(`docs/guides/unified-attention-architecture.md`), so we don't need to re-fetch them.
PDFs live in `papers/`. Repos are recorded as pointers (we do not vendor their code; note
the license and the specific files of interest instead).

## Papers (cached in `papers/`)

| File | Title / what it is | Source | Why it matters to us |
|---|---|---|---|
| `native-sparse-attention-2502.11089.pdf` | **Native Sparse Attention (NSA)** — Yuan et al., DeepSeek (ACL 2025 Best Paper) | arXiv [2502.11089](https://arxiv.org/abs/2502.11089) | The design template for our content-adaptive sparse policy (L2): 3 branches (compression + selection + sliding window), natively trainable, 9×/6× fwd/bwd @64k, quality ≥ full attention. |
| `flashattention-3-2407.08608.pdf` | **FlashAttention-3** — Shah, Dao et al. | arXiv [2407.08608](https://arxiv.org/abs/2407.08608) | Kernel backend (L1): 1.5–2× over FA2 on Hopper, FP8 forward. Adopt as the dense/static Flash backend. |
| `muon-scalable-moonlight-2502.16982.pdf` | **Muon is Scalable for LLM Training (Moonlight)** — Liu et al., Moonshot AI | arXiv [2502.16982](https://arxiv.org/abs/2502.16982) | Optimizer (orthogonal to attention): ~2× compute-optimal efficiency (~52% AdamW FLOPs), validated to frontier scale. Adopt for the training recipe. |
| `flexattention-2412.05496.pdf` | **Flex Attention: A Programming Model for Generating Optimized Attention Kernels** | arXiv [2412.05496](https://arxiv.org/abs/2412.05496) | The L1+static-L2 substrate: programmable `score_mod`/`mask_mod` lowered to a fused FlashAttention kernel with `BlockMask` block-sparsity. Depend on, don't hand-roll. |

## Repos (pointers — not vendored)

| Repo | License | Form | Disposition | Notes |
|---|---|---|---|---|
| [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) | MIT | Triton (fwd+bwd) | **PORT base** for the learned-selection (L2-adaptive) path | Production-oriented NSA: online top-k block-selection kernel that avoids materializing the full attention matrix; fused selected+sliding kernel; e2e training via the `flame` framework. Strongest reference for content-adaptive selection. |
| [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) | MIT | Triton + FlexAttention path | **PORT / LEARN-FROM** | Faithful NSA: `SparseAttention` exposing `compress_block_size`, `selection_block_size`, `num_selected_blocks`, `sliding_window_size`; real Triton fwd+bwd kernels (not masking-only). |
| [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) | MIT | Triton + pure-PyTorch | **LEARN-FROM** | Triton ring-flash kernel + Tri Dao FA integration. We have our own ring family; use as a cross-check for L3. |
| [lucidrains/local-attention](https://github.com/lucidrains/local-attention) | MIT | pure PyTorch | **LEARN-FROM only** | Reference local windowed attention. |
| [KellerJordan/Muon](https://github.com/KellerJordan/Muon) · [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) | MIT | PyTorch | **ADOPT / DEPEND-ON** | Muon optimizer + a distributed (ZeRO-1-style) implementation with the standard `muon_params` vs `adamw_params` split. |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3 | CUDA/CuTe | **ADOPT** (FA3) / **TRACK** (FA4) | FA3 is beta (Hopper); FA4 (CuTeDSL, Hopper+Blackwell) is alpha. |

## Provenance
- Papers downloaded from arXiv on 2026-05-30. arXiv IDs are stable; PDFs are the cited versions.
- Repo dispositions are from the deep-research passes summarized in
  `docs/guides/unified-attention-architecture.md` §"Prior art". Repo maturity can drift —
  re-check before depending on a specific commit.
- Items still pending fact-check (DeepSeek V3.2 DSA, DeepSeek V4, MLA, MoBA, Muon variants,
  AdEMAMix) will have references added here once verified.
