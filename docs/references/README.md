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
| `flexattention-2412.05496.pdf` | **Flex Attention: A Programming Model for Generating Optimized Attention Kernels** | arXiv [2412.05496](https://arxiv.org/abs/2412.05496) | The L1+static-L2 substrate: programmable `score_mod`/`mask_mod` lowered to a fused FlashAttention kernel with `BlockMask` block-sparsity. Depend on, don't hand-roll. (Resolved: also supports learned selection, at a per-step mask-build cost.) |
| `moba-2502.13189.pdf` | **MoBA: Mixture of Block Attention** — Moonshot AI | arXiv [2502.13189](https://arxiv.org/abs/2502.13189) | Parameter-less centroid top-k **block** routing = *exactly* our L2 "block-centroid top-k" idea. Learn-from the gating. |
| `flashmoba-2511.11571.pdf` | **FlashMoBA** — MIT Han Lab + NVIDIA | arXiv [2511.11571](https://arxiv.org/abs/2511.11571) | Open CUDA kernel realizing MoBA block-selection with real fwd+bwd savings (≤14.7× vs FA2). **PORT** as our canonical learned-selection kernel. |
| `deepseek-v3.2-dsa-2512.02556.pdf` | **DeepSeek-V3.2 / DeepSeek Sparse Attention (DSA)** — DeepSeek | arXiv [2512.02556](https://arxiv.org/abs/2512.02556) | Production NSA successor: lightning indexer + top-k **token** selection on MLA, O(L²)→O(Lk). Offer as a token-level L2 policy option. |
| `deepseek-v2-mla-2405.04434.pdf` | **DeepSeek-V2 / Multi-head Latent Attention (MLA)** — DeepSeek | arXiv [2405.04434](https://arxiv.org/abs/2405.04434) | Low-rank KV-cache compression (~57× inference). Inference-side; learn-from / deprioritize for training cost. Underpins DSA/V4. |
| `ademamix-2409.03137.pdf` | **The AdEMAMix Optimizer** — Pagliardini et al., Apple | arXiv [2409.03137](https://arxiv.org/abs/2409.03137) | Dual-EMA (fast + very-slow) optimizer; ~half AdamW's tokens at 1.3B. Experimental option + basis for the novel Muon×AdEMAMix. |
| `dion-2504.05295.pdf` | **Dion: distributed orthonormalized updates** — Microsoft Research | arXiv [2504.05295](https://arxiv.org/abs/2504.05295) | Sharded-weight Muon successor (power iteration, FSDP/TP-friendly, ~3B-validated). **TRACK** for ring/distributed training. |
| `kimi-k2-muonclip-2507.20534.pdf` | **Kimi K2 / MuonClip** — Moonshot AI | arXiv [2507.20534](https://arxiv.org/abs/2507.20534) | QK-clip stabilizes Muon at 1T scale (15.5T tokens, zero loss spikes). **ADOPT** MuonClip at scale. |

## Repos (pointers — not vendored)

| Repo | License | Form | Disposition | Notes |
|---|---|---|---|---|
| [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) | MIT | Triton (fwd+bwd) | **PORT base** for the learned-selection (L2-adaptive) path | Production-oriented NSA: online top-k block-selection kernel that avoids materializing the full attention matrix; fused selected+sliding kernel; e2e training via the `flame` framework. Strongest reference for content-adaptive selection. |
| [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) | MIT | Triton + FlexAttention path | **PORT / LEARN-FROM** | Faithful NSA: `SparseAttention` exposing `compress_block_size`, `selection_block_size`, `num_selected_blocks`, `sliding_window_size`; real Triton fwd+bwd kernels (not masking-only). |
| [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) | MIT | Triton + pure-PyTorch | **LEARN-FROM** | Triton ring-flash kernel + Tri Dao FA integration. We have our own ring family; use as a cross-check for L3. |
| [lucidrains/local-attention](https://github.com/lucidrains/local-attention) | MIT | pure PyTorch | **LEARN-FROM only** | Reference local windowed attention. |
| [KellerJordan/Muon](https://github.com/KellerJordan/Muon) · [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) | MIT | PyTorch | **ADOPT / DEPEND-ON** | Muon optimizer + a distributed (ZeRO-1-style) implementation with the standard `muon_params` vs `adamw_params` split. |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3 | CUDA/CuTe | **ADOPT** (FA3) / **TRACK** (FA4) | FA3 is beta (Hopper); FA4 (CuTeDSL, Hopper+Blackwell) is alpha. |
| [mit-han-lab/flash-moba](https://github.com/mit-han-lab/flash-moba) · [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) | BSD-3 / MIT | CUDA (FA2-style) / PyTorch | **PORT** (FlashMoBA = canonical learned-selection kernel) / **LEARN-FROM** (MoBA gating) | Block-centroid top-k routing; FlashMoBA gather-densify-scatter kernel, real fwd+bwd savings. Block-level → composes with our sparse-ring L3. |
| [apple/ml-ademamix](https://github.com/apple/ml-ademamix) | MIT | PyTorch + JAX/Optax | **EXPERIMENT** | AdEMAMix optimizer; basis for the novel Muon×AdEMAMix combination (no published precedent). |
| DeepSeek DSA kernels: [FlashMLA](https://github.com/deepseek-ai/FlashMLA) · [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | MIT | CUDA / TileLang | **TRACK / port-if-token-level** | Lightning-indexer + top-k token selection kernels (DeepGEMM indexer logits, FlashMLA sparse attention). For the optional DSA token-level policy. |

## Provenance
- Papers downloaded from arXiv on 2026-05-30. arXiv IDs are stable; PDFs are the cited versions.
- Repo dispositions are from the deep-research passes summarized in
  `docs/guides/unified-attention-architecture.md` §"Prior art". Repo maturity can drift —
  re-check before depending on a specific commit.
- All previously-pending items (DeepSeek V3.2 DSA, DeepSeek V4, MLA, MoBA, Muon variants,
  AdEMAMix) were fact-checked on 2026-05-30 and their references added above. Verdicts are in
  `docs/guides/unified-attention-architecture.md` §9. **DeepSeek V4** is a recent (~Apr 2026)
  *preview* — treat its specifics as time-sensitive. **DeepSeek V4** has no cached PDF here
  (preview tech report is on HuggingFace, not arXiv); see the model card linked in §9 discussion.
