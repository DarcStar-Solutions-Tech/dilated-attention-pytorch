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

## Scaling laws, data limits & capability (cached in `papers/`) — supports §6.1, §11.1

Gathered for the 500T/50T training-cost and capability analyses (the token-budget sweep and the
active-param/sparsity sweet-spot). Magnitudes were adversarially fact-checked in the workflows behind
those sections; several papers sit **outside** the regimes they measured (sparsity ≤50×, bases ≤~1T), so
extrapolation to 50T/500B is flagged in the doc.

| File | Title / what it is | Source | Why it matters to us |
|---|---|---|---|
| `scaling-laws-kaplan-2001.08361.pdf` | **Scaling Laws for Neural LMs** — Kaplan et al. 2020 | arXiv [2001.08361](https://arxiv.org/abs/2001.08361) | Power-law loss + "compute-optimal stops *before* convergence" → §6.1 "no crisp *completely-trained* point." |
| `chinchilla-hoffmann-2203.15556.pdf` | **Training Compute-Optimal LLMs (Chinchilla)** — Hoffmann et al. 2022 | arXiv [2203.15556](https://arxiv.org/abs/2203.15556) | ~20 tokens/param compute-optimal — the **floor** anchor for the §6.1 token budget (D ≈ 20×active), explicitly *not* "completely trained." |
| `chinchilla-replication-epoch-2404.10102.pdf` | **Chinchilla Scaling: A Replication Attempt** — Besiroglu/Epoch 2024 | arXiv [2404.10102](https://arxiv.org/abs/2404.10102) | Re-fit confirms ~20:1 and the loss form `L=E+A/Nᵃ+B/Dᵇ` used in §6.1. |
| `beyond-chinchilla-inference-2401.00448.pdf` | **Beyond Chinchilla-Optimal (inference-aware)** — Sardana et al. 2024 | arXiv [2401.00448](https://arxiv.org/abs/2401.00448) | Loss keeps falling far past 20:1 → rationale for the §6.1 frontier-overtraining 100–500T band. |
| `routed-lm-scaling-clark-2202.01169.pdf` | **Unified Scaling Laws for Routed (MoE) LMs** — Clark et al. 2022 | arXiv [2202.01169](https://arxiv.org/abs/2202.01169) | Effective-param count **saturates** (~80–900B dense-equiv) → debunks the unbounded √(N·a) heuristic in §11.1; routing gains diminish. |
| `fine-grained-moe-scaling-2402.07871.pdf` | **Scaling Laws for Fine-Grained MoE** — Krajewski/Ludziejewski et al. 2024 | arXiv [2402.07871](https://arxiv.org/abs/2402.07871) | MoE data-exponent β > dense (needs longer training); granularity scaling → §6.1 MoE-scaling lane. |
| `optimal-sparsity-moe-abnar-2501.12370.pdf` | **Parameters vs FLOPs: Optimal Sparsity for MoE** — Abnar et al. 2025 | arXiv [2501.12370](https://arxiv.org/abs/2501.12370) | Compute-optimal D anchors to **active** params; over-sparsifying a small-for-compute model *hurts*; max tested ~50× → the key §11.1 sparsity-risk source (100–1000× is extrapolation). |
| `moe-leverage-scaling-ling-2507.17702.pdf` | **Towards Greater Leverage: Scaling Laws for Efficient MoE** — Ling Team 2025 | arXiv [2507.17702](https://arxiv.org/abs/2507.17702) | MoE allocation law (compute-optimal MoE is smaller but trained on more data); activation ratios 0.8–10.9% (500× far beyond). |
| `optimal-sparsity-reasoning-nakamura-2508.18672.pdf` | **Optimal Sparsity of MoE for Reasoning** — Nakamura et al. 2025 | arXiv [2508.18672](https://arxiv.org/abs/2508.18672) | At matched loss, more **active** compute → higher *reasoning*; low loss can *mask* weak reasoning at high sparsity → §11.1 "active is the capability axis." |
| `will-we-run-out-of-data-2211.04325.pdf` | **Will We Run Out of Data?** — Villalobos/Epoch 2022/24 | arXiv [2211.04325](https://arxiv.org/abs/2211.04325) | ~300T-token human-text stock (CI 100–1000T) — the **data wall** binding §6.1 (capacity-fill at ~1e16 tokens is infeasible). |
| `data-constrained-scaling-2305.16264.pdf` | **Scaling Data-Constrained LMs** — Muennighoff et al. 2023 | arXiv [2305.16264](https://arxiv.org/abs/2305.16264) | ~4 epochs near-lossless (decays by ~16) → bounds repetition; §6.1/§11.1 data-feasibility of the 100–500T band. |
| `lm-memorization-morris-2505.24832.pdf` | **How Much Do LMs Memorize?** — Morris et al. 2025 | arXiv [2505.24832](https://arxiv.org/abs/2505.24832) | ~3.6 bits/param capacity (dense, ≤1.5B — flagged as extrapolation) → §11.1 knowledge-ceiling. |
| `emergent-abilities-wei-2206.07682.pdf` | **Emergent Abilities of LLMs** — Wei et al. 2022 | arXiv [2206.07682](https://arxiv.org/abs/2206.07682) | Emergence onset thresholds (~1e23 FLOP) → §11.1/§11.2 "all configs clear onset." |
| `emergence-mirage-schaeffer-2304.15004.pdf` | **Are Emergent Abilities a Mirage?** — Schaeffer et al. 2023 | arXiv [2304.15004](https://arxiv.org/abs/2304.15004) | >92% of "emergence" is a metric artifact → §11.1/§11.2 "ceiling is a smooth gradient, not a cliff." |
| `deepseek-v3-2412.19437.pdf` | **DeepSeek-V3 Technical Report** — 2024 | arXiv [2412.19437](https://arxiv.org/abs/2412.19437) | 671B/37B (18×), 14.8T tok → ~400 tok/active anchor used throughout §6.1/§11.1; aux-loss-free load balancing (Lever C). |
| `llama-3-herd-2407.21783.pdf` | **The Llama 3 Herd of Models** — Meta 2024 | arXiv [2407.21783](https://arxiv.org/abs/2407.21783) | Dense over-training + FLOP anchor (405B / 15.6T ≈ 38×; ~3.8e25 FLOP). |

## Post-training & test-time compute (cached in `papers/`) — supports §11.2

| File | Title / what it is | Source | Why it matters to us |
|---|---|---|---|
| `deepseek-r1-2501.12948.pdf` | **DeepSeek-R1** — 2025 | arXiv [2501.12948](https://arxiv.org/abs/2501.12948) | Cleanest same-base RLVR A/B (V3→R1: AIME 39→80, MATH-500 90→97); §11.2 anchor; reasoning is distillable/portable. |
| `rl-reasoning-boundary-yue-2504.13837.pdf` | **Does RL Incentivize Reasoning Beyond the Base Model?** — Yue et al. 2025 | arXiv [2504.13837](https://arxiv.org/abs/2504.13837) | RLVR **elicits** (pass@1 → base pass@k ceiling), doesn't add → §11.2 "post-training is a ceiling-*reacher*; a strong base is the prerequisite." |
| `prorl-nvidia-2505.24864.pdf` | **ProRL** — NVIDIA 2025 | arXiv [2505.24864](https://arxiv.org/abs/2505.24864) | The contested **expansion** margin: prolonged RL can extend the boundary, scaling with base competence → §11.2. |
| `test-time-compute-snell-2408.03314.pdf` | **Scaling LLM Test-Time Compute Optimally** — Snell et al. 2024 | arXiv [2408.03314](https://arxiv.org/abs/2408.03314) | Inference-vs-pretrain compute tradeoff (`M + 3(D_pre/D_inf)(M−1)`); §11.2 test-time row + the 1B-context multiplier. |
| `weak-to-strong-burns-2312.09390.pdf` | **Weak-to-Strong Generalization** — Burns et al. 2023 | arXiv [2312.09390](https://arxiv.org/abs/2312.09390) | Naive RLHF scales **poorly** to stronger models → §11.2 alignment caveat. |

## Quantization (cached in `papers/`) — TurboQuant evaluation

| File | Title / what it is | Source | Why it matters to us |
|---|---|---|---|
| `turboquant-2504.19874.pdf` | **TurboQuant: Online VQ with Near-Optimal Distortion** — Zandieh et al. (Google Research / DeepMind), ICLR 2026 | arXiv [2504.19874](https://arxiv.org/abs/2504.19874) | Data-oblivious, **online**, bounds MSE *and* inner-product distortion (~2.7× off optimal); ~3.5-bit KV quality-neutral, ≥6× memory. Evaluated for our training stack: best fit = **read-only offloaded-expert fetch I/O** (~4.5× on the §11 Lever A disk→HBM term); KV-ring / routing-index / gradient codecs are **research bets** (the walls they attack are mostly already closed by sparse-ring pruning + MLA); inference KV-cache for *serving* the trained model is the free, no-regret win. |
| `qjl-2406.03482.pdf` | **QJL: 1-Bit Quantized JL Transform for KV Cache** — Zandieh et al. | arXiv [2406.03482](https://arxiv.org/abs/2406.03482) | The 1-bit JL residual TurboQuant builds on; cited in §11.3 as a routing/KV-index quant primitive (QJL-prescreen recall is *unmeasured* in the `gide` study — flagged as a gap). |
| `spinquant-2405.16406.pdf` | **SpinQuant: LLM Quantization with Learned Rotations** — Liu et al. (Meta), ICLR 2025 | arXiv [2405.16406](https://arxiv.org/abs/2405.16406) | Learned-rotation quantization; §11.3 contrasts data-free random rotation vs learned (OPQ/SpinQuant) for the routing/KV index. |
| `qes-2602.03120.pdf` | **Quantized Evolution Strategies** — Xu, Miikkulainen, Qiu | arXiv [2602.03120](https://arxiv.org/abs/2602.03120) | High-precision fine-tuning of quantized LLMs at low-precision cost; §11.3 reference for the routing-quant research item. |

## Compute-operation levers — beyond dense matmul (cached in `papers/`) — supports §11.4

Surveyed for "can we do better than dense FP matmul?" Verdict for the B300 training target: **none beats
dense FMA in wall-clock today** — each is inference-only, ~break-even at frontier quality, or needs new
silicon — and low-bit hurts *more* at our heavy token budget. The realizable lever stays tensor-core-native
**precision** (FP8 → NVFP4); co-designed hardware (ternary/LUT/analog) is the only path that flips it.

| File | Title / what it is | Source | Why it matters to us |
|---|---|---|---|
| `bitnet-b1.58-era-of-1bit-2402.17764.pdf` | **The Era of 1-bit LLMs (BitNet b1.58)** — Ma et al. 2024 | arXiv [2402.17764](https://arxiv.org/abs/2402.17764) | Ternary {−1,0,+1} weights turn the FFN multiply into a signed **add** (~71× per-op energy). But it's an inference weight format — TRAINING keeps FP master weights + grads, and B300 has no ternary datapath → §11.4 "research, not adopt." |
| `bitnet-b1.58-2b4t-2504.12285.pdf` | **BitNet b1.58 2B4T Technical Report** — 2025 | arXiv [2504.12285](https://arxiv.org/abs/2504.12285) | Largest *natively-trained* ternary parity datapoint (2B/4T, ~1 pt of Qwen2.5-1.5B) — and it tops out at 2B, ~4 orders below our target: the scale-extrapolation risk. |
| `matmul-free-lm-2406.02528.pdf` | **Scalable MatMul-free Language Modeling** — Zhu et al. 2024 | arXiv [2406.02528](https://arxiv.org/abs/2406.02528) | Whole-model matmul-free (ternary BitLinear + MLGRU mixer); 13 W FPGA @ 1B params. Win is memory/energy on FPGA/CPU, not a B300 training-FLOP cut; validated ≤2.7B. |
| `monarch-structured-matrices-2204.00595.pdf` | **Monarch: Expressive Structured Matrices** — Dao et al. 2022 | arXiv [2204.00595](https://arxiv.org/abs/2204.00595) | Butterfly-block factorization → `O(d^1.5)` FFN, GEMM-friendly (can hit tensor cores). Most plausible structured FFN lever → §11.4 "research." |
| `monarch-mixer-m2-2310.12109.pdf` | **Monarch Mixer (M2)** — Fu et al. 2023 | arXiv [2310.12109](https://arxiv.org/abs/2310.12109) | Sub-quadratic in seq *and* model dim. Source of the ~25.6% naive FLOP-util datapoint — structured ops under-fill MMA tiles, so FLOP-cut ≠ wall-clock. |
| `maddness-multiply-without-multiplying-2106.10860.pdf` | **Multiplying Matrices Without Multiplying (MADDNESS)** — Blalock & Guttag, ICML 2021 | arXiv [2106.10860](https://arxiv.org/abs/2106.10860) | PQ + LUT approximate matmul, no multiplies, 10–100× on CPU. LUT gathers strand tensor cores; inference/small-model only → §11.4 "track." |
| `low-bit-favors-undertrained-2411.17691.pdf` | **Low-Bit Quantization Favors Undertrained LLMs (scaling laws, 100T tokens)** — Ouyang et al. 2024 | arXiv [2411.17691](https://arxiv.org/abs/2411.17691) | **The decisive constraint:** quantization degradation *rises* with tokens/param. Our heavy over-training (100–500T tokens) is exactly where ternary / aggressive-FP4 hurt MOST → prefer FP8 over NVFP4/ternary. |

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

## Web sources (pointers — not cached)

Non-arXiv sources behind the §11.1 frontier-comparison caveat (no PDF to vendor; recorded as links).

| Source | URL | Why it matters to us |
|---|---|---|
| Epoch AI — frontier training-compute trend | [epoch.ai/data-insights/open-models-threshold](https://epoch.ai/data-insights/open-models-threshold) | Frontier crossed **~1e26 FLOP** in 2025 (Grok-3 first), scaling **~4.7×/yr** → §11.1 "22× GPT-4 is a stale, 2023-era baseline." |
| Anthropic — Claude Opus 4.8 (2026-05-28) | [anthropic.com/news/claude-opus-4-8](https://www.anthropic.com/news/claude-opus-4-8) | Discloses **no** parameter/compute figures → §11.1 "a direct compute comparison is not possible." |
| OpenAI GPT-5 / GPT-5.5 (2026; secondary coverage) | [o-mega.ai/articles/gpt-5-5-the-complete-guide-2026](https://o-mega.ai/articles/gpt-5-5-the-complete-guide-2026); [epoch.ai gradient-updates](https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45) | Keep the two facts distinct: the **~10×-less-pre-training-for-same-quality** result is **GPT-5** (via post-training; Epoch AI) — which is what §11.1/§11.2 attribute it to. **GPT-5.5** (2026-04-23) is the *first full retrain since GPT-4.5* and by definition *required* significant new pre-training, so it is **not** the ~10×-less model. No official specs for either → §11.1/§11.2 "train-FLOP is a weak capability proxy." |

## Provenance
- Papers downloaded from arXiv on 2026-05-30. arXiv IDs are stable; PDFs are the cited versions.
- **Second batch (2026-05-30):** 22 papers across scaling laws / data limits / capability (§6.1, §11.1),
  post-training & test-time compute (§11.2), and quantization (TurboQuant) — all validated as PDFs and the
  cited versions. The 2025-class items (Abnar 2501.12370, Ling 2507.17702, Morris 2505.24832, Nakamura
  2508.18672, Yue 2504.13837, ProRL 2505.24864, TurboQuant 2504.19874) were adversarially fact-checked in
  the workflows behind §6.1/§11.x, but several measure regimes **far below** 50T/500B (sparsity ≤50×, bases
  ≤~1T); treat their application to our scale as the *flagged extrapolations* the doc marks, not validated results.
- Repo dispositions are from the deep-research passes summarized in
  `docs/guides/unified-attention-architecture.md` §"Prior art". Repo maturity can drift —
  re-check before depending on a specific commit.
- All previously-pending items (DeepSeek V3.2 DSA, DeepSeek V4, MLA, MoBA, Muon variants,
  AdEMAMix) were fact-checked on 2026-05-30 and their references added above. Verdicts are in
  `docs/guides/unified-attention-architecture.md` §9. **DeepSeek V4** is a recent (~Apr 2026)
  *preview* — treat its specifics as time-sensitive. **DeepSeek V4** has no cached PDF here
  (preview tech report is on HuggingFace, not arXiv); see the model card linked in §9 discussion.
- **Third batch (2026-05-31):** 7 papers on compute-operation levers beyond dense matmul (BitNet b1.58 +
  2B4T, MatMul-free LM, Monarch + M2, MADDNESS, Low-Bit-Favors-Undertrained), supporting §11.4 — all
  validated as PDFs. Every one is small-model / inference / non-B300 evidence; the §11.4 verdict is that
  none beats dense FMA in B300 *training* wall-clock today, and (per 2411.17691) low-bit hurts *more* at
  our heavy token budget.
- **Fourth batch (2026-06-04):** 3 quantization-primitive papers cited by §11.3 — QJL (2406.03482),
  SpinQuant (2405.16406), QES (2602.03120) — added during the 2026-06-04 architecture-doc pressure-test
  (41 findings) so §11.3's evidence base is fully cached. Note §11.3 also leans on out-of-repo `gide`
  findings, which remain **unversioned and un-cacheable**; the doc flags that promotion as pending in-repo
  replication.
