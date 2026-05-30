> **⚠️ ARCHIVED (2026-05-30).** This is **v1** of the unified attention architecture, written
> *before* the prior-art research pass. It has been **superseded by v2**:
> `docs/guides/unified-attention-architecture.md`. v1 proposed hand-writing a Triton Flash
> kernel and inventing a content-adaptive sparse policy; v2 revises this after finding that
> FlexAttention + FlashAttention-3 and DeepSeek NSA already cover those layers (build-vs-buy).
> Kept for historical reference only.

---

# Unified Attention Architecture

> **Status:** design proposal / RFC. Forward-looking; not yet implemented beyond the
> foundational pieces noted under "What already exists."
> **Scope:** how to unify block-sparse, dilated, Flash, and ring attention into one correct,
> composable engine — and the math that says what it can and cannot guarantee.
> **Goal of the library:** reduce the training cost of transformer-based networks by making
> long-context attention sub-quadratic in compute and bounded in memory, without a forced
> quality ceiling.

---

## 1. Motivation

A correctness audit of this codebase (`docs/reports/correctness-audit-main-2026-05-30-0219-UTC.md`)
found that nearly every long-sequence attention variant — ring SDPA, the ring autograd
`Function`, the Triton kernels, and all four block-sparse classes — failed in the **same
place**: combining attention computed over *pieces* of the key space without a shared softmax
denominator (summing independently-normalized softmaxes, dividing by the wrong term, or
masking whole rows to `-inf`).

That is not a coincidence. It tells us what the foundation must be: **one numerically-correct
way to attend over a key set split into pieces, reused everywhere.** That operator is the
**online softmax** (Flash Attention's running-max `m`, running-denominator `ℓ`, rescaled
output `O`), and it is **associative and commutative** — which is the deep reason it unifies
all of these variants.

## 2. Core insight

| "Variant" | What the *pieces* are | Combination |
|---|---|---|
| Block-sparse | selected KV blocks | online softmax |
| Dilated (LongNet) | strided KV subsets | online softmax |
| Flash | KV tiles streamed through SRAM | online softmax **is** Flash |
| Ring / sequence-parallel | KV shards owned by different ranks | online softmax across the network |

So the design is **not** "make block-sparse and dilated talk to each other." It is:

> **Build one correct online-softmax engine, and make block-sparsity, dilation, causality, and
> ring all just *descriptors of which KV pieces a query block sees*.**

Because the combination operator is associative, the *same* engine scales from a single-GPU
sparse kernel to a multi-node ring with no change to the math — only a communication wrapper.

### 2.1 Reframing dilation as a block pattern

`BlockSparseDilatedAttention` conflated two mechanisms because it treated dilation as
*token-scatter inside a block*. Don't. **Express LongNet dilation as a multi-resolution
block-selection pattern**: at scale level `ℓ`, a query block attends to KV blocks at block
stride `2^ℓ` (near blocks densely, far blocks sparsely, different heads at different scales).
This gives dilation's logarithmic long-range reach at *block* granularity — hardware-friendly
(dense tile matmuls, no gather/scatter), and it drops straight into a block-sparse Flash
kernel. The two mechanisms collapse into one: **a pattern over blocks.**

## 3. The proposed variant: Multi-Scale Adaptive Block-Sparse Flash Attention

Combine the strengths and design out the weaknesses by feeding **one Flash core** the **union
of two block-selection policies**:

1. **Structured multi-scale skeleton** — local dense window + exponentially-strided coarse
   blocks. Cheap, deterministic; gives every token a full-sequence receptive field in
   `O(log n)` block hops. *(Dilation's strength: guaranteed coverage + long range.)*
2. **Content-adaptive top-k blocks** — mean-pool each KV block to a centroid, score
   query-block-centroid · key-block-centroid, take top-k. Recovers the *relevant* far tokens a
   fixed stride would skip. *(Block-sparse's strength: content flexibility — and it fixes
   dilation's biggest weakness, content-blindness.)*

```
active_blocks(q) = skeleton(q) ∪ top_k_routed(q)      # deduped
```

One Flash kernel consumes the index, loops each query block's active KV blocks, accumulates
with online softmax, and applies a per-block mask flag (causal diagonal triangular, skip
future). Causality, dilation, sparsity, adaptivity all live in the **index + a tiny mask
flag** — never in the combination math.

## 4. Layered architecture (the library)

The payoff is that every future variant becomes a *configuration*, not a new (buggy)
reimplementation:

```
L0  AttentionAccumulator     online-softmax merge (m, ℓ, O); associative; the ONLY place
                             softmax-combination lives. Pure, gradcheck-tested.
L1  Flash kernels (Triton)   fwd + bwd over (query_block, [KV blocks], mask_flags), fp32
                             accumulation, autotuned; eager fallback with identical results.
L2  SparsityPolicy           builds the BSR block index. Plugins: local_window, global,
                             dilated (= multi-scale strides), content_adaptive (top-k),
                             and unions.   <-- block-sparse & dilated UNIFY here
L3  ExecutionStrategy        single-GPU (loop) | ring/distributed (rotate KV shards, reduce
                             via L0). The sparse index prunes which shards to communicate.
L4  Module API               nn.Module wrappers, MAGNETO init, the factory.
```

Then: dilated = `Flash(policy=Dilated)`; block-sparse = `Flash(policy=LocalWindow|Adaptive)`;
**ring dilated** = `Flash(policy=Dilated, strategy=Ring)`; the new variant =
`Flash(policy=MultiScale ∪ Adaptive)`. **One correct engine, many configs.**

### 4.1 How Flash and Ring compose for free

- **Flash is not a separate class** — L1 *is* Flash. Every variant is Flash-backed by
  construction (no `n²` score materialization).
- **Ring** is L3 wrapping L1: each rank runs the same kernel over the KV it currently holds,
  accumulates `(O, m, ℓ)`, rotates KV via `isend/irecv`, and after the full ring the
  accumulator *is* the exact global softmax — no new math, no `all_gather`. And the block
  index tells ring **which KV shards a rank's queries actually need → skip unneeded shards** =
  sparse ring attention with reduced communication. Block-sparsity makes ring *cheaper*.

## 5. Strengths / weaknesses scorecard (candid)

| Variant | Strength kept | Weakness fixed | How |
|---|---|---|---|
| Block-sparse | hardware tiles, FLOP-skipping, flexible/adaptive | coarse all-or-nothing + broken cross-block softmax | selection policy over the Flash core |
| Dilated | cheap multi-scale, guaranteed long-range | content-blindness; gather/scatter traffic | block pattern (no scatter) ∪ adaptive top-k |
| Ring | O(n/p) memory, multi-node scale | `all_gather` blowups, wrong cross-chunk combine/causal | accumulator associativity; sparse comm pruning |

### 5.1 Genuine tensions the design does NOT erase

These are managed, not solved by elegance — treat them as first-class engineering:

1. **Load imbalance.** Sparse/causal patterns are irregular (different query blocks attend to
   different numbers of KV blocks); ring assumes uniform per-shard work → stragglers. Needs
   **workload-aware block→rank assignment**.
2. **Adaptivity vs. ring communication.** Top-k routing must *score* a KV block to select it,
   but in ring a query hasn't seen all KV yet. Mitigation (exchange tiny per-block centroids
   first, then prune the full rotation) works but adds a round and **softens** comm savings.
3. **Block granularity loses token-level dilation resolution.** Finest stride is one block
   (~64–128 tokens), not 1 token. Fine for long context; a fidelity reduction vs. true
   token-stride dilation.
4. **Enable ≠ guarantee.** The architecture *enables* cost reduction; the realized
   FLOP/quality tradeoff is empirical (§7).

## 6. What is mathematically determinable a priori

### 6.1 Compute — YES, exact

Standard attention costs `≈ 4·n²·d_model·L` FLOPs (the QKᵀ and A·V matmuls, all heads, all
layers). A pattern allowing a per-query budget of `W` keys costs `≈ 4·n·W·d_model·L`. Hence:

```
attention compute speedup  =  n / W            (exact, fixed by the pattern, a priori)
```

The realized (wall-clock) speedup is bounded below this by `MFU_sparse / MFU_dense` and the
routing overhead (`≈ n²·d / b²` to score block-pairs — quadratic in *blocks*; keep the router
hierarchical to stay sub-quadratic asymptotically).

A separate **linear term** (QKVO + FFN, the "2N" rule) `≈ 2·params·n` does *not* grow with
context. The two attention terms cross it at:

```
4·n²·d·L = 2·P·n   ⇒   n* = P / (2·d·L)
```

**Beyond `n*`, dense attention costs more than the entire rest of the model**, and keeps
doubling each time `n` doubles. For the 7B config below, `n* ≈ 26,700`.

### 6.2 Memory — YES, exact

- **Non-Flash scores:** `n²·heads·bytes` *per layer* — the wall. Flash removes it (running
  state is `O(n)`).
- **KV working set:** `2·n·d_model·bytes·L` total — `O(n)`. **Sparsity does not reduce this**
  (training keeps all K,V; adaptive routing may select any block). Ring shards it to `O(n/p)`
  per device.

So peak attention memory is a closed form of (Flash on/off, ring `p`, block size, dtype).

### 6.3 Quality — NO tight a-priori number; YES structural + conditional guarantees

This cannot be a tight a-priori floor, for reasons intrinsic to deep learning, not to this
design:

- **Expressivity (provable, qualitative).** Universal-approximation results for sparse
  attention show that a **connected** pattern — every token reaches every other in a bounded
  number of hops — retains the full representational class of dense attention. Our multi-scale
  skeleton gives `O(log n)` reachability, so **no representational ceiling is imposed**;
  degradation is not *structurally forced*. (Says "expressible," not "loss ≤ Z".)
- **Approximation error (conditional).** For a fixed model, dropping keys causes per-token
  error bounded by the **dropped softmax mass** `δ_q = Σ_{j∉S} p_qj`:

  ```
  ‖o_q − õ_q‖ ≤ 2·δ_q·max_j‖v_j‖
  ```

  `δ` depends on learned weights + data, so it is boundable only *under a concentration
  assumption* (attention mass on ≤ m keys/query). The adaptive top-k is precisely what makes
  that assumption hold in practice.
- **End-task loss (not provable a priori).** The model *trains with* the pattern — it learns a
  constrained function, not an approximation of dense attention; and Lipschitz error
  propagation through many layers is mathematically valid but vacuously loose. So loss parity
  must be **measured**, not assumed.
- **Runtime certificate (achievable).** The block-centroid routing scores upper-bound the max
  score of every *unselected* block → an upper bound on `δ̂` per forward. Each step can emit:
  *"dropped ≤ δ̂ of the attention mass."* Not a static guarantee, but an auditable quality
  signal (and an early warning that a pattern is too aggressive for an input distribution).

## 7. Cost model — worked example

Reproduce with `python analysis/attention_cost_analysis.py` (pass flags to size your own
config). Config: 7B-class — `d_model=4096`, `32` layers, `32` heads, bf16; sparse budget
`W=4096` attended keys/query (context-independent); one H100 (≈989 TFLOP/s peak, 50% dense /
40% sparse MFU, 80 GB).

```
   context n |  dense attn | sparse attn | attn x | model x | KV (all L) | scores/L* | ring p
       8,192 | 35.18 TFLOP | 17.59 TFLOP |     2x |    1.1x |     4.0 GB |    4.0 GB |     1
      32,768 |   563 TFLOP | 70.37 TFLOP |     8x |    1.9x |    16.0 GB |   64.0 GB |     1
     131,072 |  9.01 PFLOP |   281 TFLOP |    32x |    5.1x |    64.0 GB |    1.0 TB |     1
   1,048,576 |   576 PFLOP |  2.25 PFLOP |   256x |   34.9x |   512.0 GB |   64.0 TB |     8
   * scores/L = non-Flash score matrix per layer (the wall Flash removes).
```

**Wall-clock at 1M tokens (attention only, one forward, this GPU):** dense ≈ **1166 s** (and
you must first shard 512 GB of KV) vs sparse ≈ **5.7 s**. A training step is ~3× a forward
(fwd + 2× bwd), over many steps — so dense 1M-context training is simply not viable.

### Reading the table

1. **Sparsity's payoff scales with context** — 2× at 8K → 256× (attention) at 1M. Below
   ~16K, don't bother; the win is entirely in the long-context regime.
2. **The crossover (~27K) is the headline.** Past it, dense attention is the *majority* of
   forward FLOPs (97% at 1M); the sparse pattern drops it to ~13%, restoring near-linear
   scaling and a **~35× whole-model forward speedup** at 1M.
3. **Flash is mandatory, not optional** — non-Flash scores are 64 GB/layer at 32K and 1 TB/
   layer at 128K. Flash is a *memory enabler*, not a speedup.
4. **Compute and memory are solved by different members of the trio:** **sparse → compute**
   (the `n²` FLOPs), **ring → memory** (the `O(n)` KV storage), **Flash → prerequisite**.
   Drop any one and you hit a different wall — which is exactly why the unified design wants
   all three composable.

## 8. Phased build plan (with validation gates)

Invert the mistake the audit found (kernels optimized before they were correct): **correct
eager reference first, kernel second, distribution last.** Each phase has a hard gate.

| Phase | Deliverable | Gate (must pass to proceed) |
|---|---|---|
| **0. Core** | `AttentionAccumulator` (L0) — promote `_merge_block_attention` to a standalone, documented module | property tests: associativity, equals one joint softmax, masked-partial safe, fp16/fp32 |
| **1. Eager Flash reference** | L1 reference in pure PyTorch: per-query-block loop over KV blocks via L0 (the merged `BlockSparseAttention` grouped path is the template) | `allclose` to a dense masked-softmax ref ≤ 1e-10 across patterns × causal × dtype; **`gradcheck`-passing backward** (float64) |
| **2. Policies** | L2 `SparsityPolicy` interface + plugins: `local_window`, `global`, `dilated` (block-stride), `content_adaptive` (top-k), unions | each policy's index → identical output to a dense reference restricted to the same allowed set; routing accuracy check |
| **3. Triton kernel** | L1 Triton fwd + bwd consuming the BSR index, fp32 accum, autotuned; FP8/Hopper later | `allclose` + `gradcheck` vs the Phase-1 eager reference *before* any autotuning; MFU floor on target GPU |
| **4. Ring strategy** | L3 ring/sequence-parallel wrapper using L0 to reduce shard partials; sparse comm pruning; global-position causal masking | multi-GPU (`torchrun`) parity vs single-GPU; comm-volume + load-balance measured |
| **5. Consolidation** | route the existing variants through the engine; deprecate/fold the broken standalone classes | benchmark-suite parity (tokens/s, peak mem/GPU, loss-curve parity on a small LM) |

### 8.1 What already exists (foundation merged to `main`)
- `_merge_block_attention` (the L0 online-softmax primitive) + the correct grouped
  block-sparse path in `BlockSparseAttention` (PR #29) — the Phase-0/1 seed.
- `RingCommunicationMixin` buffer-aliasing fix (PR #27) — correct ring K/V exchange for L3.
- The audit + cost calculator (`analysis/attention_cost_analysis.py`) — the justification and
  the sizing tool.

### 8.2 Disposition of the current classes
- `BlockSparseAttention` (+ Hilbert/Adaptive subclasses, Multihead wrapper) → become
  `Flash(policy=…)` configurations.
- `BlockSparseDilatedAttention` → **deprecate / fold in** as `Flash(policy=MultiScale)`
  (its within-block token dilation reaches only intra-block distances and adds little over a
  dilated *block* pattern — see §2.1; the standalone class is not worth a bespoke fix).
- Ring variants → `Flash(strategy=Ring, policy=…)`; retire the bug-prone bespoke paths.

## 9. Acceptance metrics (every new variant must meet)

- **Correctness:** `allclose` ≤ 1e-10 (fp64) to a dense masked reference; `gradcheck` passes;
  multi-GPU parity vs single-GPU.
- **Compute:** realized attention speedup ≥ target fraction of the `n/W` ceiling; MFU ≥ floor.
- **Memory:** peak/GPU within the closed-form budget for the chosen Flash/ring/block config.
- **Quality:** loss-curve parity vs dense within tolerance on a small LM, gated by the runtime
  dropped-mass certificate `δ̂ ≤` threshold.

## 10. Risks & open questions

- A correct *and* fast Triton block-sparse Flash **backward** is the hard part (where the
  repo's autograd bugs lived) — budget for it explicitly and gate on `gradcheck`.
- Content-adaptive top-k is non-differentiable — start with the fixed multi-scale skeleton;
  add adaptivity via straight-through / learned-but-fixed centroids once the core is proven.
- Routing must stay hierarchical to keep the whole thing sub-quadratic asymptotically.
- Backward must cache the selected index so gradients match the forward selection.
- The sparsity↔quality tradeoff (the `W` budget) is empirical and model-dependent — the
  calculator sizes compute/memory; only training pins quality.

---

*Cost model and all figures in §7 are produced by `analysis/attention_cost_analysis.py`.*
