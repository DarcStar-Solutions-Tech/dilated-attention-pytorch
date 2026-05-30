# Correctness Audit Report — dilated-attention-pytorch Core Library

> **Scope:** `src/` core library on `main` — correctness only (no tests, benchmarks, or performance).
> **Method:** 6 domain lenses (numerical, ring-distributed, autograd, memory, kernels, sparse/dead-code) × adversarial verification (2 independent skeptics per finding) via the `audit-scoped-1m.workflow.js` 1M-context harness.
> **Run:** `wf_ddbd9190-3f5` · 84 agents · ~4.07M tokens · 2026-05-29.
> **Result:** 31 confirmed + 4 contested findings — 15 critical / 13 high / 6 medium / 1 low (a few near-duplicates merged in the prose below).
> 🤖 Generated with [Claude Code](https://claude.com/claude-code).


## 1. Executive Summary

This scoped audit surfaced **30 findings that survived adversarial verification**: **29 confirmed** (both skeptics agreed) and **1 contested** (split decision); two further findings were split on severity but unanimous on reality. By severity the confirmed set breaks down as **13 critical, 10 high, 4 medium, 3 low** (using the higher of the two reviewer severities where they diverged). The most pervasive and serious class of defect is **incorrect cross-block / cross-chunk softmax accumulation**: at least six independent implementations (ring SDPA, ring autograd, the three Triton/Hilbert kernels, and all four block-sparse variants) either sum independently-normalized softmaxes or divide by the wrong denominator, silently producing mathematically meaningless attention outputs. The **single most serious issue** is the family of K/V ring-buffer aliasing bugs in `RingCommunicationMixin` (`ring/base/ring_communication_mixin.py`), because it sits on the **default, production-recommended** path (`StandardRingAttention`, aliased to `RingDilatedAttention`/`RingDilatedAttentionProduction`) and silently corrupts every multi-GPU forward/backward via both a shared recv buffer (K and V alias one tensor) and concurrent in-place `isend`/`irecv` on the same memory.

---

## 2. Findings by Severity

### CRITICAL

#### C1. Ring K/V share one cached recv buffer (buffer aliasing) — *default production path*
- **File:** `src/dilated_attention_pytorch/ring/base/ring_communication_mixin.py:113-114, 159-183`
- **What's wrong:** `_get_comm_buffer` keys its cache on `(shape, dtype, buffer_type)` only — not on the logical tensor (K vs V) nor tag. K and V are communicated back-to-back with identical shape/dtype, so both `ring_pass_forward(current_k)` and `ring_pass_forward(current_v)` receive into the **same** tensor object. After both calls `current_k` and `current_v` alias one buffer holding the neighbor's V; K is lost.
- **Why it matters:** Manifests at `world_size >= 2` on `StandardRingAttention`, `HilbertRingAttention`, `BlockSparseRingAttention`, `DistributedRingAttention` — all exported and factory-registered (`ring`/`ring_standard`/`ring_distributed`/`ring_hilbert`), with `StandardRingAttention` aliased to the recommended `RingDilatedAttentionProduction`. Next ring step computes attention with `K == V`; output is numerically wrong on every multi-rank run.
- **Fix:** Include tag (or a caller-supplied buffer id) in the cache key, or allocate `torch.empty_like(tensor)` per call, or pass K/V together through one `batch_isend_irecv` with separate buffers (as `ring_pass_kv_fixed` already does).

#### C2. Ring recv buffer reused across steps → send-into-recv aliasing (in-place P2P hazard)
- **File:** `src/dilated_attention_pytorch/ring/base/ring_communication_mixin.py:117-143, 175-183`
- **What's wrong:** Because `_get_comm_buffer` returns the same buffer `B` every call, from ring step 1 onward the tensor being sent **is** the buffer being received into: `send_tensor = tensor.contiguous()` returns `B` (already contiguous) and `recv_buffer = _get_comm_buffer(...)` returns the same `B`. `dist.isend(B)` and `dist.irecv(B)` then operate on identical memory concurrently — undefined behavior.
- **Why it matters:** Triggers at `world_size >= 3` (first reuse at step 1) on the same four exported classes as C1, independent of the K/V-sharing bug. NCCL/Gloo give no read/write ordering guarantee on overlapping send/recv memory; forwarded and/or received chunks are corrupted.
- **Fix:** Never receive into a buffer currently being sent — double-buffer or allocate `torch.empty_like(send_tensor)` per step and assign `current_k = received_buffer`.
- **Note:** C1 and C2 are two facets of the same root cause (`_get_comm_buffer` returning a shared, key-collapsed buffer). Fixing the buffer keying + per-call allocation resolves both.

#### C3. `ring_communication_fix.py` (the lucidrains fix) is dead code; live path is unpatched
- **File:** `src/dilated_attention_pytorch/ring/utils/ring_communication_fix.py:1-181`
- **What's wrong:** This module implements exactly the CLAUDE.md-mandated multi-GPU pattern (contiguity, `batch_isend_irecv`, wait-all, `barrier`) and `ring_pass_kv_fixed` (lines 92-149) uses **separate** `k_recv`/`v_recv` buffers — which would avoid C1/C2. A grep across `src/` finds **zero** importers; only `benchmarks/` reference a different function. The live `RingCommunicationMixin._ring_communication_with_retry` uses bare `isend`/`irecv`, no barrier, no batching, shared recv buffer.
- **Why it matters:** The correct fix exists but is applied nowhere it's needed; the buggy mixin (C1/C2) is the only reachable path for all four standardized ring impls.
- **Fix:** Fold the separate-buffer + `batch_isend_irecv` pattern into `RingCommunicationMixin`, or have the mixin call `ring_pass_kv_fixed` for paired K/V passes; then delete the orphaned module.

#### C4. `RingDilatedAttentionCorrect`: causal mask dropped for dilated offsets > 0 (future-key leak)
- **File:** `src/dilated_attention_pytorch/ring/base/ring_dilated_attention_correct.py:389-435` (mask guard at line 419)
- **What's wrong:** In `_compute_dilated_segment` (dilation_rate > 1), the loop runs `offset` over `range(dilation_rate)` but the causal mask is gated by `if is_causal and offset == 0`. For every `offset >= 1`, scores are left unmasked, so those dilated query positions attend to future keys.
- **Why it matters:** Reachable single-GPU via `forward(is_causal=True)` for any segment with `dilation_rate > 1` (the normal dilated mode, e.g. `[1,2,4]`). Registered as `ring_correct` and exported top-level. Empirically verified: perturbing a future-position value changed past-query output by ~89; the offset-1 substream leaks future information for `(dilation_rate-1)/dilation_rate` of positions. The identical buggy guard is copied at `ring/hilbert/ring_dilated_attention_hilbert_gpu_optimized.py:478,488,720`.
- **Fix:** Remove the `and offset == 0` guard — compute `q_pos`/`k_pos` from per-offset indices and apply `scores.masked_fill_(q_pos.unsqueeze(1) < k_pos.unsqueeze(0), -inf)` for every offset.
- **Note:** Reported twice (lines `400-433` and `389-435`); same defect, merged here.

#### C5. `RingDilatedAttentionSDPA`: ring outputs averaged instead of softmax-renormalized
- **File:** `src/dilated_attention_pytorch/ring/base/ring_dilated_attention_sdpa.py:253-322` (accumulate 312, divide 322)
- **What's wrong:** `_ring_sdpa_forward` calls `F.scaled_dot_product_attention` independently per KV block (each block self-normalized), accumulates `output_accum += attn_output`, and returns `output_accum / self.world_size`. This computes `mean_j(O_j)`, not the joint softmax over all keys. No online-softmax/LSE rescaling; an LSE buffer is allocated then discarded (line 272, assigned to `_`); the code comment admits "Simple averaging". For causal, future chunks are zeroed but the divisor stays `world_size`, extra down-weighting valid chunks.
- **Why it matters:** Triggers on every `world_size > 1` run (the class's sole purpose). Registered `ring_sdpa`, exported top-level. Numerically verified: ~36-50% relative error vs concatenated-key softmax.
- **Fix:** Use an LSE-returning SDPA path (or explicit scores) with online-softmax rescaling (running max + denominator), and remove the `/ world_size` average; skip future chunks entirely for causal.
- **Note:** Reported twice (lines `289-322` and `253-322`); same defect, merged.

#### C6. `RingAttentionFunction` forward divides output by `lse` instead of `exp(lse)` — wrong even single-GPU
- **File:** `src/dilated_attention_pytorch/ring/utils/ring_attention_autograd.py:119-129`
- **What's wrong:** Forward builds numerator `sum(exp(scores - max) * v)` and running log-sum-exp `lse`, then normalizes `output = output / lse.unsqueeze(-1).clamp(min=1e-6)`. The correct denominator is `exp(lse - scores_max)` (the sum-of-exponentials), not the scalar `lse` log-value. Output is off by `exp(lse - max)/lse`. The `clamp(min=1e-6)` also masks the `lse <= 0` case (catastrophic blow-up).
- **Why it matters:** `ring_attention`/`RingAttentionFunction` are exported public API, importable and wrong on a single GPU (`world_size=1`). Verified: max-abs error ~0.50 vs reference; with all-negative scores output explodes to ~1e6. (No internal production class currently calls it — see dead-code note.)
- **Fix:** Track a running denominator `denom = sum(exp(scores - running_max))` and divide by it; remove the clamp.

#### C7. `RingAttentionFunction` backward inconsistent with forward (gradcheck fails single-GPU)
- **File:** `src/dilated_attention_pytorch/ring/utils/ring_attention_autograd.py:146-247`
- **What's wrong:** Backward ignores the saved `output`/`lse` (unpacked line 157, unused) and recomputes a plain `attn_weights = F.softmax(scores)` (line 207) with the textbook softmax-attention VJP — i.e. the gradient of a **different** function than the (already-wrong, C6) forward. `torch.autograd.gradcheck` fails in the simplest single-GPU float64 config (O(1) Jacobian disagreement).
- **Why it matters:** Exported public API; the custom Function exists specifically for gradients. Any user importing it gets silently wrong gradients.
- **Fix:** Make forward/backward consistent — either replace the local-block forward with standard differentiable attention (let autograd handle backward), or rewrite backward as the exact analytic gradient of the corrected forward; add a `world_size=1` gradcheck test.

#### C8. `RingAttentionFunction` double-scales q (backward uses `q*scale^2`)
- **File:** `src/dilated_attention_pytorch/ring/utils/ring_attention_autograd.py:69-85, 178, 223`
- **What's wrong:** Forward line 69 sets `q = q * scale` and saves the already-scaled `q` (line 85). Backward line 178 does `q = q * scale` again, so recomputed scores use `q_orig * scale^2`, yielding a flatter (wrong) softmax; `grad_q` (line 223) multiplies by `scale` yet again.
- **Why it matters:** Exported public API; default `scale = 1/sqrt(d)` is non-trivial so nothing neutralizes the double-application. Verified gradient relative errors of 61-89%. Compounds C7.
- **Fix:** Save **unscaled** `q` and scale exactly once (in forward or backward, one convention); ensure backward scores equal forward scores and the `grad_q` chain-rule `scale` factor is applied once.
- **Note:** C6, C7, C8 are three distinct defects in the same Function; all three (plus the contested C-dropout below) must be fixed together to make it usable.

#### C9. Triton Hilbert/standard kernels: softmax accumulation is not attention
- **File:** `src/dilated_attention_pytorch/kernels/hilbert_attention_core.py:127-142, 249-258`
- **What's wrong:** Inside the per-key loop, `scores = tl.sum(q * k[None,:], axis=1)` is `[BLOCK_M]` (one score per query for one key); then `scores_max = tl.max(scores, axis=0)` reduces over the **query** axis (wrong — must be per-query over keys). The accumulators `acc`/`norm` are never rescaled by `exp(prev_max - new_max)` across iterations. `out = acc / norm` is not `softmax(QK^T)V`.
- **Why it matters:** Both kernels affected; `HilbertAttentionCore` is exported and wired via `HilbertAttentionMixin`. Verified max-abs diff 0.467 vs dense reference (error scales with score magnitude — algorithmic, not numerical). Forward output and the (correct) custom backward are mutually inconsistent.
- **Fix:** Implement a proper online softmax with per-query running max `m_i` and denom `l_i`, `[BLOCK_M, BLOCK_N]` score tiles, `correction = exp(m_i - m_new)` rescaling of `acc`/`l_i`, finalize `out = acc / l_i[:,None]` (requires the shape fix in H1 below).

#### C10. Triton kernel custom backward differentiates a different function than forward
- **File:** `src/dilated_attention_pytorch/kernels/hilbert_attention_core.py:270-442`
- **What's wrong:** `HilbertAttentionFunction.forward` uses the broken kernel (C9); `backward` re-derives gradients from a clean PyTorch full-softmax over Hilbert-reordered segments (`attn = F.softmax(...)`). The returned gradients do not correspond to the kernel's actual forward output, breaking the autograd contract. (One reviewer additionally found backward treats `dout` as already in reordered space while forward output is in original order — a further mismatch.)
- **Why it matters:** Default training path triggers it — `use_custom_backward=True` (line 501) and the guard at line 554 (`use_hilbert and use_custom_backward and self.training`). Used by `HilbertAttentionMixin` and `HilbertAttentionTritonWrapper`; exported in `kernels/__init__.py`.
- **Fix:** Make forward/backward consistent — either replace the forward kernel with the reordered PyTorch attention the backward already assumes (let autograd handle it), or mirror the corrected kernel in backward exactly (same dilation/masking, padded-key exclusion). Until then, do not use the custom Function for training.

#### C11. `BlockSparseAttention` sums independent per-block softmaxes
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_attention.py:344-375, 413-430`
- **What's wrong:** Both batched (`_compute_sparse_attention_batched`, `output_blocks[:, row_idx] += block_outputs[:, i]` line 375) and sequential (`output[:, row_start:row_end] += block_output` line 430) paths compute a separate softmax per `(row_block, col_block)` pair and **sum** them. `softmax(S1 ∪ S2)` ≠ `softmax(S1)·V1 + softmax(S2)·V2`; summing N normalized softmaxes overcounts mass ~N.
- **Why it matters:** Every shipped pattern (local_window, dilated_sparse, global_local) produces multiple key blocks per query block — including the default. Top-level export, factory `base`, and base class for adaptive/Hilbert/multihead/dilated variants. Verified max-abs diff up to ~11 vs joint-softmax reference.
- **Fix:** Group active pairs by query block and run one softmax over the concatenation of attended key blocks, or accumulate via online-softmax (running max `m`, denom `l`, rescale by `exp(m_old - m_new)`, divide by `l` at end). Replace the `+=` of normalized outputs.

#### C12. `BlockSparseAttention` sequential causal path leaks future tokens
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_attention.py:417-421, 474-478`
- **What's wrong:** The causal mask is gated by `if is_causal and row_idx >= col_idx`. Above-diagonal pairs (`row_idx < col_idx`) get no mask and reach the unmasked matmul+softmax, so queries attend to strictly-future keys. Same guard in `_compute_sparse_attention_with_weights` (used by `BlockSparseMultiheadAttention` when `need_weights=True`).
- **Why it matters:** Reachable single-GPU via sequential dispatch (`<=16` active pairs) and **unconditionally** via the weighted path. Verified: perturbing a future block changed earlier-query output by exactly 100.0. The helper `_get_causal_mask_for_block` would handle this correctly but is short-circuited by the guard.
- **Fix:** Remove the `and row_idx >= col_idx` condition (both sites): `continue` when `row<col`, `triu(diagonal=1)` on the diagonal, no mask when `row>col`.

#### C13. `BlockSparseAttention` batched causal path produces NaN
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_attention.py:348-375, 605-644`
- **What's wrong:** `_get_batched_causal_mask` sets the mask all-True for above-diagonal pairs (lines 635-637); `scores.masked_fill(mask, -inf)` makes whole rows `-inf`; `F.softmax` over an all-`-inf` row returns NaN; the scatter loop (line 375) adds NaN into output, poisoning every query block with any above-diagonal attended block.
- **Why it matters:** This is the **default** path (`enable_batched_ops=True`, `>16` pairs). Verified: default config (block_size=64, seq_len=2048, causal) → NaN at 31744/32768 positions; `BlockSparseMultiheadAttention` (default `need_weights=False`) routes here. *(One reviewer rated this high since `is_causal` defaults to False; the other critical. Marked critical: any ordinary causal use is fully broken.)*
- **Fix:** Drop fully-masked above-diagonal blocks from the active-pair list before batching when `is_causal`; or `torch.nan_to_num` the softmax output. Combine with the joint-softmax fix (C11).

#### C14. `BlockSparseDilatedAttention` — same sum-of-independent-softmaxes bug
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_dilated_attention.py:147-176, 220-235`
- **What's wrong:** `forward()` loops `(row_idx, col_idx)` pairs, runs an independent softmax per pair (`_apply_dilated_attention_to_block`, line 229) and accumulates `output[:, q_start:q_end] += block_output` (line 176). Same defect as C11. (Causal masking here is correct at block level — skips `row<col`, triu on diagonal — so only the joint-softmax bug applies.)
- **Why it matters:** Top-level export, factory variant `dilated`; default local_window pattern gives multiple key blocks per query block. Verified max-abs diff ~2.5 vs joint-softmax reference.
- **Fix:** Group active pairs by query block and run a single softmax over concatenated attended key blocks, or use online-softmax accumulation.

#### C15. `BlockSparseRingDistributedDilatedAttention` — sum-of-softmaxes + wrong per-block causal mask
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_ring_distributed_dilated_attention.py:440-503`
- **What's wrong:** (a) `_standard_sparse_attention` accumulates `output[:, q_start:q_end, h, :] += block_output` (line 459) over independent per-pair softmaxes (line 496) — same joint-softmax error. (b) `_compute_block_attention` applies `triu(diagonal=1)` to **every** pair (lines 488-493) using only in-block indices, never receiving `q_block_idx`/`k_block_idx`, so it's correct only for diagonal pairs: below-diagonal pairs wrongly hide valid past keys; above-diagonal pairs leak future keys.
- **Why it matters:** Top-level export, factory `distributed`; `_standard_sparse_attention` is the single-GPU fallback (and async-path delegate), reachable without a distributed launch — `local`/`global` levels run at `world_size=1`. *(Both reviewers confirmed both bugs; one downgraded to high noting it's a specialized enterprise variant. One correction: above-diagonal pairs do **not** NaN here — each row keeps its own diagonal key — the consequence is silent future-token leakage, not NaN.)*
- **Fix:** Accumulate per query block with online-softmax/LSE rescaling; pass `q_block_idx`/`k_block_idx` into `_compute_block_attention` and branch (skip above-diagonal, triu on diagonal, no mask below-diagonal).

---

### HIGH

#### H1. Triton kernels fail to compile when `head_dim != BLOCK_M`
- **File:** `src/dilated_attention_pytorch/kernels/hilbert_attention_core.py:92-125, 219-247`
- **What's wrong:** `seg_idx = offs_m // segment_size` is `[BLOCK_M]`, so `key_pos`/`key_hilbert` are `[BLOCK_M]` vectors (not scalars). K/V pointer math `key_hilbert * stride_kn + offs_d * stride_kd` adds `[BLOCK_M]` to `[BLOCK_D]`; Triton cannot broadcast unless `BLOCK_M == BLOCK_D`. The wrapper sets `BLOCK_M = min(64, M_padded)`, `BLOCK_D = min(64, head_dim)`, so any `head_dim != 64` (e.g. 32) or seq_len < 64 fails to compile.
- **Why it matters:** Verified `CompilationError: Cannot make_shape_compatible: incompatible dimensions at index 0: 64 and 32`. The Triton path is dead for the majority of realistic head dims; the only test that exercises head_dim=32 swallows the exception. *(One reviewer critical, one high; marked high — experimental `kernels/` module, opt-in, default head_dim=64 happens to compile.)*
- **Fix:** Compute a scalar `seg_start = (pid_m * BLOCK_M // segment_size) * segment_size`, make `key_pos` a scalar, load K/V as `[BLOCK_D]` (or a `[BLOCK_N, BLOCK_D]` tile) — co-required with the C9 rewrite.

#### H2. Triton forward kernel treats padded key positions as valid keys
- **File:** `src/dilated_attention_pytorch/kernels/hilbert_attention_core.py:101-102, 228, 294-319, 538-548`
- **What's wrong:** `forward` pads M up to a multiple of `segment_size` and passes `M_padded` as the kernel's `M`. The key mask `mask_k = (key_pos < M) & (key_pos < seg_end)` then treats padded indices `[M_orig, M_padded)` as valid; their zero K/V contribute `exp(0)` mass to the softmax denominator for real queries near the boundary. `M_orig` is saved to `ctx` but never used for masking.
- **Why it matters:** **Contested on reachability.** One reviewer marked it not-real (the kernel doesn't compile — H1 — and the path is unreachable via production `use_hilbert_core=False`); the other reproduced it directly on `head_dim=64` (which compiles), measuring max diff 0.54 at the padded tail and disputing the "masked by softmax bug" caveat. Treated as **high but conditional on H1 being fixed and the kernel being invoked directly** (`head_dim=64`).
- **Fix:** Pass `M_orig` as a separate kernel arg and mask keys with `key_pos < M_orig`.

#### H3. Triton backward dilation mask asymmetric, mismatches forward key set
- **File:** `src/dilated_attention_pytorch/kernels/hilbert_attention_core.py:378-405`
- **What's wrong:** For `dilation_rate > 1`, backward sets `mask[:, active_positions] = True` **and** `mask[active_positions, :] = True` (line 387) — a union: active query rows attend to all keys, others only to active keys. The forward iterates keys at stride `dilation_rate` (every query → dilated keys only). Masks differ (verified 16 differing cells for seg_len=8, d=2), so `dq/dk/dv` use a different connectivity than the forward.
- **Why it matters:** Default training path (`use_custom_backward=True`, training). Silent wrong gradients for `dilation_rate > 1`; the only test checks only finiteness. *(One reviewer medium, one high; experimental kernel + the forward is independently broken.)*
- **Fix:** Use only `mask[:, active_positions] = True` (drop the row line); derive forward and backward masks from one shared helper; exclude padded keys in both.

#### H4. `StandardRingAttention` causal mask ignores remote chunk global position
- **File:** `src/dilated_attention_pytorch/ring/standard_ring_attention.py:144-158, 270-298`
- **What's wrong:** `_local_attention` builds `triu(diagonal=1)` over `(q_len, kv_len)`, assuming Q and K both start at index 0, and applies it on **every** ring step. At step `s>0`, `current_k`/`current_v` hold a chunk from rank `(rank - s) % world_size` occupying a different global range. No `q_start`/`k_start`/rank is threaded in (they're computed in `forward` but never passed). A fully-past chunk gets wrongly masked; a fully-future chunk partially leaks.
- **Why it matters:** Triggers at `world_size >= 2` and `is_causal=True` on the **default** `StandardRingAttention` (aliased `RingDilatedAttention`/`RingDilatedAttentionProduction`, factory `auto` default). The sibling `RingDilatedAttentionCorrect` masks via global positions, proving the standardized impl is wrong. Duplicated verbatim in `hilbert_ring_attention.py` (~239-248), `block_sparse_ring_attention.py` (~211-220), `distributed_ring_attention.py` (310-319, 445-472) via the shared `create_causal_mask` helper.
- **Fix:** Thread source-rank / global offsets into `_local_attention`; build the mask from global positions (`q_pos = q_start + arange(q_len)`, `k_pos = k_start + arange(kv_len)`, mask `k_pos > q_pos`); skip wholly-future chunks. Apply to all four standardized ring impls.
- **Note:** Reported twice (lines `144-158, 270-298` and `106-178, 270-297`); same defect.

#### H5. `DistributedRingAttention` silently returns zeros on comm failure
- **File:** `src/dilated_attention_pytorch/ring/distributed_ring_attention.py:253-266`
- **What's wrong:** With `config.enable_error_recovery=True` (the **default**, `ring_config.py:19`), a failed ring communication is swallowed and `torch.zeros_like(tensor)` is returned as the "received" K/V chunk (line 266). The zero block is fed into `_local_attention` and LSE-accumulated. A zero K gives finite nonzero `lse = log(kv_len)`, inflating the accumulation denominator and silently attenuating the legitimate local output. Other ranks remain blocked → desync/hang.
- **Why it matters:** On-by-default, exported "enterprise" class (`ring_distributed`), reachable via `create_ring_attention("distributed")`. Transient NCCL faults yield silently corrupted (not erroring) outputs.
- **Fix:** Do not fabricate data — re-raise unconditionally (ring comm can't be locally recovered), or implement genuine collective-aware recovery. At minimum gate the zeros fallback behind an explicit off-by-default flag and log loudly.

#### H6. `RingDilatedAttentionSDPA` dilation indexing ignores segment boundaries; multi-segment outputs averaged
- **File:** `src/dilated_attention_pytorch/ring/base/ring_dilated_attention_sdpa.py:162-243`
- **What's wrong:** The dilated pattern is built once over the whole local sequence (`positions = arange(0, min(seg_len, max_positions)*dilation, dilation)`) — no per-segment reshape — so it attends across segment boundaries LongNet isolates and drops tokens. Multiple `(seg_len, dilation)` groups are combined via `torch.stack(...).mean(dim=0)` (line 243), blending scales that should be head-disjoint instead of routing to disjoint head groups.
- **Why it matters:** Exported, factory `ring_sdpa`. Computes a different function than the documented `DilatedAttention` for the same `(segment_lengths, dilation_rates)`; verified divergence vs reference (token@8 in segment 2 leaks to token@0 in segment 1).
- **Fix:** Reshape into `[b, n//seg_len, seg_len, h, d]`, subsample dilated indices within each segment, run attention per segment, assign each `(segment, dilation)` group to its own head range (mirror `base/dilated_attention.py`).

#### H7. `HeadParallelDilatedAttentionOptimized` crashes at construction (wrong kwargs)
- **File:** `src/dilated_attention_pytorch/base/head_parallel_dilated_attention_optimized.py:91-121`
- **What's wrong:** (1) `_create_attention_impl` passes `use_xformers`/`use_flex_attention`/`use_memory_pool`/`use_pattern_cache` to `ImprovedDilatedAttention`, which forwards them into the fixed-field `DilatedAttentionConfig` → `TypeError: unexpected keyword argument 'use_xformers'`; only `ImportError` is caught (line 124), so it propagates. (2) `__init__` calls `optimize_attention_computation(prefer_flash=..., prefer_xformers=...)`, but the real signature is `(q, k, v, is_causal=..., attention_mask=..., dropout_p=...)` returning a Tensor → `TypeError`.
- **Why it matters:** Both reproduced; the class (and its multihead wrapper, which constructs it at line 389) are non-instantiable. Exported via `base/__init__.py`. *(Auxiliary impl, not in the auto factory — limited blast radius.)*
- **Fix:** Pass only supported kwargs; catch broader exceptions or fix the kwargs; remove the `optimize_attention_computation(prefer_*=...)` call and instead call it as `(q, k, v, ...)` at forward time.

#### H8. `BlockSparseRingDistributedDilatedAttention` causal mask — *(see C15(b))*
> Folded into C15.

#### H9. `SimplifiedMemoryPool.deallocate` enables double-free / aliasing — *(see M-list / L-list)*
> See L1 (low) — severity split medium/low; placed in Low.

---

### MEDIUM

#### M1. `EnterpriseDistributedDilatedAttention._handle_forward_failure` accesses non-existent `attention_core.ring_attention`
- **File:** `src/dilated_attention_pytorch/ring/distributed/ring_distributed_dilated_attention.py:1031-1051`
- **What's wrong:** The NCCL/distributed recovery branch reads/writes `self.attention_core.ring_attention.ring_size` (line 1038). In the standard path `attention_core` is an `ImprovedMultiheadDilatedAttention` with no `ring_attention` attribute → `AttributeError`, caught at line 1050, logged, then the original error re-raised.
- **Why it matters:** `enable_fault_tolerance` defaults True; any error whose message contains `nccl`/`distributed` (but not `out of memory`) hits this. The documented "single device fallback" recovery is dead/broken for the only non-model-parallel path users run (and masks the real error with a secondary `AttributeError` in logs). Verified by instantiation.
- **Fix:** Guard with `hasattr(self.attention_core, 'ring_attention')`, or fall back via `self.attention_core(query, key, value, is_causal, False, None)` directly without `ring_size` toggling.

#### M2. `ImprovedDilatedAttention.extra_repr` calls `get_stats()` on a plain dict
- **File:** `src/dilated_attention_pytorch/base/improved_dilated_attention.py:98, 319-326`
- **What's wrong:** `self._pattern_cache = get_global_pattern_cache()` returns the module-global `{}` (a plain dict). `extra_repr` (line 322) calls `self._pattern_cache.get_stats()` and indexes `cache_stats['size']`, but dict has no `get_stats()`.
- **Why it matters:** `repr(module)`/`print(module)`/`str(module)` on any `ImprovedDilatedAttention` (a core export and factory `improved` target) raises `AttributeError`. Reproduced directly and via the factory. Forward math is unaffected — breaks common debugging/logging.
- **Fix:** Guard with `hasattr(self._pattern_cache, 'get_stats')`, report `len(self._pattern_cache)` for a dict, or assign an actual cache instance.

#### M3. Two distinct classes named `BlockSparseAdaptive` with divergent constructors
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_adaptive_fixed.py:18-67`
- **What's wrong:** Top-level `dilated_attention_pytorch.BlockSparseAdaptive` is the **original** (requires positional `num_heads`, `head_dim`), while `create_block_sparse_attention('adaptive')` / `create_adaptive_block_sparse` build the `_fixed` **subclass** (accepts `embed_dim`, lazily infers heads/dim). `isinstance` passes (subclass), but constructing the top-level class with factory-style kwargs raises `TypeError: missing 1 required positional argument: 'head_dim'` (verified).
- **Why it matters:** A documented public class behaves differently depending on entry point; mixing the two construction styles fails loudly and silently drops `embed_dim`.
- **Fix:** Pick one canonical `BlockSparseAdaptive` — export the `_fixed` subclass at top level, or rename the `_fixed` class so two classes don't share the name across public entry points.

#### M4. `get_block_sparse_preset('hilbert_standard'/'hilbert_ultra')` always raises
- **File:** `src/dilated_attention_pytorch/sparse/block_sparse_factory.py:135-143, 291-312, 315-327`
- **What's wrong:** The presets dict still defines `hilbert_standard`/`hilbert_ultra` with `variant='hilbert'`, but `create_block_sparse_attention` raises `ValueError('Hilbert variant has been removed')` for `variant=='hilbert'`. The membership guard at line 315 doesn't exclude them.
- **Why it matters:** `get_block_sparse_preset` is a public top-level export; two advertised presets are guaranteed runtime failures. Reproduced. Fails loudly with a helpful message (hence medium, not higher).
- **Fix:** Remove the two `hilbert_*` entries (and the dead `hilbert` branch), or repoint them to a working variant (e.g. `base` with `dilated_sparse`).

---

### LOW

#### L1. `SimplifiedMemoryPool.deallocate` enables double-free / buffer aliasing
- **File:** `src/dilated_attention_pytorch/core/unified_memory_pool.py:103-176`
- **What's wrong:** `deallocate()` unconditionally appends to `self._free_tensors[key]` with no duplicate/liveness check and never removes `id(tensor)` from `_allocated_tensors`. Two `deallocate(t)` calls push `[t, t]`; subsequent `allocate()` hands the **same** object to two callers (aliasing). `_allocated_tensors` grows unbounded and is never consulted.
- **Why it matters:** Exported public pool API (`SimplifiedMemoryPool`/`UnifiedMemoryPool`/`MemoryPool`/`get_global_memory_pool`); reproduced (`a is b == True`). *(Severity split medium/low: no live library path calls this `deallocate` — the active pool is `EnhancedMemoryPool` and `head_parallel` imports the different `core.memory_pool` pool — so impact is confined to external callers. Marked low.)*
- **Fix:** In `deallocate()`, only re-add if `id(tensor)` is currently tracked as allocated, then discard it; guard against pushing a tensor already in the free list (track free ids in a set).

#### L2. `OptimizedPatternCache._add_to_gpu_cache` double-counts GPU memory on overwrite
- **File:** `src/dilated_attention_pytorch/core/optimized_pattern_cache.py:262-287`
- **What's wrong:** `self._gpu_cache[key] = pattern` (line 286) overwrites an existing entry while `self._gpu_memory_used += pattern_size_mb` (line 287) adds unconditionally — the old size is never subtracted. Re-adding a key inflates the counter, causing premature/spurious LRU eviction of hot patterns and an inaccurate `gpu_memory_used_mb` stat.
- **Why it matters:** Exported `OptimizedPatternCache`; reproduced (15× `put` of one 1MB tensor → 10MB counted; eviction of still-hot keys). *(Correction: `pin_pattern` is guarded with `if key not in self._gpu_cache` and does **not** double-count — the genuine triggers are `put(store_on_gpu=True)` and `_promote_to_gpu`.)* No live attention path calls this — degraded cache efficiency only.
- **Fix:** Before inserting, if `key in self._gpu_cache`, pop the old entry and subtract its size, then add the new pattern and its size.

#### L3. `head_parallel` calls non-existent `.allocate()` on `UnifiedMemoryPool`
- **File:** `src/dilated_attention_pytorch/base/head_parallel_dilated_attention_optimized.py:21, 79-81, 176-180, 224-225`
- **What's wrong:** `get_global_memory_pool()` (from `core.memory_pool`) returns a `UnifiedMemoryPool` exposing `get_buffer()` but **no** `allocate()`. Lines 178/225 call `self._memory_pool.allocate(...)` → `AttributeError` (verified).
- **Why it matters:** **Contested reachability.** One reviewer high (multi-GPU path reaches line 178 and crashes); the other not-real, because the constructor crashes first with the `TypeError` from H7 — the class is **never instantiable**, so `allocate` is unreachable. Given H7, this is masked/latent; marked low. (Line 225 is in a dead fallback branch regardless.)
- **Fix:** Use `self._memory_pool.get_buffer(shape, dtype=..., device=...)`, or switch to `core.unified_memory_pool.get_global_memory_pool` (its `SimplifiedMemoryPool` implements `allocate()`). Fix H7 first or this stays dormant.

---

### CONTESTED

#### CT1. `RingAttentionFunction` does not replay dropout in backward
- **File:** `src/dilated_attention_pytorch/ring/utils/ring_attention_autograd.py:131-133, 87, 207-224`
- **Claim:** Forward applies `F.dropout(output, ...)` and stores `ctx.dropout_p`, but no mask/RNG state is saved and `ctx.dropout_p` is never read in backward, so gradients omit the dropout mask and `1/(1-p)` scaling.
- **Status: CONTESTED (split decision).** Reviewer A (medium): real latent bug, but `dropout_p` defaults to 0 and no first-party caller passes `>0`; only externally importable users with dropout enabled get wrong gradients. Reviewer B (not real): the forward dropout **never fires** — the guard is `if dropout_p > 0 and q.requires_grad`, and inside `Function.forward` (no-grad context) `q = q * scale` makes `q.requires_grad == False`, so `F.dropout` is never executed regardless of `dropout_p`. Verified: forward output identical for `dropout_p=0.0` vs `0.5`; zero `F.dropout` calls.
- **Net:** There **is** a genuine but different latent defect — dropout is silently a no-op (never applies even when requested) and `ctx.dropout_p` is dead code. The reported *gradient-inconsistency* defect does not currently trigger. Both reviewers agree the `RingAttentionFunction` should not be used as-is (see C6/C7/C8).
- **Fix:** Either make dropout actually apply and replay it in backward (save mask/RNG state), or move dropout outside the Function (apply `F.dropout` in the `ring_attention` wrapper). Also fix the never-fires guard so requested dropout is honored.

#### CT2. Head-parallel multi-GPU forward uses non-differentiable in-place `all_reduce`
- **File:** `src/dilated_attention_pytorch/base/head_parallel_dilated_attention_optimized.py:175-190`
- **Claim:** The `world_size>1` path scatters `local_output` into `full_output`, then calls non-autograd `dist.all_reduce(full_output, SUM)` (line 188), which has no backward, so cross-rank gradients are dropped.
- **Status: CONTESTED (split decision).** Reviewer A (not real): the index-assignment produces a `CopySlices` autograd node, the local gradient reaches `local_output` correctly even with a pooled buffer, and the cross-rank gradient that plain `all_reduce` omits is the **correct** behavior for disjoint head-parallelism (each rank computes only its own heads' gradient). Reviewer B (medium): real — with **replicated** projection weights, each rank accumulates gradient only from its local head slice, not the all-heads sum, so distributed gradients are incomplete; verified `rank0 W.grad != full-heads reference`. Also notes the `use_memory_pool=True` default crashes via L3 first, so the buggy path is only reached with `use_memory_pool=False`.
- **Net:** Genuine concern only for the replicated-projection-weight case; the local-head gradient is correct. Reachable only in a narrow non-default config (and gated behind H7/L3 construction crashes). Low practical impact today.
- **Fix:** If cross-rank gradient summation is intended, use the autograd-aware `torch.distributed.nn.functional.all_reduce` or a custom `autograd.Function`; otherwise rely on DDP for replicated-parameter sync. Resolve H7/L3 first so the path is reachable at all.

---

## 3. Forked / Dead Code (`*_fixed` / versioned variants)

Confirmed issues where a "fixed"/alternate variant diverges from or shadows the live path:

- **`ring_communication_fix.py` is dead; the buggy mixin is live (C3).** The lucidrains-pattern fix with separate K/V buffers and `batch_isend_irecv` (`ring/utils/ring_communication_fix.py:92-149`) has **zero importers** in `src/`. The unpatched `RingCommunicationMixin` (C1/C2) is what every standardized ring impl actually runs. **Action:** fold the fix into the mixin (or call `ring_pass_kv_fixed`), then delete the orphan.

- **`RingDilatedAttentionCorrect` is the "correct" reference yet has its own causal bug (C4); the default `StandardRingAttention` has a different causal bug (H4).** The two diverge: `*Correct` masks via global positions for `dilation_rate==1` (proving `Standard` wrong) but itself drops the mask for dilated offsets `>0`. Neither is fully correct. **Action:** converge both onto one shared, correct global-position mask helper.

- **Two live `BlockSparseAdaptive` classes (M3).** Top-level export = original (`block_sparse_adaptive.py`, strict positional ctor); factory builds `_fixed` subclass (`block_sparse_adaptive_fixed.py`, lazy `embed_dim` ctor). Same name, incompatible constructors. **Action:** pick one canonical class / export, rename the other.

- **`hilbert_standard`/`hilbert_ultra` presets reference a removed variant (M4).** Public presets in `block_sparse_factory.py` point at `variant='hilbert'`, which the same module unconditionally rejects. **Action:** delete the stale entries or repoint to a working variant.

- **Triton `kernels/` custom-backward Function (C10):** the forward kernel and the hand-written backward implement two different functions — a fork of intent within one class. **Action:** make backward the exact gradient of the (corrected) forward, or drop the custom Function and let autograd differentiate the PyTorch reordered-attention path.