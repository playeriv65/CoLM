# Selection / step optimisation backlog

Goal: make a CoLM training step faster **without changing the mini-batch selection semantics**.
Every item states whether it is bitwise-exact, mathematically exact (float rounding only), or a
semantic change that needs an explicit decision. Measure before and after each item with the
`profile_timing` breakdown (one run, ≥120 steps, drop 10 warmup, closure against step wall clock).

Default config analysed: `configs/math_phi2_efficient.json` (phi-2 + LoRA r=128, per-device bs 4,
GAS 8, `small_batch_ratio` 0.5, efficient MeZO on `layers.31.self_attn.v_proj.lora_B`,
327,680 params, `zo_dim` 2560, l1 facility location, proportional per source, `mezo_optim=adam`).
Per rank: 32 examples forwarded for selection, 16 trained (8 micro-batches of 2).

## Findings (read before optimising)

- **F1 — selection features are rank one.** `zo_random_seed` is drawn once in `__init__` and
  re-seeded on every estimate, so every example, step and rank uses the same direction z.
  Features are `g_i · z` (g_i a scalar). CoLM's Adam state stays collinear with z (m) and z² (v),
  so after the transform rows differ only by `adam_epsilon`-order terms; top-k masking picks the
  largest |z_k| (a fixed set); l1 distance is `|r_i − r_j| · const`. Facility location is
  effectively one-dimensional on r_i = m̂_i / sqrt(v̂_i). Upstream behaviour, kept on purpose.
- **F2 — per-sample ZO loss divides by the padded micro-batch length**
  (`per_token_losses.mean(dim=1)`), so an example's g_i depends on which examples share its
  micro-batch. Probably an upstream bug; preserved. Any re-batching of the selection forward must
  keep the original divisor (drop rows but keep width, or divide by the stored original length).
- **F3 — precision.** Upstream runs `forward_till_penultimate` with no autocast → pure fp32 on
  phi-2 (fp32 weights, fp16 AMP), and the ±eps final-layer calls under
  `compute_loss_context_manager` (fp16 autocast). The port's `zo_forward_final_layer` lost the
  autocast (fp32 now): a fidelity deviation to fix. The fp32 selection forward is a likely reason
  selection is ~60% of the step.
  **Correction (measured 2026-09-28):** in transformers 4.43 `compute_loss_context_manager` only
  enters autocast for `use_cpu_amp` (`autocast_smart_context_manager` returns `nullcontext` on
  GPU; GPU AMP comes from accelerate wrapping `model.forward`). The decomposer never goes through
  `model.forward`, so upstream's ±eps calls were fp32 as well; the port matches upstream and O2 is a
  precision change (D2), not a fidelity fix. The fp32 prefix forward is confirmed as the dominant
  cost (see "Measured baseline").
- **F4 — `keep_sources` = the 10 smallest MathInstruct sources = 26.9% of examples**
  (70,615 / 262,039; an earlier version of this note said 36.9%, wrong arithmetic). Their
  features are discarded before the Adam transform; the 4 selected sources (aqua_rat 34.1%,
  math50k_camel 18.9%, gsm_rft 10.8%, mathqa 9.3%, together 73.1%) are the only ones whose features
  matter. Measured with O1: 28.1% of the 32 examples per step need no MeZO forward (23.0 forwarded,
  mean of the 120 timed steps; kept sources plus the occasional zero-budget source).
- **F5 — packed inputs in transformers 5.17 work** (`DataCollatorWithFlattening`, restarting
  `position_ids`, `attention_mask=None`): eager / sdpa / flex_attention all match padded logits to
  ~2e-7 in fp32 on a tiny Phi. **Trap:** packing is only detected when `past_key_values is None`;
  Phi's default `use_cache=True` builds a `DynamicCache` and sequences silently attend across
  boundaries (max logit diff 0.49). Always pass `use_cache=False`, and test for it.
- **F6 — sdpa + packing wastes compute.** transformers passes a dense `[1,1,T,T]` bool mask with
  `is_causal=False`. SDPA's flash backend rejects arbitrary masks, so it falls back to
  efficient/math kernels that compute every tile: cost (ΣLᵢ)² instead of ΣLᵢ². Negligible for
  2 sequences per row; ~equal to the layer GEMM cost for 32×400 tokens (T/6h ≈ 0.83, estimate).
  Kernels that pay only ΣLᵢ²: flash varlen (`torch.nn.attention.varlen.varlen_attn` in torch 2.13,
  register via `AttentionInterface`; or flash-attn's `flash_attn_varlen_func`), flex_attention
  (block-sparse). Flash kernels need fp16/bf16.
- **F7 — SDPA backends.** SDPA dispatches to FLASH (fp16/bf16, no arbitrary mask), EFFICIENT
  (fp32 ok, masks as bias, no tile skipping), CUDNN, MATH. Padded training batches pass a 4D mask
  and the selection forward is fp32, so the current code likely never uses the flash backend.
  To verify on GPU (`torch.nn.attention.sdpa_kernel` restriction or kernel names in a profile).
  **Verified (profile kernel names, sm_120):** the fp32 selection forward runs
  `fmha_cutlassF_f32_aligned_64x128_rf_sm80` (EFFICIENT), the padded fp16 training forward
  `fmha_cutlassF_f16_aligned_64x128_rf_sm80` (EFFICIENT; FLASH is rejected by the padding mask);
  neither ever used FLASH or cuDNN.
- **F8 — the selection is decided at fp32-rounding level.** g_i = (L₊ − L₋) / 2ε is a difference
  of two fp32 losses of size ~1–2 at ε = 1e-3, so one ulp of a loss moves g_i by ~1e-4 absolute
  (~1e-3 relative); at step 0 the Adam transform turns every feature into ≈ sign(g_i z) (all
  examples of one sign tie up to `adam_epsilon`), later steps stay near-degenerate (F1). Teacher
  forced on phi-2 (same weights, batch, Adam state, RNG), the *original* implementation disagrees
  with itself when only the fp32 operation order changes: rows of each micro-batch reversed →
  identical selection in 6/20 and 10/20 steps (two runs), every row as its own micro-batch (same
  widths and divisors) → 9/20; mean overlap 14.9–15.2 of 16 selected. Identical selected indices
  are therefore not attainable by any change that reorders fp32 arithmetic; the criteria used are
  identity in float64 on CPU (`tests/test_exact_opts.py`) and, on GPU, agreement with the
  original path no worse than these noise floors (`scripts/check_exact_opts.py`).
- **F9 — fp16-AMP training gradients of phi-2 are precision-dominated.** Against an fp32 eager
  reference (itself 1.4e-3 from fp64), the original training path (fp16 autocast, sdpa
  EFFICIENT) has relative gradient error 0.63 (cosine 0.82), mean of 8 steps (range 0.40–0.87);
  every fp16 kernel is in that class (flash-attn 0.72 / cos 0.80, cuDNN, math). In fp32, sdpa
  EFFICIENT's *backward* alone is 0.38 off on phi-2 (cos 0.94) while eager / math are 1e-3; its
  forward is accurate (hidden states 4.6e-5 from fp64). Upstream (transformers 4.43) used the
  same unupcast fp16 SDPA path (q/k are upcast only in eager), so this is the paper's recipe, not a
  port deviation; not changed here (a precision decision like D2).
- **F10 — the original run is not reproducible run to run.** Two identical HEAD runs (same
  seed): step-0 grad norm differs by 1e-6 relative, step 1 by 7e-5 (fp16 EFFICIENT backward
  atomics; repeat-to-repeat gradient difference 1e-4–1.7e-2), and the selection differs from
  step 2 on. End-to-end comparisons of two runs say nothing about exactness; compare teacher
  forced.

## Candidates

| id | change | exactness | notes |
|---|---|---|---|
| O1 | Skip the ZO forward for `keep_sources` examples and for sources with a zero quota | math-exact | **done** (`skip_unused_features`). Quotas from all-gathered source ids before the forward (`features_needed`); sources selected in full keep g_i under `mezo_optim=adam`; if the kept examples fill the budget no forward runs (the RNG epilogue `manual_seed` + one z draw still runs, so the training dropout stream is unchanged); `balanced` / `mezo_topk=sampling` fall back to skipping `keep_sources` only (their budgets / masks draw `np.random`). Skips 28.1% of examples (F4), not ~37%. |
| O2 | fp16 autocast around the ±eps final-layer ZO calls | precision change | upstream was fp32 too (F3 correction); decide with D2. Measured: 290 ms/step fp32 |
| O3 | Gather the 32 scalars g_i per rank instead of `[32, 327680]` fp32 features; rank 0 regenerates z from the seed | bitwise | removes 42 MB/rank D2H + pickle + gather + H2D; multi-GPU only. Move gathered example tensors to CPU (or gather indices) to avoid cross-device CUDA contexts. |
| O4 | Drop KV cache + deepcopy in the selection forward | math-exact | done in the port |
| O5 | lm_head + CE only at label positions in the ZO final layer (keep F2 divisor) | math-exact | **done** (`zo_label_positions_only`): final layer norm + LM head on the positions that predict a label, per-example sum / original divisor. Liger 0.8.3 was checked and not used: no patch for `phi` (phi-2; only `phi3`), so `use_liger_kernel` does nothing here; its fused linear CE has `reduction="none"` (per-sample sums possible) but only saves the [n, vocab] logits memory, not the LM-head GEMM that dominates, and in training it would replace HF's loss with a fused fp16 backward. Training uses `logits_to_keep` (built in) at label positions instead. |
| O6 | Packed inputs: training (pack each original sub-batch → identical token mean; bigger packs with per-token weights), selection (pack + per-segment loss sum / original divisor) | math-exact (training: dropout masks differ) | **done**: `zo_packing` (one row of all needed examples, `colm_varlen` fp32 attention), `train_packing=sub_batch|merged`; `use_cache=False` everywhere, tested (F5) |
| O7 | Attention backend: identify the SDPA backend actually used; test `varlen_attn` on sm_120; flex_attention | kernel only (fp16: same precision class, F9) | **done**: see F7 (EFFICIENT everywhere before) and "Attention on sm_120" below; training uses flash-attn 2 (`train_attn_implementation=auto`), selection `colm_varlen` (exact fp32) |
| O8 | Train the 16 selected examples in 1–2 large batches with per-token weights reproducing the per-sub-batch means | math-exact (dropout masks differ) | **done** (`train_packing=merged`): one packed forward per step (or rows of `train_pack_max_tokens`), loss = Σ_sub-batches (Σ token CE / label count), HF loop sees placeholders + the packed rows |
| O9 | Reuse selection-forward activations for the training backward | needs D3 | Blockers: eval vs train dropout, fp32 vs fp16, backward cannot drop unselected rows of a batched graph. Viable variant: per-example graphs, backward only the selected ones; est. −16% compute, 40–50 GB activation memory (estimate). |
| O10 | 1-D facility location on r_i (F1) | approximate (adam-ε, ties) | needs D4 |
| O11 | Last decoder layer of the ±eps calls: queries, attention output and MLP only at label positions (K/V still for every token) | math-exact | not done (needs a custom Phi layer, no preset); 22 ms per call now, est. −25 ms per step |
| H | Host walks: `model.eval()` / `model.train()` per forward, HF `floating_point_ops` parameter walk | bitwise | **done** (`lazy_mode_switch`, `cache_flops`): mode (and attention implementation) switched once per phase, parameter count cached |

## Exact optimisations: implementation and checks (2026-09-28)

Flags (`colm/train/training_arguments.py`, "Exact step optimisations"; all default on, all off =
`configs/timing_phi2_efficient_original.json`, re-timed at 2860 ms vs the 2868 ms baseline, so
the flags reproduce the old code): `lazy_mode_switch`, `cache_flops` (H), `skip_unused_features`
(O1), `zo_packing` + `zo_attn_implementation=colm_varlen` (O6 selection, O7), `zo_label_positions_only`
(O5), `train_packing=merged` + `train_attn_implementation=auto` (O6/O8 training, O7). Code:
`colm/train/packing.py`, `colm/train/attention.py`, `features_needed`
(`colm/train/facility_location.py`), `SubsetTrainerEfficient.zo_features` / `_train_microbatches`,
`SubsetTrainer._packed_loss`.

Exactness classes:

- **bitwise**: H (the mode and the attention implementation are set once per phase and put back at
  every step end; the flop count uses the same integers).
- **math-exact, fp32 rounding only**: O1, O5, O6 (selection). Each example keeps the F2 divisor
  (padded width of its original micro-batch − 1). The RNG stream after selection is unchanged
  (checked bit for bit, also when no forward runs at all), so the training dropout masks are the
  original ones when only these flags are on. The perturb/restore arithmetic (w + εz − 2εz + εz) is
  the original one, run once per forward group instead of once per micro-batch, so the LoRA-B
  rounding drift differs at ulp level.
- **math-exact, dropout masks differ, fp16 kernel changes**: O6/O8 training + flash-attn. The loss
  is Σ over sub-batches of (Σ token CE / label count) / n_sub-batches, exactly the original; dropout
  draws the same distribution on differently shaped tensors, so the masks differ (as in any
  re-batching); flash-attn replaces SDPA EFFICIENT inside the same fp16 precision class (F9).

Checks:

- CPU, tiny random Phi in **float64** (`tests/test_exact_opts.py`; fp32 makes g_i noise-dominated,
  F8): for every flag alone and all together, `mezo_optim` adam and sgd, over 4 steps: selected
  indices identical, g_i to 1e-6, MeZO Adam state to 1e-6, CPU RNG state after selection
  identical, trained micro-batches identical; no-forward steps (3 of 4 sources kept); the
  multi-rank budget slice; `features_needed` rules against `_per_class_budget`; packed decomposer
  (dense-mask SDPA, rows or one row, `colm_varlen`) = padded hidden states and losses to 1e-5;
  packed training forward = padded logits, and without `use_cache=False` it is not (F5 trap);
  training gradients of `sub_batch`, `merged` and `merged` with a row budget = original to 1e-5
  (the CE is fp32 in both), returned loss to 1e-6; cached flops identical; modes after
  selection / training. The existing 2-rank gloo test runs on the new defaults.
- GPU, phi-2, teacher forced (`scripts/check_exact_opts.py`, 3 runs × 20 steps; logs
  `logs/traces/check-exact-opts-*`): selected set identical to the original in 6, 7, 11 of 20 steps
  (24/60); the original against itself with the rows of each micro-batch reversed: 3, 6, 10 (19/60);
  with every row as its own micro-batch: 9/20. Mean overlap of the 16 selected: 14.7–15.35 (new)
  vs 14.9–15.15 (reversed) and 15.1 (rows). g_i median relative difference 2.1–2.3e-3 (new) vs
  1.7e-3 (rows floor); maxima are at g_i ≈ 0. CUDA RNG state after selection identical in all
  60 steps. So the new selection agrees with the original as well as the original agrees with
  itself under fp32 reordering (F8); identical indices in every step are not attainable.
- GPU training gradients, same selected examples, dropout off, 8 steps: packed (one row per
  sub-batch) vs padded in fp32 eager without autocast: relative gradient difference mean 1.5e-3
  (max 2.1e-3), loss 2.7e-6 — the fp32 floor (fp32 eager vs fp64: 1.4e-3). In fp16 AMP against
  that fp32 reference: original 0.63 relative / cosine 0.82, new (flash-attn, one merged row) 0.72 /
  0.80, per step overlapping (0.40–0.87 vs 0.40–1.09; cosine 0.68–0.94 vs 0.66–0.94); repeat of
  the original 5.5e-3 (nondeterminism).

## Exact optimisations: measured (2026-09-28)

Protocol as the baseline: `configs/timing_phi2_efficient.json` via `scripts/run.sh` (torchrun, 1
process), GPU 2 alone, `profile_timing=fine`, 130 steps, census on steps 1–10, means over steps
11–130, W&B off, no GPU sampling. One run per column; the columns add the items cumulatively
(flags in the run configs; the default config is the "rows ≤1024" column). Same seed, so the same
data order; the selected examples differ slightly (F8).

| phase (ms / step) | baseline (09-28, old code) | original flags | +H +O1 | +O5 +O6/O7 sel. | + training, rows ≤1024 (default) | + training, 1 row |
|---|---:|---:|---:|---:|---:|---:|
| **step** | 2868 | 2860 | 2259 | 1737 | 1402 | 1257 |
| selection | 1869 | 1870 | 1325 | 808 | 810 | 810 |
| · ZO features | 1828 | 1829 | 1284 | 767 | 767 | 768 |
| · · forward_till_penultimate (31 layers, fp32) | 1525 | 1525 | 1107 | 687 | 686 | 687 |
| · · · layers | 1489 | 1489 | 1096 | 682 | 682 | 682 |
| · · · `model.eval()` walk | 28 | 28 | 3.7 | 4.0 | 3.8 | 3.7 |
| · · final layer +eps | 148 | 148 | 82 | 39 | 39 | 39 |
| · · final layer −eps | 142 | 142 | 81 | 39 | 39 | 39 |
| · · · lm_head (+eps) | 57 | 57 | 42 | 15 | 15 | 15 |
| · · · last decoder layer (+eps) | 49 | 49 | 36 | 22 | 22 | 22 |
| · · plan (source gather, budgets, packing) | – | – | 1.8 | 1.6 | 1.5 | 1.5 |
| · features D2H | 20 | 20 | 20 | 20 | 21 | 21 |
| · rank-0 selection | 16 | 16 | 16 | 16 | 17 | 16 |
| train | 917 | 908 | 887 | 882 | 545 | 400 |
| · forward | 394 | 391 | 393 | 393 | 247 | 172 |
| · backward | 478 | 472 | 472 | 468 | 282 | 221 |
| · `model.train()` walk | 28 | 28 | 4.3 | 4.4 | 4.3 | 4.2 |
| HF `floating_point_ops` | 36 | 36 | 0.3 | 0.3 | 0.2 | 0.1 |
| data | 25 | 25 | 24 | 24 | 24 | 25 |
| optimizer | 18 | 18 | 19 | 19 | 19 | 19 |
| p90 step | 3154 | 3133 | 2540 | 1944 | 1639 | 1466 |
| closure residual (ms) | 2.6 | 2.6 | 2.4 | 2.4 | 1.8 | 1.1 |
| 50-step sliding max dev (%) | 1.5 | 1.1 | 2.8 | 2.4 | 3.5 | 3.5 |
| loadavg start / end | 12.1 / 19.4 | 12.4 / 7.0 | 7.0 / 12.6 | 10.5 / 4.4 | 11.9 / 6.3 | 11.1 / 9.5 |
| MeZO examples forwarded / 32 | – | – | 23.0 | 23.0 | 23.0 | 23.0 |
| MeZO tokens forwarded | – | – | 8343 | 5366 | 5366 | – |
| MeZO logit positions | – | – | 8343 | 3191 | 3191 | – |
| train tokens (padded) | 6499 | 6532 | 6511 | 6531 | 3576 | 3589 |
| train tokens (real) | 3549 | 3587 | 3570 | 3590 | 3576 | 3589 |

Peak memory (training): 26.5 GB (original, +H +O1, +selection), 26.3 GB (rows ≤ 1024), 53.9 GB
(one row per step). Per item, from the columns: host walks −150 ms (`model.eval()` 28 + 60 →
3.7, `model.train()` 28 → 4.3, flop count 36 → 0.3); O1 −418 ms prefix and −127 ms final layers
(23.0 of 32 examples forwarded, 28.1% skipped; 8.3k instead of 11.6k padded tokens); O6 + O7
(`colm_varlen`) + O5 −420 ms prefix (5.4k packed tokens, no padding) and −85 ms final layers
(LM head on 3.2k label positions instead of 8.3k positions: 42 → 15 ms per call); training
packing + flash −337 ms (rows ≤ 1024: whole sub-batches of ~450 tokens, so ~4 forwards per step
instead of 8, no padding) or −482 ms (one row).
Total: 2860 → 1402 ms (−51%, 2.04×) at the default, 1257 ms (−56%, 2.28×) with one training row.
The prefix costs 127–131 µs per forwarded token in every column (47.9 → 22.0 ms per layer): it
is linear in tokens and the packed varlen path adds nothing per token; the next lever there is
the fp32 precision itself (D2). Selection still spends ~37 ms on the [32, 327680] feature matrix
round trip (D2H 20 + H2D 11 + host), O3 territory.

## Attention on sm_120 (RTX PRO 6000 Blackwell, torch 2.13 cu130, transformers 5.17)

Short diagnostic, not a trainer timing run (`logs/diag-attn-backends-phi2-gpu2-*.log`, 5 random
large batches, host loaded by a parallel flash-attn build; relative numbers only).

- **Before:** SDPA EFFICIENT for both forwards (F7).
- **Training (fp16 autocast), in the order asked:** (a) flash-attn 2.8.3.post1 builds from source
  for sm_120 only (`FLASH_ATTN_CUDA_ARCHS=120`, 32 jobs, 4 min 54 s; `uv sync --extra flash`) and
  runs; transformers' `flash_attention_2` takes packed rows through `cu_seq_lens_q/k` +
  `max_length_q/k` kwargs (no per-layer position-id scan). Used by `train_attn_implementation=auto`.
  Numerics: same fp16 precision class as SDPA EFFICIENT (F9). (b) `kernels-community` and (d)
  flex_attention were not needed. (c) `torch.nn.attention.varlen.varlen_attn` (flash, in torch)
  also works on sm_120: 16 examples in one packed row, fwd+bwd 374 ms vs 777 ms padded SDPA
  (8 × 2); the same row with SDPA and its dense block mask: 741 ms (F6), one row per sub-batch
  with SDPA: 717 ms. A no-build alternative to flash-attn if needed.
- **Selection (fp32, precision unchanged):** first 31 layers over the examples O1 keeps
  (5.3k real tokens per step on average, vs 11.3k padded): padded 8 × 4 1402 ms, O1 on padded
  micro-batches 1107 ms, packed + SDPA dense block mask 986 ms (one row) / 924 (rows ≤ 2048
  tokens) / 897 (≤ 1024) / 915 (≤ 512), packed + nested-tensor (jagged) SDPA 730 ms, packed +
  the EFFICIENT kernel called with cumulative sequence lengths 691 ms. The last one is
  `colm_varlen` (`colm/train/attention.py`, registered through `AttentionInterface`; one SDPA
  call per sequence on CPU). Every packed variant deviates from the padded forward by the same
  max|Δh| / max|h| = 2.8e-4 at the output of layer 30 (dense-mask SDPA with rows ≤ 1024, jagged
  SDPA, `colm_varlen`; one dense row 4.7e-4), so the deviation comes from the changed GEMM
  shapes, not from the attention kernel (fp32 SDPA EFFICIENT hidden states are 4.6e-5 median per
  token from fp64). flash kernels are fp16/bf16 only and transformers' flash path silently casts
  fp32 queries, so the selection never uses them (D2).

## Open decisions (user)

- D1: keep F2 divisor (current) or normalise by valid tokens (changes selection).
- D2: selection forward precision: keep fp32 (upstream) or fp16 autocast like training (enables
  flash varlen and O9).
- D3: accept `enable_dropout=False` (no LoRA / residual dropout) to make O9 exact.
- D4: whether O10 is acceptable.

## Measured baseline (2026-09-28, commit of `profile_timing`)

`configs/timing_phi2_efficient.json` (the default config above, 130 steps, `profile_timing=fine`,
census on steps 1–10; code before the exact optimisations, i.e. today's
`configs/timing_phi2_efficient_original.json`), 1× RTX PRO 6000 Blackwell, torch 2.13 cu130,
transformers 5.17. Means over steps 11–130 (120 steps); 50-step sliding mean within ±1.5% of the
window mean; loadavg 12.1 at start, 19.4 at end. Top-level residual vs the step clock: 2.6 ms
(0.09%). Fine timers synchronise at every section boundary (per layer). Instrumentation overhead,
from `configs/timing_phi2_efficient_coarse.json` (coarse timers only, 60 steps, same seed and so
the same batches) over steps 11–60: step 2866 vs 2836 ms (+1.1%), selection 1867 vs 1841 ms
(+1.4%), prefix forward +1.6%, training +0.4%; below 5%, so the fine run's numbers stand.

| phase (ms / optimizer step) | mean | % step | p50 | p90 |
|---|---:|---:|---:|---:|
| **step** (wall clock between optimizer steps) | 2868 | 100 | 2868 | 3154 |
| selection | 1869 | 65.2 | 1870 | 2104 |
| · ZO features (8 micro-batches of 4) | 1828 | 63.7 | 1829 | 2065 |
| · · forward_till_penultimate (31 layers, fp32, no autocast) | 1525 | 53.2 | 1525 | 1731 |
| · · · per layer: 47.9 ms (min 47.8, max 48.1) × 31 | 1489 | 51.9 | | |
| · · · `model.eval()` walk | 28 | 1.0 | | |
| · · final layer +eps / −eps (fp32) | 148 + 142 | 10.1 | | |
| · · · lm_head GEMM (each) | 57 | 2.0 | | |
| · · · last decoder layer (each; mlp 30, attn 18) | 49 | 1.7 | | |
| · · · `model.eval()` walk (each) | 33 / 27 | 1.1 | | |
| · · perturb / restore / z (4 × `manual_seed` + normal) | 13 | 0.4 | | |
| · features D2H [32, 327680] fp32 (42 MB) | 20 | 0.7 | | |
| · rank-0 selection (incl. 42 MB H2D 10 ms; FL 2 ms) | 16 | 0.5 | | |
| · other selection host work (split, recollate, stats) | 5 | 0.2 | | |
| train (8 micro-batches of 2) | 917 | 32.0 | 919 | 965 |
| · forward (fp16 autocast) | 394 | 13.7 | | |
| · backward | 478 | 16.7 | | |
| · `model.train()` walk | 28 | 1.0 | | |
| HF `floating_point_ops` (walks all parameters per micro-batch) | 36 | 1.3 | | |
| data (8 micro-batches + H2D) | 25 | 0.9 | | |
| optimizer (clip 5, step 10, sched+zero_grad 3) | 18 | 0.6 | | |

Tokens per step (means): selection forward 11,585 padded vs 7,368 real (36.4% padding), 4,257
label tokens; training 6,499 padded vs 3,549 real (45.4% padding; examples keep the padding of
their original micro-batch when re-collated), 1,969 label tokens. Step time correlates with the
selection padded-token count at r = 0.99; the prefix forward costs 123 µs per padded token
(≈ 4 µs per token-layer, ≈ 40 TFLOP/s fp32 effective), the lm_head 4.5 µs per padded token per
call. Training forward/backward have a large per-step intercept (≈ 320 ms each for 8
micro-batches, estimate from a linear fit): small micro-batches, 5.3k/7.1k aten ops per
micro-batch forward/backward and a per-forward fp32→fp16 cast of the frozen base weights.

Census (steps 2–10): per step 150 synchronising CUDA calls, 65 of them in selection (24 per-tensor
`.cpu()` of the micro-batches, 16 in `create_causal_mask`'s `.all()` checks, the 42 MB feature
D2H and H2D, ~20 small index copies on rank 0); training forward has another 16 mask checks and
the HF loop 16 `isnan/isinf` checks. Selection moves 42.1 MB D2H and 42.2 MB H2D per step
(single GPU; both are the feature matrix round trip).

Reading: selection is compute in the fp32 prefix forward (53% of the step), not facility
location (0.1%) or communication. Levers by size: precision of the prefix forward (D2), the
padded and the kept-source rows it processes (O1, O6), then lm_head at label positions only (O5)
and the fp32 final layer (F3/O2); training is overhead-bound at micro-batch 2 (O8). Host
walks (`model.eval()` ×3 and `model.train()` per micro-batch, HF flop counting) cost ≈ 150 ms
(5%) per step and are bitwise-free to remove.
