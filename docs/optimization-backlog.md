# Selection / step optimisation backlog

> Status: the exact (execution-only) items O1, O3, O5, O6, O7, O8 and the host items are done on the
> refactored code; results at the end of this file ("Execution-only optimisations on the
> refactored code"). Still open: O11 (optional), and the items that are user decisions (O2, O9,
> O10, D1-D4). The per-step numbers of the older sections were measured on the padded upstream
> path (`legacy`, 512-truncated data): the default path is packed without padding and untruncated
> (`docs/errors.md`), so their token counts and step times do not apply to it. F4 is corrected
> (26.9%, not 36.9%).

Goal: make a CoLM training step faster **without changing the mini-batch selection semantics**.

The profiles and timing tables through the stock-attention section below describe
the FP32-prefix baseline. The later FP16-prefix default and its measured
selection change are documented in `docs/fp16-prefix.md`.
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
- **F4 — `keep_sources` = the 10 smallest MathInstruct sources = 26.9% of examples** (70,615 / 262,039; an earlier version said 36.9%). Their
  features are discarded before the Adam transform; the 4 selected sources (aqua_rat 34.1%,
  math50k_camel 18.9%, gsm_rft 10.8%, mathqa 9.3%) are the only ones whose features matter.
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

## Candidates

| id | change | exactness | notes |
|---|---|---|---|
| O1 | Skip the ZO forward for `keep_sources` examples and for sources with a zero quota | math-exact | ~37% of the selection forward. Needs source ids all-gathered before the forward (multi-GPU). Sources selected in full still need g_i (Adam state update). Keep F2 divisor. **Done** (`CoresetSelector.needed`; 27.5% of the examples and 25.3% of the tokens skipped on the recipe). |
| O2 | fp16 autocast around the ±eps final-layer ZO calls | precision change | upstream was fp32 too (F3 correction); decide with D2. Measured: 290 ms/step fp32 |
| O3 | Gather the 32 scalars g_i per rank instead of `[32, 327680]` fp32 features; rank 0 regenerates z from the seed | bitwise | removes 42 MB/rank D2H + pickle + gather + H2D; multi-GPU only. Move gathered example tensors to CPU (or gather indices) to avoid cross-device CUDA contexts. **Done** (`Extractor.expand`). |
| O4 | Drop KV cache + deepcopy in the selection forward | math-exact | done in the port |
| O5 | lm_head + CE only at label positions in the ZO final layer (keep F2 divisor) | math-exact | lm_head is the largest part of each ±eps call. **Done** in the refactor (M1; the divisor is the example's own label count, E2). |
| O6 | Packed inputs: training (pack each original sub-batch → identical token mean; bigger packs with per-token weights), selection (pack + per-segment loss sum / original divisor) | math-exact | combine with O1/O5; `use_cache=False` (F5). **Done** in the refactor. |
| O7 | Attention backend: identify the SDPA backend actually used; test `varlen_attn` on sm_120; flex_attention | kernel only | flash needs fp16/bf16 (see decision D2). **Done**, then replaced by the stock mechanisms (last section): torch `varlen_attn` + a custom efficient-kernel selection were ~4% faster in the step than `flash_attention_2` (hub kernel) + stock `sdpa`. |
| O8 | Train the 16 selected examples in 1–2 large batches with per-token weights reproducing the per-sub-batch means | math-exact (dropout masks differ) | bs 2 micro-batches underuse the GPU. **Done** (packs under the token budget `train_max_tokens`, default 1536; 0 = one pack per step). |
| O9 | Reuse selection-forward activations for the training backward | needs D3 | Blockers: eval vs train dropout, fp32 vs fp16, backward cannot drop unselected rows of a batched graph. Viable variant: per-example graphs, backward only the selected ones; est. −16% compute, 40–50 GB activation memory (estimate). |
| O10 | 1-D facility location on r_i (F1) | approximate (adam-ε, ties) | needs D4 |
| O11 | Last decoder layer of the ±eps calls: queries, attention output and MLP only at label positions (K/V for every token) | math-exact | **Not done**: needs a per-architecture layer (the replay is the model's own layer module); est. −20 ms of 1324. |
| O12 | LoRA merged into the base weights for the fp32 selection forward (W + s·B·A per layer, once per step) | math-exact up to fp32 rounding | **Not done, measured candidate**: LoRA's skinny GEMMs and its scale/add passes are 27% of the prefix GPU time (profile below); merging costs ~20 ms per step; est. −150 ms of 1324. Needs the whole selection in one pack and non-destructive merged weights (E-drift). |
| O13 | fp32 GEMM in the NN layout: `x @ W^T` runs at 54 TFLOP/s, `x @ Wt` (weights stored transposed) at 70 on this GPU | math-exact up to rounding | **Not done, measured**: est. −130 ms (the base GEMMs are 74% of the prefix); needs a second (transposed) copy of the weights or a custom linear. |

## Open decisions (user)

- D1: keep F2 divisor (current) or normalise by valid tokens (changes selection).
- D2: FP16 prefix / FP32 last-layer and loss was adopted in the Phi-2 profile
  (`selection_prefix_dtype=float16`; `float32` restores the old
  prefix). It measured 1011 vs 1425 ms/step, but selected-set overlap is 13.0/16
  against an FP32 packing floor of 15.17/16. Learning quality remains untested;
  see `docs/fp16-prefix.md`.
- D3: accept `enable_dropout=False` (no LoRA / residual dropout) to make O9 exact.
- D4: whether O10 is acceptable.

## Measured baseline (2026-09-28, commit of `profile_timing`)

`configs/timing_phi2_efficient.json` (the default config above, 130 steps, `profile_timing=fine`,
census on steps 1–10), 1× RTX PRO 6000 Blackwell, torch 2.13 cu130,
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


## Execution-only optimisations on the refactored code (2026-09-28)

Baseline is the refactored main `fe4ccfc` (packed batches, fp32 selection, fp16 training, LM head at
label positions, no truncation, per-example loss = mean of its own label tokens). Nothing here
changes the algorithm; `docs/errors.md` semantics are untouched, the fp32 selection forward and
the fp16 AMP training are unchanged (no fp16 selection, no TF32, O9 not touched).

What changed and its exactness class (code in `colm/selection/`, `colm/train/trainers.py`):

| item | change | class | evidence |
|---|---|---|---|
| O1 | `CoresetSelector.needed(sources, total)` decides from the source ids, before any forward, which features can change the selection; the others are not computed (zeros, never read). The pool is all-gathered first (cheap), the needed examples are shared over the ranks by token count (`balanced_shares`). Skipped: `keep_sources`, sources with zero quota, sources selected in full when `mezo_optim=sgd`; everything if the kept examples fill the budget. With random tie-breaks (`balanced`), sampled coordinates, a global coordinate ranking (`source_wise_selection=none`) or a pool transform only the kept sources are skipped. | math-exact | float64 CPU: randomised pools/sources/quotas, indices, weights and the Adam moments identical (`tests/test_opt.py`), trainer path identical over 4 steps with kept sources / adam / sgd; 2- and 4-rank gloo runs identical to one process. |
| O3 / M2 | The MeZO extractor returns g_i (one scalar per example); rank 0 builds the features `g_i z` (`Extractor.expand`, z from the same seeded generator). Only scalars are gathered: no `[N, 327680]` D2H, pickle, gather, H2D. | bitwise on the same g_i | feature matrix = `g[:, None] * z` with `atol=0`; g_i bit-identical whatever the packing in float64. z is regenerated on rank 0: identical to the other ranks' z on GPUs of one architecture. |
| O8 | The selected examples of a step are packed greedily into forwards of at most `train_max_tokens` tokens (default 1536, memory mode; 0 = unlimited: one pack for the whole step, speed mode). The gradients of the packs accumulate; an example longer than the budget goes alone, none is split. The loss is a token sum over the step divided by the step's label count: independent of the grouping (`test_unlimited_and_bounded_budgets_train_the_same_step`: float64, one pack vs several, gradients 1e-9, loss 1e-12). An out-of-memory error says which value to set. `SubsetTrainer` packs too. | math-exact (fp rounding order; dropout masks differ, as for any re-batching) | float64: gradients of one pack / packs under a budget = the micro-batch loop to 1e-9 (weighted and unweighted), loss to 1e-12. GPU, fp32 with exact attention: micro-batch packs vs the new packs 1.3e-3 relative gradient difference (max 1.9e-3), loss 1.5e-6 (the fp32 floor). |
| O7 | (Historic, before the stock attention: ) Training attention kernel: torch `varlen_attn` (flash) on the packed fp16 rows. flash-attn 2.8.3 (built for sm_120) gives the same numbers (relative 7e-6 to 2e-5 between the two kernels, both 2.4e-4 to 3.8e-4 from fp32 per sequence) and no speed-up (fwd+bwd of one layer, 16 x 220 tokens: 0.42 ms torch, 0.49 ms flash-attn; 6k tokens: 0.99 vs 1.00; 2 x 2048: 0.90 vs 0.91): not adopted, no dependency, no build. End to end, fp16 gradient against the exact fp32 gradient (relative, cosine): `colm_varlen` 0.44 (0.90) one pack of 16 examples, 0.41 (0.94) one example per forward; stock fp16 sdpa 1.17 (0.65) (`docs/errors.md`). | kernel only | see left; the fp16 gradient error is the pending precision decision. |
| H | Label geometry (positions, targets, segment, counts) computed on the CPU in `pack` (no `nonzero` / `bincount` synchronisation in the losses); `ModeSwitch`: flat flag pass instead of the recursive `train()` walk (4 ms -> 0.3 ms per switch, PreTrainedModel's override is skipped only while `use_kernels` is off); `dataloader_num_workers=1` (tokenising the next pool overlaps the step: 11.5 ms); the unused pool norm of `mezo_transform=none` is no longer computed; `_features` casts to at least fp32 (float64 stays). | bitwise / host only | worker: same pools (test); mode switch: same flags on all modules (test); the census below. |
| O11 | not done (per-architecture layer replay). | | |

### Measured (phi-2, `configs/timing_phi2_efficient.json`)

Protocol: `--profile_timing fine`, 130 steps, 10 warm-up dropped, means over steps 11-130, closure
against the step wall clock, 50-step sliding mean, W&B off, no GPU sampling, physical GPU 0 alone
(other agents' jobs on other GPUs and their host load: loadavg is recorded). Same seed, so the
same pools in the same order; the trained sets differ slightly (fp32 rounding decides ties). The
old column is the previous baseline on the old code (`Measured baseline` below: 512-truncated
data, padded; not comparable, kept for reference). One run per column.

| phase (ms / optimizer step) | old code, padded (09-28) | main `fe4ccfc` | optimised |
|---|---:|---:|---:|
| **step** (wall clock between optimizer steps) | 2868 | 2329 | 1324 |
| selection | 1869 | 1221 | 886 |
| · prefix: 31 layers, fp32 | 1525 | 1029 | 774 |
| · ±eps final layer + LM head + loss (2 calls) | 290 | 135 | 101 |
| · features D2H | 20 | 21 | 0 |
| · rank-0 selection | 16 | 15 | 3 |
| · other selection (mode walk, pack, plan, expand) | 5 | 20 | 7 |
| train | 917 | 1079 | 419 |
| · forward | 394 | 511 | 182 |
| · backward | 478 | 560 | 236 |
| · mode walk, prepare, other | 28 | 8 | 1 |
| optimizer | 18 | 8 | 8 |
| HF loop, data, log (top-level other) | 61 | 22 | 11 |
| p90 step | 3154 | 2625 | 1604 |
| closure residual (ms, % of step) | 2.6 (0.09) | 21.8 (0.94) | 10.6 (0.80) |
| 50-step sliding mean, max deviation | 1.5% | 1.9% | 4.9% |
| loadavg start / end | 12.1 / 19.4 | 12.3 / 26.4 | 22.1 / 20.8 |

Per step (means): pool 7,850 tokens; MeZO forward on 23.2 of 32 examples, 5,863 tokens (74.7% of
the pool's tokens); trained 3,727 tokens in 1.0 forward (main: 8-9 forwards). The prefix costs
132 us per forwarded token in both columns (1029/7850, 774/5863): linear in tokens. The step
time of the optimised run is a function of the token counts (least squares over the 120 steps:
0.154 ms per forwarded token + 0.130 ms per trained token - 65 ms, R^2 = 0.99; correlation with
the forwarded tokens 0.96), so the 50-step deviation is the data (both runs saw the same pools);
the optimised run sits at the 5% edge because the fixed part of the step is smaller.

Production step time (`profile_timing=off`, no synchronisation; `step_time_s` of the trainer log,
steps 11-130, same protocol otherwise): main 2317 ms (sliding +-1.5%), optimised 1305 ms (+-4.7%,
0.9 fraction): **1.78x**; the fine timers cost <= 1.5% on both. Peak GPU memory (allocated, training
phase; selection 13.1 GB in both): main 32.3 GB, optimised 62.2 GB with the memory-derived budget of
8,599 tokens per forward that this commit replaced (reserved 90.8 GB): one pack of the whole step holds ~8 MiB per token
(measured: 7.92 MiB per token + 17.9 GB fixed, the check at the derived size 84.0 of 84.4 GB).

Training pack budget (`train_max_tokens`; replaces the former memory-derived budget, `train_memory_fraction`
0.9 / 0.5 / 0.3 had given 8,599 / 3,746 / 1,322 tokens, 62.2 / 45.9 / 32.4 GB, 1305 / 1316 / 1380 ms).
phi-2, default recipe, physical GPU 0 (one job on the card, loadavg 15-25 from other users), 60 steps,
step time = mean of `step_time_s` over steps 11-60 (`profile_timing=off`), memory from `MemoryMeter`
(peak of the training phase over the 60 steps), one run each, same pools. Worst case: 2 selected
examples of 2048 tokens (`scripts/memory_worst_case.py --train_examples 2`, forward + backward of
every pack, no optimizer step):

| `train_max_tokens` | step (ms) | vs unlimited | peak allocated | reserved | worst case 2 x 2048 (packs, allocated) | mean loss, 60 steps |
|---|---:|---:|---:|---:|---:|---:|
| main `fe4ccfc` (8-9 forwards of ~450 tokens) | 2317 | - | 32.3 GB | 35.4 GB | - | (130 steps) 0.689 |
| 1024 | 1483 | +7.2% | 32.3 GB | 35.8 GB | 2 packs, 32.4 GB | 0.717 |
| **1536 (default)** | **1382** | **-0.1%** | **32.3 GB** | **42.1 GB** | **2 packs, 32.4 GB** | 0.732 |
| 2048 | 1362 | -1.5% | 33.9 GB | 41.7 GB | 2 packs, 32.4 GB | 0.725 |
| 4096 | 1385 | +0.1% | 48.7 GB | 65.5 GB | 1 pack, 47.1 GB | 0.723 |
| 0 = unlimited (one pack per step) | 1383 | 0 | 57.3 GB | 94.2 GB | 1 pack, 47.1 GB | 0.712 |

Choice: the smallest budget within 5% of unlimited is 1536 (1024 is 7% slower: about 20% of the
steps then need a second pack); its peak equals main's 32.3 GB. The step time is flat from 1536 up
(the 10-step block means scatter by +-3% inside each run, so 1362-1385 ms are the same speed), memory is
not: ~8 MiB per packed token beyond ~2k. The worst case of any budget below 4096 is one 2048-token
example per forward (32.4 GB), because an example is never split; from 4096 up two such examples share
a forward (47.1 GB, and 57 GB for the longest real step). Unlimited is kept as an explicit option
(`train_max_tokens=0`): it is not faster on phi-2 at this batch size (the step is bound by the fp32
selection forward, 58%), so it only pays where the training forward is a larger share of the step.
Losses differ between the runs (first loss 0.850-0.881 for identical pools) because the selection
and the dropout masks depend on fp rounding order; one seed each, differences of 0.02 in a 60-step
mean are that noise, not an effect of the budget. Logs: `logs/pack-budget-2026-09-28/` of the main worktree
(machine-local).

Census (steps 2-10, syncing CUDA calls per step): selection 166 -> 120 (88 of them are the
host-to-device copies of the packs, done before any kernel is queued; the `nonzero` / `bincount`
of the losses are gone: 40 per step), training 101 -> 14 (forward 27.7 -> 2.0), feature D2H
42 MB -> 91 bytes, rank-0 H2D 42 MB -> 0.24 MB.

Reading: what is left is the fp32 prefix (58% of the step; precision decisions D2/O12/O13 only),
the fp16 training forward + backward (32%; 3.7k tokens in one pack), the
±eps final layers (7.6%) and 0.8% top-level HF loop.

Raw logs (machine-local, not in git): `logs/opt-2026-09-28/` of the main worktree on pro6000
(fine-timing jsonl and summaries of both columns, the `off` runs' trainer states and memory,
the GPU check).

### Checks on the GPU (phi-2, teacher forced, `scripts/check_opt.py`)

20 steps on GPU 2 (shared with another job, packs of ~1k tokens),
at every step from the same weights, pool and selector state: the plain path (all features, pack
by pack) vs the optimised `_select`, and the plain path with the pool reversed (other packs:
the noise floor of fp32 rounding).

| | optimised vs plain | plain (reversed) vs plain |
|---|---:|---:|
| identical selected set (of 20 steps) | 9 | 2 |
| mean overlap of the 16 selected | 14.90 | 13.65 |
| median relative difference of g_i | - | 1.0e-3 |

Examples forwarded: 70.3% of the pool. Median relative difference of the selector's Adam moments
after the step 1.1e-2 (0 to 3e-4 in the steps with the same selected set). Gradients of the
selected examples (dropout off, 8 steps): optimised packs vs micro-batch packs in fp16 AMP 0.48
relative (the fp16 gradient itself is 0.62 (cosine 0.84) from the exact fp32 gradient for the
micro-batch packs and 0.63 (0.83) for the new packs: the same precision class; repeating the fp16
run alone differs by 0.047); in fp32 with exact attention the two groupings differ by 1.3e-3
(the packing is exact), loss 1.5e-6, grad-norm ratio new / plain 0.99 (0.95-1.05 in fp16).

### Not done, measured (candidates)

Profile of the fp32 prefix (3.8k tokens, 482 ms, `torch.profiler`): GEMM 79% of the GPU time
(`cutlass_80_simt_sgemm` 128x256 / 256x128, TN layout, 54 TFLOP/s; the same GEMM as `x @ Wt` (NN)
reaches 70 TFLOP/s: O13), LoRA (skinny GEMMs + `mul` + `add`) 27%: O12, layer-norm / GELU /
attention the rest. O11 and the memoisation of the parallel MLP in the ±eps replays (Phi's MLP
does not depend on v_proj: ~17 ms) are each worth ~1.5% and need per-architecture code.

### Stock attention instead of `colm_varlen` (`colm/train/attention.py` deleted)

The custom attention is replaced by transformers 5.17's own mechanisms; nothing else about the
computation changes (precision recipe unchanged: fp16 autocast training, fp32 selection).

- **Training: `attn_implementation="flash_attention_2"`.** Without the `flash-attn` package (no wheel
  for torch 2.13, a source build takes minutes and gigabytes) transformers loads the hub kernel
  `kernels-community/flash-attn2` through the `kernels` package (`kernels>=0.16,<0.17`, added to
  `pyproject.toml`; transformers 5.17 refuses 0.17; version 2 of the repo has a torch 2.13 / sm_120
  build). The stock flash path takes the packed layout of
  `DataCollatorWithFlattening(return_flash_attn_kwargs=True)` (`position_ids`, `cu_seq_lens_*`,
  `max_length_*`; without the cumulative lengths it derives them from `position_ids` with a
  `nonzero` per layer) and casts q, k, v to the autocast dtype itself. Needs Ampere or newer:
  on Turing pass `--attn_implementation sdpa`.
- **Selection (fp32, no gradient): `selection_attn_implementation="sdpa"`** (dense block mask built
  from `position_ids`; the flash kernels need fp16). `flex_attention` also runs in fp32 but was
  2.8x slower (recompiles per pack shape). The trainer switches between the two with
  `model.set_attn_implementation` (1.1 ms + 1.5 ms per step).
- **Selection agreement** (6 pools of 32 examples, phi-2 at initialisation, packs of the default
  selection size; reference: stock sdpa with the exact MATH kernel): median relative difference of
  g_i 1.1e-3 (`colm_varlen`), 1.2e-3 (sdpa), 1.3e-3 (flex); the same pool run twice with one
  implementation differs by 1.4e-4 to 2.3e-4 (median); selected sets overlap the reference in
  14.8 / 14.7 / 14.8 of 16 (identical sets in 2 / 1 / 3 of 6 pools; the selection is decided at
  fp32 rounding level, `docs/errors.md` E1: upstream against itself 13.5 of 16).
- **Training gradients**: `docs/errors.md` (all fp16 kernels are in the same class, 0.4-0.5 relative
  from the fp32 gradient; hub kernel and torch `varlen_attn` agree to 1e-5 at the operator level;
  fwd + bwd of one layer 0.67 / 0.76 / 0.83 ms (hub) against 0.47 / 0.52 / 0.85 ms (torch) for
  16 x 220 / 4 sequences of 300-500 / 2 x 2048 tokens).
- **Step time** (phi-2, `configs/timing_phi2_efficient.json`, 70 steps, steps 11-70, `--profile_timing
  fine`, physical GPU 0 alone, same seed = same pools, loadavg 22-35 from other users, one run each):

| ms / optimizer step | `colm_varlen` (varlen_attn train, efficient kernel selection) | stock (flash_attention_2 train, sdpa selection) |
|---|---:|---:|
| step | 1374 | 1425 (+3.7%) |
| selection | 925 | 967 (+4.5%) |
| · prefix (31 layers, fp32) | 807 | 845 (+4.7%) |
| train | 433 | 442 (+2.1%; per trained token +3.2%) |
| · forward / backward | 202 / 227 | 206 / 231 |
| forwarded / trained tokens per step | 6097 / 3819 | 6097 / 3780 |
| 50-step sliding mean, max deviation | 1.8% | 1.4% |

  The selection cost is the dense `[T, T]` block mask of `sdpa` on packs of ~1.5k tokens (4 examples).
