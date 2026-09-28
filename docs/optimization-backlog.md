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
- **F4 — `keep_sources` = the 10 smallest MathInstruct sources = 36.9% of examples.** Their
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
| O1 | Skip the ZO forward for `keep_sources` examples and for sources with a zero quota | math-exact | ~37% of the selection forward. Needs source ids all-gathered before the forward (multi-GPU). Sources selected in full still need g_i (Adam state update). Keep F2 divisor. |
| O2 | fp16 autocast around the ±eps final-layer ZO calls | precision change | upstream was fp32 too (F3 correction); decide with D2. Measured: 290 ms/step fp32 |
| O3 | Gather the 32 scalars g_i per rank instead of `[32, 327680]` fp32 features; rank 0 regenerates z from the seed | bitwise | removes 42 MB/rank D2H + pickle + gather + H2D; multi-GPU only. Move gathered example tensors to CPU (or gather indices) to avoid cross-device CUDA contexts. |
| O4 | Drop KV cache + deepcopy in the selection forward | math-exact | done in the port |
| O5 | lm_head + CE only at label positions in the ZO final layer (keep F2 divisor) | math-exact | lm_head is the largest part of each ±eps call |
| O6 | Packed inputs: training (pack each original sub-batch → identical token mean; bigger packs with per-token weights), selection (pack + per-segment loss sum / original divisor) | math-exact | combine with O1/O5; `use_cache=False` (F5) |
| O7 | Attention backend: identify the SDPA backend actually used; test `varlen_attn` on sm_120; flex_attention | kernel only | flash needs fp16/bf16 (see decision D2) |
| O8 | Train the 16 selected examples in 1–2 large batches with per-token weights reproducing the per-sub-batch means | math-exact (dropout masks differ) | bs 2 micro-batches underuse the GPU |
| O9 | Reuse selection-forward activations for the training backward | needs D3 | Blockers: eval vs train dropout, fp32 vs fp16, backward cannot drop unselected rows of a batched graph. Viable variant: per-example graphs, backward only the selected ones; est. −16% compute, 40–50 GB activation memory (estimate). |
| O10 | 1-D facility location on r_i (F1) | approximate (adam-ε, ties) | needs D4 |

## Open decisions (user)

- D1: keep F2 divisor (current) or normalise by valid tokens (changes selection).
- D2: selection forward precision: keep fp32 (upstream) or fp16 autocast like training (enables
  flash varlen and O9).
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
