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
| O2 | Restore fp16 autocast around the ±eps final-layer ZO calls | restores upstream | fidelity fix (F3) |
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
