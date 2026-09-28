# TODO

Living task list for this research repo. Keep it short: move finished items to "Done" with the
commit / result pointer, delete them once they are recorded in docs. Optimisation details live in
`docs/optimization-backlog.md`; this file only tracks what is next.

## In progress

## Next

- [ ] Refactor to a modern, hack-free structure (scope to be agreed; see "Refactor" below).
- [ ] LoRA rank sweep: r ∈ {8, 16} vs the paper's 128 (design and launch in "Experiments").

## Experiments

- [ ] **LoRA rank sweep** (`configs/rank_sweep/sweep.json`, code in `colm/jobs`, `colm/eval`).
      Single GPU, scaled down, 1 seed, to see the trend first; **not comparable to the paper's
      4-GPU numbers**. Recipe = `configs/math_phi2_efficient.json` defaults with 1024 steps, efficient
      MeZO, per-device bs 4, GAS 8, `small_batch_ratio` 0.5 (pool 32, 16 trained per step), seed 0,
      W&B off, lr 2e-5 linear, warmup 0.03, lora dropout 0.05, targets q/k/v/fc1/fc2, phi-2 fp32
      weights + fp16 AMP. Arms (r, alpha): (128, 512) paper baseline, (16, 64), (16, 512), (8, 32),
      (8, 512), i.e. alpha/r = 4 and fixed alpha = 512 both. The ZO feature is `lora_B` of the last
      v_proj (2560 x r values), `zo_dim` 2560 <= 20480 also for r = 8 (no path hardcodes r = 128).
      Deviations from the paper recipe (identical for all arms): 1000 MathInstruct examples are held
      out from training (`holdout_size`, seed 0; training set 261,039 instead of 262,039 examples, so
      the data order differs from the paper too); checkpoints are model-only (`save_only_model`) at
      steps 512 and 1024.
      Metrics per arm: train loss every step; eval loss (token-pooled NLL) on the held-out set and on
      the GSM8K test solutions at steps 256/512/768/1024 (in-training callback), plus a standalone
      recomputation on checkpoint-512/1024 and on the base model (`base_eval_loss.json`);
      accuracy on gsm8k / math / numglue / svamp / deepmind / simuleq for checkpoint-512 and 1024
      (vLLM, PoT + CoT backup, 0-shot); step time, training peak memory, trainable params.
      Launch (queue is generated once, the worker takes the GPU explicitly):
      `python -m colm.jobs.rank_sweep --sweep configs/rank_sweep/sweep.json --queue queues/rank-sweep`
      then `python -u -m colm.jobs.worker --queue queues/rank-sweep --gpu <id>` in tmux window
      `CoLM-rank-sweep`. Job order: base eval loss, then per arm train -> eval loss -> eval accuracy,
      then a CPU summary (`out/rank-sweep/summary.{json,md}`). Results/logs: `out/rank-sweep/<run>/`,
      `logs/rank-sweep-<job>-r<r>-a<alpha>-<steps>steps-seed<seed>-<timestamp>.log`.
      Estimate (baseline 2868 ms/step): ~49 min training + 4 in-training evaluations x 26 s (measured:
      13.7 s held-out + 11.8 s GSM8K) + ~1 min load, ~20 s standalone loss job, vLLM accuracy job
      ~2.5 min engine start + PoT/CoT generation and program execution for ~10k prompts per
      checkpoint (execution alone ~43 ms x 10k = 7 min; total not yet measured, guess 15-30 min per
      arm for both checkpoints). About 1.3 h per arm, ~6.5-7 h for the queue (drop checkpoint 512
      from `eval.checkpoints` to save ~10 min per arm). Read training peak memory from
      `summary.md`, not from `train_results.json` (evaluations reset the CUDA peak counter).
      Smoke test of the whole chain (passed 2026-09-28 on GPU 0: train -> checkpoints -> eval loss
      -> vLLM LoRA accuracy -> summary): `configs/rank_sweep/smoke.json` (6 steps, 20 examples);
      in-training and standalone eval loss agree to 4 digits.

## Refactor (hacks to remove)

- [ ] `custom_phi.py` re-implements the Phi forward split; Phi-only, layer index 31 default.
- [ ] Architecture-specific string checks (`"phi-2" in model_name` for LoRA targets / precision).
- [ ] MeZO perturbs `param.data` in place and re-seeds the global RNG (`torch.manual_seed`) on every
      estimate; use a dedicated `torch.Generator` and a functional / out-of-place perturbation
      (must keep the same z and the same training RNG stream, or be flagged as a change).
- [ ] Placeholder micro-batches so the HF loop's accumulation count / DDP sync line up.
- [ ] Logged loss divided by `small_batch_ratio` (kept for curve comparability).
- [ ] `keep_sources` as an underscore-separated string of source indices; sources as ints.
- [ ] fp32 base weights + fp16 AMP for phi-2, `embed_tokens` / `lm_head` `.float()` cast
      (a workaround for fp16 unscale errors).
- [ ] Many selection units / trainers (`rep`, `masked_grad`, `completion_length`, …) around one paper
      method; decide which are needed for the paper's tables before pruning.
- [ ] Attention implementation switched per phase on the shared model config (`_set_attention`,
      restored at step end by a callback) because selection (fp32, `colm_varlen`) and training
      (fp16, flash) need different kernels; a per-call choice would be cleaner.
- [ ] Optimisation flags (`skip_unused_features`, `zo_packing`, `train_packing`, …) keep the original
      paths alive for A/B; drop the original paths once signed off.

## Decisions pending (user)

- Training rows: the default `train_pack_max_tokens=1024` keeps the padded path's peak memory
  (26.3 vs 26.5 GB, 1402 ms/step); `0` (one forward per step) is 1257 ms but 53.9 GB, and CoLM's
  claim is memory. Keep 1024 or switch (`docs/optimization-backlog.md`, "Exact optimisations").
- fp16-AMP training gradients of phi-2 are only ~0.8 cosine-similar to fp32 (F9); upstream recipe,
  unchanged. Decide whether that matters for the paper numbers (bf16 / fp32 attention is a
  precision change).
- The rank sweep started from `c397142` runs the original step path; its later jobs switch to the
  new defaults if the main checkout is pulled before the sweep ends (the worker records the commit
  but does not check it). Pull only after the sweep, or pin the original flags
  (`configs/timing_phi2_efficient_original.json` lists them) in the sweep's base config.

- D1–D4 in `docs/optimization-backlog.md` (F2 divisor, selection precision, dropout for activation
  reuse, 1-D facility location).

## Cleanup pending confirmation

- Smoke / timing outputs: `_worktrees/CoLM-original/out/default` (2.6G, local NVMe);
  `/mnt/data2/zelin4593/artifacts/CoLM/out/*-5steps-*`, `*-130steps-*`, `*-60steps-*`;
  the aborted fine timing log + jsonl and the first failed smoke log in `logs/`.

## Done

- 2026-09-28 Exact step optimisations H, O1, O5, O6, O7, O8 (`COMMIT_HASH`): 2860 → 1402 ms/step
  (2.04×) at the default (training rows ≤ 1024 tokens, peak 26.3 GB), 1257 ms (2.28×) with one
  training forward per step (53.9 GB); selection 1870 → 810 ms, training 908 → 545 / 400 ms.
  All behind flags (`configs/timing_phi2_efficient_original.json` = original path, re-timed at
  2860 ms). Exactness evidence, attention findings on sm_120 and the per-item table:
  `docs/optimization-backlog.md` ("Exact optimisations").

- 2026-09-28 Port to uv + transformers 5.17 + stock vLLM (`113005a`); unported baseline env on
  branch `task/original-env` (`96673d0`); per-phase step timing (`69a9235`, baseline 2868 ms/step).
