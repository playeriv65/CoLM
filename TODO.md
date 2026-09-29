# TODO

Living task list for this research repo. Keep it short: move finished items to "Done" with the
commit / result pointer, delete them once they are recorded in docs. Optimisation details live in
`docs/optimization-backlog.md`; this file only tracks what is next.

## In progress

- [ ] Nothing in progress: the refactor is merged (`legacy` switch removed; tags `pre-refactor`
      and `legacy-bridge`).

## Next

- [ ] LoRA rank sweep: r in {8, 16} vs the paper's 128. **Blocked** until the refactor is merged;
      runs with the fixed code, i.e. with the fixes E4 (tokenisation, no
      truncation), E8 (question-grouped held-out set) and E9 (eval scoring) in `docs/errors.md`.
      The queue was regenerated against the fixed code (`queues/rank-sweep-v2`, outputs in
      `out/rank-sweep-v2`); the pre-refactor queue `queues/rank-sweep` and `out/rank-sweep` (old
      tokenisation, holdout and scoring) must not be reused (deletion pending confirmation). The GPU is assigned by the user.
- [ ] Precision decisions with evidence in `docs/errors.md` (fp16 attention gradients; fp32
      selection forward must stay), not switched without a decision.
- [ ] Make evaluation report loss as well as accuracy in `math_eval` (accuracy only today).
- [ ] Optimisation (stopped): `docs/optimization-backlog.md`; the features could be gathered as
      32 scalars, the base weights stored once (fp32 + per-forward fp16 copies today).

## Experiments

- [ ] **LoRA rank sweep** (`configs/rank_sweep/sweep.json`, code in `colm/jobs`, `colm/eval`).
      Single GPU, scaled down, 1 seed, to see the trend first; **not comparable to the paper's
      4-GPU numbers**. Recipe = the defaults of `colm-train` (`configs/math_phi2_efficient.json`), 1024 steps, efficient
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
      `colm-sweep create --sweep configs/rank_sweep/sweep.json --queue queues/rank-sweep-v2`
      then `colm-sweep work --queue queues/rank-sweep-v2 --gpu <id>` in tmux window
      `CoLM-rank-sweep`. Job order: base eval loss, then per arm train -> eval loss -> eval accuracy,
      then a CPU summary (`out/rank-sweep-v2/summary.{json,md}`). Results/logs: `out/rank-sweep-v2/<run>/`,
      `logs/rank-sweep-<job>-r<r>-a<alpha>-<steps>steps-seed<seed>-<timestamp>.log`.
      Estimate (baseline 2868 ms/step): ~49 min training + 4 in-training evaluations x 26 s (measured:
      13.7 s held-out + 11.8 s GSM8K) + ~1 min load, ~20 s standalone loss job, vLLM accuracy job
      ~2.5 min engine start + PoT/CoT generation and program execution for ~10k prompts per
      checkpoint (execution alone ~43 ms x 10k = 7 min; total not yet measured, guess 15-30 min per
      arm for both checkpoints). About 1.3 h per arm, ~6.5-7 h for the queue (drop checkpoint 512
      from `eval.checkpoints` to save ~10 min per arm). Training peak memory (maximum over the ranks,
      per phase) is in `memory.json` / `summary.md`.
      Smoke test of the whole chain (passed 2026-09-28 on GPU 0: train -> checkpoints -> eval loss
      -> vLLM LoRA accuracy -> summary): `configs/rank_sweep/smoke.json` (6 steps, 20 examples);
      in-training and standalone eval loss agree to 4 digits.

## Decisions pending (user)

- Precision of the recipe (fp16 attention gradients, `docs/errors.md`); D2-D4 of
  `docs/optimization-backlog.md` (selection precision, dropout for activation reuse, 1-D facility
  location). D1 (padded divisor) is decided: mean over the example's label tokens.

## Cleanup pending confirmation

- Smoke / timing outputs: `_worktrees/CoLM-original/out/default` (2.6G, local NVMe);
  `/mnt/data2/zelin4593/artifacts/CoLM/out/*-5steps-*`, `*-130steps-*`, `*-60steps-*`;
  the aborted fine timing log + jsonl and the first failed smoke log in `logs/`.

## Done

- 2026-09-28 Refactor: HF-native trainers (one pool = one HF batch), `colm/selection`, packed
  padding-free batches, per-rank peak memory, entry points and config with the paper recipe as
  default, errors fixed (`docs/errors.md`), goldens of the upstream code (`tests/equivalence`).
- 2026-09-28 Port to uv + transformers 5.17 + stock vLLM (`113005a`); unported baseline env on
  branch `task/original-env` (`96673d0`); per-phase step timing (`69a9235`, baseline 2868 ms/step).
