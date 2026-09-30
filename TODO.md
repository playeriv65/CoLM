# TODO

Short, current list. Finished work goes to "Done" with a pointer and is deleted once it is recorded in
a document (`docs/README.md` is the index); measurements live in `docs/`, not here.

## Running

- Nothing. (Updated 2026-09-29 after the wrap-up acceptance run, `docs/startup-overhead.md`.)

## Next

- [ ] **LoRA rank sweep v5** (design below). The queue is not created yet: `colm-sweep create --sweep
      configs/rank_sweep/sweep.json --queue queues/rank-sweep-v5`, then `colm-sweep work --queue
      queues/rank-sweep-v5 --gpu <id>` in tmux window `CoLM-rank-sweep`. The GPU is assigned by the
      user. v5 = FP16 selection prefix with an FP32 tail of 2 blocks (the Phi-2 default). The v4
      queue/outputs are the plain FP16 prefix (tail 0), a different arm: do not mix or resume them.
- [ ] Make `math_eval` report loss as well as accuracy (accuracy only today).
- [ ] System audit follow-ups (`docs/system-audit.md`, prioritised list at its end): measure a real 2/4-GPU
      run (needs GPUs assigned) to confirm the estimate for the paper setting and the eval sharding.
- [ ] Optimisation: profile the current fp16-prefix step (launch overhead, training forward/backward)
      before choosing another kernel change; O11 is an optional ~20 ms architecture-specific change
      (`docs/optimization-backlog.md`). Start-up is settled (`docs/startup-overhead.md`).
- [ ] `superglue_eval/eval_superglue.py` accepts `--dtype` but loads with `dtype="auto"`: honour it
      or drop it (changes what `colm-eval superglue` loads by default, so it needs a decision).

## Decisions pending (user)

- Precision of the recipe (fp16 attention gradients, `docs/errors.md`); D3-D4 of
  `docs/optimization-backlog.md` (dropout for activation reuse, 1-D facility location).
  Decided: D1 (mean over the example's label tokens), D2 (selection precision: fp16 prefix, fp32 tail
  of 2, `docs/selection-precision.md`).
- Whether the queue keeps the standalone `evalloss` job of every arm: the in-training evaluation
  already gives the same numbers at steps 512 / 1024 (agreement to 6e-5 was checked); the job
  costs ~1.3 min per arm (was 1.6 min before the token cache).

## Cleanup pending confirmation

Nothing below is deleted without the user's yes.

- Worktrees: `_worktrees/CoLM-exact-opts` (unfinished WIP branch, superseded by the merged
  optimisations), `CoLM-layer-signal` (unmerged; `docs/fp16-prefix.md` cites its
  `docs/layer-signal.md`, which is not on this branch), `CoLM-rank-sweep-ops` and
  `CoLM-rank-sweep-refresh` (merged; the latter holds the v3/v4 queues), `CoLM-original`
  (unported baseline environment, `out/default` = 2.6 GB on the local NVMe), `CoLM-opt-base` (not a git worktree).
- Queues: `queues/rank-sweep`, `rank-sweep-eval`, `rank-sweep-v2` (main checkout, all pending,
  never run), `rank-sweep-v3` (12 failed jobs, missing kernel) and `rank-sweep-v4` (6 done, 12
  pending) in `CoLM-rank-sweep-refresh`.
- Artifacts under `/mnt/data2/zelin4593/artifacts/CoLM/`: `out/rank-sweep`, `rank-sweep-v2`,
  `rank-sweep-v3` (1.3 GB), `rank-sweep-v4` (2.2 GB, valid for tail 0), the smoke/timing runs
  `out/*-5steps-*` (2.6 GB), `*-130steps-*`, `*-60steps-*`, `out/stockattn-*`, and the diagnostics
  directories (`precision-*`, `fp16-prefix-*`, `layer-signal-*`, `seed-recheck-*`).
- `$HF_HOME/datasets/json` (local NVMe): ~3,300 Arrow-cache directories (~3 GB), most of them 24 KB
  leftovers of earlier pytest runs (tests now use a temporary datasets cache) and one 189 MB cache per
  distinct absolute path of `MathInstruct.jsonl` (one per worktree that trained).
- Logs in the main checkout `logs/` from 2026-09-28 (aborted timing runs, first failed smoke run).
- The wrap-up run of 2026-09-29: `artifacts/CoLM/wrapup/` (acceptance output + logs, ~2.5 GB with the
  checkpoints) after the numbers in `docs/startup-overhead.md` are no longer needed.

## Rank sweep design (`configs/rank_sweep/sweep.json`, code in `colm/jobs`, `colm/eval`)

Single GPU, scaled down, 1 seed, to see the trend first; **not comparable to the paper's 4-GPU
numbers**. Recipe = the defaults of `colm-train` (paper recipe), 1024 steps, efficient MeZO, per-device
bs 4, GAS 8, `small_batch_ratio` 0.5 (pool 32, 16 trained), seed 0, lr 2e-5 linear, warmup 0.03,
LoRA dropout 0.05, targets q/k/v/fc1/fc2, phi-2 fp32 weights + fp16 AMP, W&B off. Arms (r, alpha):
(128, 512) paper baseline, (16, 64), (16, 512), (8, 32), (8, 512): alpha/r = 4 and fixed alpha = 512.
The ZO feature is `lora_B` of the last v_proj (2560 x r values, `zo_dim` 2560 for every r).
Deviations from the paper (identical for all arms): 1000 MathInstruct examples held out
(`holdout_size`, seed 0; 261,008 training examples after the 31 over-context ones are dropped), model-only checkpoints at steps 512 and 1024.

Metrics per arm: train loss every step; eval loss (token-pooled NLL) on the held-out set and the GSM8K
test solutions at steps 256/512/768/1024 (in-training callback), a standalone recomputation on
checkpoint-512/1024 and on the base model; accuracy on gsm8k / math / numglue / svamp / deepmind /
simuleq for both checkpoints (vLLM, PoT + CoT backup, 0-shot); step time, peak memory per phase.
Queue order: base eval loss, per arm train -> eval loss -> eval accuracy, base accuracy, CPU summary
(`out/rank-sweep-v5/summary.{json,md}`). Logs: `logs/rank-sweep-<job>-...-<timestamp>.log`.

Wall clock per job (v4, GPU alone, before the token cache): train ~17-18 min, standalone eval loss
1.6 min, accuracy ~11-18 min (engine start 73 s), so ~0.6 h per arm and ~3 h for the queue; the
measured phases of a train job are in `docs/startup-overhead.md`. Smoke test of the whole chain
(`configs/rank_sweep/smoke.json`, 6 steps, 20 examples): train -> checkpoints -> eval loss -> vLLM LoRA
accuracy passed on 2026-09-28; on 2026-09-29 the same chain (1024-step train, standalone loss, vLLM
accuracy, SuperGLUE) was run by hand on GPU 2 (`docs/startup-overhead.md`).

## Done

- 2026-09-29 System audit: eval-callback collective mismatch fixed, sharded eval loss, token-balanced training
  shares, numpy `pack`, fp16-stored frozen weights, resume / disk / launcher preflight (`docs/system-audit.md`).

- 2026-09-29 Selection Adam moments saved with every checkpoint and restored on resume (`docs/errors.md` E12,
  `colm/train/selection_state.py`, `tests/test_trainers.py`).
- 2026-09-29 Wrap-up: token-count cache and `startup.json` phase record (`docs/startup-overhead.md`),
  SuperGLUE tasks load again (namespaced hub ids), `colm-sweep create --dry-run`, dead code removed,
  diagnostics moved to `scripts/diagnostics/`, docs index (`docs/README.md`).
- 2026-09-29 Selection precision: fp16 prefix with an fp32 tail of 2 blocks is the Phi-2 default
  (`docs/fp16-prefix.md`, `docs/selection-precision.md`).
- 2026-09-28 Execution-only step optimisations (skip unused MeZO forwards, gather g_i scalars, packed
  training forwards, host overhead): 2317 -> 1305 ms/step on phi-2 with the fp32 prefix, same
  selections up to rounding (`docs/optimization-backlog.md`, `scripts/diagnostics/check_opt.py`,
  `tests/test_opt.py`).
- 2026-09-28 Refactor: HF-native trainers, `colm/selection`, packed padding-free batches, per-rank peak
  memory, entry points, the paper recipe as default, upstream errors fixed (`docs/errors.md`), goldens
  of the upstream code in the tag `legacy-bridge`.
- 2026-09-28 Port to uv + transformers 5.17 + stock vLLM (`113005a`); unported baseline environment on
  branch `task/original-env` (`96673d0`); per-phase step timing (`69a9235`).
