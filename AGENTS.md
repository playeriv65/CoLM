# CoLM — agent notes

ICLR 2025 "Mini-batch Coresets for Memory-efficient LM Training on Data Mixtures" (LESS + MeZO
based), ported to transformers 5.x. This fork is a **public** GitHub repo: never commit secrets,
tokens, W&B keys or machine-private data.

## Environment

- `uv sync` builds `.venv` (Python 3.12, torch 2.13.0+cu130 from the PyTorch cu130 index,
  transformers 5.x, peft, accelerate, submodlib from git). Extras: `--extra eval` (vLLM 0.30,
  pinned to the same torch), `--extra wandb`. Lock file `uv.lock` is committed; upgrade with
  `uv lock --upgrade` and re-run the tests.
- The uv cache comes from `UV_CACHE_DIR` (machine env), not from pyproject.
- HF caches must point at the local NVMe (`HF_HOME`), never at a network disk. Pass no
  `cache_dir` in configs unless overriding.
- After a fresh clone / new worktree run `bash scripts/link-external.sh` (data and out symlinks,
  declared in `external-paths.json`; root from `COLM_ARTIFACT_ROOT`).

## Entry points

`colm-train` (1 / 2 / N GPUs with `--gpus`, torchrun underneath), `colm-eval loss|accuracy|superglue`,
`colm-sweep create|work|summary`; all print `--help`. README "Quickstart" has the commands. There
are no other launch scripts: do not add shell wrappers, extend the entry points.

## Code map

- `colm/train/trainers.py` — thin subclasses of the HF 5.x `Trainer`: `CustomTrainer` (full-batch
  baseline), `SubsetTrainer` (one example per forward, every selection unit) and
  `SubsetTrainerEfficient` (batched last-layer MeZO, the paper's method). The selection pool of a
  step (`per_device_train_batch_size x gradient_accumulation_steps` examples) is ONE HF batch
  (`TrainingArguments` sets HF's gradient accumulation to 1); the coreset trainers override only
  `training_step` (documented extension point): plan (gather the pool, which features are needed)
  -> features of the needed examples (a token-balanced share per rank) -> gather (scalars g_i for
  MeZO) -> rank-0 selection -> broadcast -> forward/backward of the selected examples in packs of
  `batching.train_tokens` (= `train_max_tokens`, a plain token budget; `accelerator.no_sync` for all
  but the last). Optimizer step, clipping, scheduler, logging, checkpointing are stock. Do not
  copy HF loop internals back in or touch private HF attributes.
  `colm/train/preflight.py` — checks before a run or resume (a complete checkpoint, `save_only_model` warning,
  free disk for the planned checkpoints); `colm/train/frozen_weights.py` — frozen fp32 Linear weights stored in
  fp16 where nothing needs them in fp32 (`frozen_base_low_precision`, same numbers, `docs/system-audit.md`).
  `colm/train/selection_state.py` — the selector's Adam moments go into every `checkpoint-N/selection_state.pt`
  (rank 0, `on_save`) and come back in `CoresetTrainer.train(resume_from_checkpoint=...)` (`docs/errors.md` E12).
- `colm/selection/` — `features.py` (one extractor per `data_selection_unit`, on packed batches;
  `extract` -> per-example values, `expand` -> features; the MeZO extractor returns g_i),
  `zo.py` (`Perturbation`: fixed-seed z, out-of-place +-eps through `functional_call`;
  `LastLayerSplit`: the model's own forward stopped by a pre-hook on the last layer, then the
  last layer replayed; works for any decoder), `select.py` (`CoresetSelector`: `needed` = which
  features can matter, keep sources, transform, Adam, coordinate mask, facility location; keeps
  the Adam moments), `facility_location.py` (`class_budgets` = the per-source quotas),
  `packing.py` (padding-free batches with the label geometry computed on the CPU, `balanced_shares`),
  `batching.py` (packed batching, the step loss), `pool.py` (collectives that are no-ops in one
  process).
- Attention is stock transformers (no custom kernel): packed rows carry `position_ids` and the
  flash cumulative lengths (`packing.model_inputs`); the training forward runs
  `attn_implementation` (`flash_attention_2`, recipe default in `configs/model_profiles.json`; the
  hub kernel through `kernels` when `flash-attn` is not installed) and the no-grad selection
  forward `selection_attn_implementation` (`sdpa`, dense block mask; fp16 prefix with an fp32 tail of two blocks and fp32
  suffix for the Phi-2 recipe); the trainer switches with
  `model.set_attn_implementation` (`_Trainer.set_attention`). `colm/train/memory.py` — peak memory
  of every rank per phase (`MemoryMeter`).
  `colm/train/step_timing.py` — opt-in per-phase step timer and its summariser
  (`--profile_timing coarse|fine`; keep new timing sections behind `timer.section`).
- `colm/train/config.py` (JSON + flags, model recipe from `configs/model_profiles.json`, resolved
  config), `training_arguments.py` / `model_arguments.py` / `data_arguments.py` — all options; the
  defaults are the paper recipe with the corrections of `docs/errors.md`.
- `colm/data/` — datasets and collators (`get_training_dataset.py`, `superglue.py`, `holdout.py`,
  `tasks.py`, `templates.py`); `colm/cli.py` — the entry points. `TokenCountCache`
  (`get_training_dataset.py`) keeps the per-example token counts that decide which examples fit
  the context window (`--token_cache_dir`, default `cache/tokens`; key = hash of texts + context
  limit + tokenizer; `cache` is an external link, see `external-paths.json`).
- `colm/phases.py` + `colm/train/phase_callback.py` — one-shot wall-clock phases of a run,
  saved as `<output_dir>/startup.json` (no synchronisation, nothing per step); wrap new start-up work
  in `CLOCK.mark(...)` / `CLOCK.detail(...)`. Results: `docs/startup-overhead.md`.
- `colm/eval/` — teacher-forced eval loss (`eval_loss.py`: sets, `evaluate_loss`, trainer callback,
  CLI); `colm/jobs/` — file queue + the single worker, sweep expansion (`rank_sweep.py`, specs in
  `configs/rank_sweep/`) and `summarize.py`. Queues live in the gitignored `queues/`. The worker
  stops on the first failure. Sweep kernel requirements are config-driven: a pinned hub snapshot
  is resolved from the local HF cache, loaded offline before the queue starts and passed to jobs
  through `LOCAL_KERNELS`.
- `math_eval/` (vLLM / HF generation; `run_open.py` takes several models/datasets per process),
  `superglue_eval/` — evaluation code behind `colm-eval`.
- `tests/equivalence/` — the tiny float64 Phi, tokenizer and data used by the tests.
- `scripts/diagnostics/`, `configs/diagnostics/` — measurement scripts and their inputs; each script
  header says what it measured and which document holds the result (`docs/README.md`). They are not
  part of the library: nothing in `colm/` or the tests imports them (except `tests/test_precision_arms.py`).

## Rules

- GPUs are shared and reserved per period: nothing defaults to a GPU list; pass `--gpus` / `--gpu`
  or `COLM_GPUS`. Tests run on CPU: `CUDA_VISIBLE_DEVICES="" uv run pytest -q` (includes 2- and
  4-rank gloo runs).
- W&B is opt-in (`--report_to wandb`); with it off nothing may import `wandb` (tested).
- Before committing: `uv run ruff format . && uv run ruff check . && CUDA_VISIBLE_DEVICES="" uv run pytest -q`.
- The upstream behaviour is not kept as an option: the tag `pre-refactor` is the upstream code and
  `legacy-bridge` the last commit with the temporary `legacy` switch and its float64 goldens
  (`tests/test_equivalence.py`); `docs/errors.md` lists every error and its fix. The default path
  must contain no known error.
- Exactness is float64 identity on CPU: in fp32 the MeZO feature is decided at rounding level
  (a fp32 run of the upstream code agrees with itself in 3 of 20 steps), so end-to-end fp32 runs
  are compared statistically (selection overlap against that noise floor), not bitwise.
- Nothing truncates and no option may reintroduce it (a test greps for it): the only sequence
  limit is the model's context window (`colm.train.config.context_length`); examples above it are
  dropped and counted per source (`PromptTooLong` in SuperGLUE evaluation). Memory is controlled
  by how many whole examples go into a packed forward (`train_max_tokens`, `pack_tokens`).
  `max_length_q/k` (flash varlen kwargs) and vLLM/HF `max_new_tokens` are not truncation.
- The Phi-2 model profile sets `selection_prefix_dtype=float16` and
  `selection_prefix_fp32_tail=2`: the unperturbed prefix (layers 0-30) uses fp16
  autocast except its last two blocks, which run in fp32 (a forward pre-hook in
  `LastLayerSplit._fp32_tail`, removed when the prefix returns); it requires fp32
  model weights, while the perturbed last layer and loss remain fp32. The tail
  is 0 for other profiles and whenever the prefix is explicitly `float32`
  (a tail with a non-fp16 prefix is an error); tail = all prefix layers equals
  the fp32 prefix, tail 0 the plain fp16 prefix. CLI/JSON overrides take
  precedence; the resolved configuration records the choice. The plain fp16
  prefix changed selected sets beyond the fp32 packing noise; the 2-block tail
  brings g_i to a median 1.2 % error (`docs/selection-precision.md`,
  `docs/fp16-prefix.md`); learning quality has not yet been compared. Packed
  inputs need `use_cache=False` (a cache ends the packed-batch detection of
  transformers).
- The Phi-2 profile sets `pack_tokens=1536` for selection. This is distinct
  from `train_max_tokens=1536`; JSON or CLI values override either budget.
  `--pack_tokens 0` restores the data-derived selection budget. The one-run
  timing comparison and its machine-load caveat are in `docs/fp16-prefix.md`.

- Multi-rank hygiene: every rank must issue the same collectives in the same order, so a callback that
  uses one (`EvalLossCallback`) runs on all ranks and only rank 0 writes; `Trainer.log` inside a callback
  clears `control.should_log` (restore it, or that rank skips the step's loss gather). Host-side batch
  building is numpy only (`packing.pack`): tiny torch CPU ops run on the intra-op thread pool and show
  tens-of-milliseconds tails on a loaded host. Multi-rank behaviour is tested with gloo runs
  (`tests/test_distributed.py`); anything that touches a collective needs a case there.

## Optimisation work

- The execution-only optimisations are done and are the code path (no switches): backlog, measured
  tables and the candidates left (O11-O13) are in `docs/optimization-backlog.md`. Keep them exact:
  `tests/test_opt.py` compares every one with the plain computation in float64 on CPU,
  `scripts/diagnostics/check_opt.py` does the phi-2 teacher-forced comparison on a GPU (selection overlap
  against the fp32 noise floor, gradients against an exact fp32 reference).
- fp32 gradient references use `sdpa_kernel(SDPBackend.MATH)`: the fp32 memory-efficient SDPA
  backward is ~0.3 off on phi-2 on this GPU (`docs/errors.md`).
- Timing follows the protocol in the backlog (`--profile_timing fine`, 130 steps, 10 warm-up,
  closure, GPU alone, loadavg recorded); a step's time is 0.154 ms per forwarded token + 0.130 ms
  per trained token, so compare runs on the same pools.
