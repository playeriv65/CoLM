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
  `training_step` (documented extension point): features -> gather -> rank-0 selection ->
  broadcast -> forward/backward of the selected sub-batches with `accelerator.no_sync` for all
  but the last. Optimizer step, clipping, scheduler, logging, checkpointing are stock. Do not
  copy HF loop internals back in or touch private HF attributes.
- `colm/selection/` — `features.py` (one extractor per `data_selection_unit`, on packed batches),
  `zo.py` (`Perturbation`: fixed-seed z, out-of-place +-eps through `functional_call`;
  `LastLayerSplit`: the model's own forward stopped by a pre-hook on the last layer, then the
  last layer replayed; works for any decoder), `select.py` (`CoresetSelector`: keep sources,
  transform, Adam, coordinate mask, facility location; keeps the Adam moments),
  `facility_location.py`, `packing.py` (padding-free batches), `batching.py` (packed batching, the
  step loss), `pool.py` (collectives that are no-ops in one process).
- `colm/train/attention.py` — `colm_varlen` attention for packed rows (registered with
  `AttentionInterface`; the default `attn_implementation`). `colm/train/memory.py` — peak memory
  of every rank per phase. `colm/train/step_timing.py` — opt-in per-phase step timer and its
  summariser (`--profile_timing coarse|fine`; keep new timing sections behind `timer.section`).
- `colm/train/config.py` (JSON + flags, model recipe from `configs/model_profiles.json`, resolved
  config), `training_arguments.py` / `model_arguments.py` / `data_arguments.py` — all options; the
  defaults are the paper recipe with the corrections of `docs/errors.md`.
- `colm/data/` — datasets and collators (`get_training_dataset.py`, `superglue.py`, `holdout.py`,
  `tasks.py`, `templates.py`); `colm/cli.py` — the entry points.
- `colm/eval/` — teacher-forced eval loss (`eval_loss.py`: sets, `evaluate_loss`, trainer callback,
  CLI); `colm/jobs/` — file queue + the single worker, sweep expansion (`rank_sweep.py`, specs in
  `configs/rank_sweep/`) and `summarize.py`. Queues live in the gitignored `queues/`.
- `math_eval/` (vLLM / HF generation; `run_open.py` takes several models/datasets per process),
  `superglue_eval/` — evaluation code behind `colm-eval`.
- `tests/equivalence/` — the tiny float64 Phi, tokenizer and data used by the tests.

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
- The packed selection forward must stay fp32 (eps 1e-3: fp16 features are noise) and packed
  inputs need `use_cache=False` and an attention implementation that reads `cu_seq_lens_q`.

## Optimisation work

- Stopped for now. Backlog and findings: `docs/optimization-backlog.md` (numbers there refer to
  the padded upstream path); the unfinished exact-optimisation branch is `task/exact-opts`.
